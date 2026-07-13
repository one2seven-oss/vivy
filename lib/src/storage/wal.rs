//! Write-Ahead Log for crash safety.
//!
//! Every insert/delete goes to the log *before* the in-memory delta.
//! Replay on restart restores the last committed state after a crash.
//!
//! Design:
//! - **Append-only**: entries always at EOF, no seeks, no overwrites.
//! - **BufWriter + sync_all**: buffered writes for throughput, fsync on commit.
//! - **Tag-length-value**: 1-byte tag + fixed-size fields + variable payload.
//!   Partial final entries (kill -9 mid-write) are detected during replay
//!   by EOF on read_exact — data before the last successful write is intact.
//! - **Flush checkpoint** (tag 0xFF): marks a point where the delta has been
//!   sealed, so entries before it can be skipped on replay (WAL truncation).

use log::warn;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WalError {
    #[error("I/O: {0}")]
    Io(#[from] io::Error),
    #[error("corrupt entry at offset {0}")]
    Corrupt(u64),
}

// WAL entry types. Insert carries the full vector for additive replay.
// Delete carries only the ID. Flush is a checkpoint marker.
#[derive(Debug, Clone)]
pub enum WalEntry {
    Insert { id: u64, vector: Vec<f32> },
    Delete { id: u64 },
    Flush,
}

// Append-only, crash-safe, single-writer WAL. Tracks committed byte offset for truncation.
pub struct WalWriter {
    file: BufWriter<File>,
    _path: Box<Path>,
    committed: u64,
}

impl WalWriter {
    // Open/create in append+read mode. `committed` = current file length
    // (bytes from a previous session are already durable).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        let committed = file.metadata()?.len();
        Ok(Self {
            file: BufWriter::new(file),
            _path: path.into(),
            committed,
        })
    }

    // Buffered write. Not durable until commit().
    // Encoding: Insert=0x01|id(8)|dim(4)|vector(dim*4), Delete=0x02|id(8), Flush=0xFF.
    pub fn append(&mut self, entry: &WalEntry) -> Result<(), WalError> {
        match entry {
            WalEntry::Insert { id, vector } => {
                self.file.write_all(&[0x01])?;
                self.file.write_all(&id.to_le_bytes())?;
                let dim = vector.len() as u32;
                self.file.write_all(&dim.to_le_bytes())?;
                let bytes: &[u8] = bytemuck::cast_slice(vector.as_slice());
                self.file.write_all(bytes)?;
            }
            WalEntry::Delete { id } => {
                self.file.write_all(&[0x02])?;
                self.file.write_all(&id.to_le_bytes())?;
            }
            WalEntry::Flush => {
                self.file.write_all(&[0xFF])?;
            }
        }
        Ok(())
    }

    // Flush BufWriter + fsync. After this returns, entries survive a crash.
    pub fn commit(&mut self) -> Result<(), WalError> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        self.committed = self.file.get_ref().metadata()?.len();
        Ok(())
    }

    // Replay from start, calling `f` for each decoded entry.
    // Stops at first corrupt/truncated entry (everything before is intact).
    // Missing file = nothing to replay (clean shutdown).
    pub fn replay(path: impl AsRef<Path>, mut f: impl FnMut(WalEntry)) -> Result<(), WalError> {
        let file = match File::open(path.as_ref()) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut off = 0u64;
        while off < len {
            let mut tag_buf = [0u8; 1];
            if reader.read_exact(&mut tag_buf).is_err() {
                warn!("WAL replay: truncated at offset {}, stopping", off);
                return Err(WalError::Corrupt(off));
            }
            off += 1;
            match tag_buf[0] {
                0x01 => {
                    let mut id_buf = [0u8; 8];
                    reader.read_exact(&mut id_buf)?;
                    let id = u64::from_le_bytes(id_buf);
                    let mut dim_buf = [0u8; 4];
                    reader.read_exact(&mut dim_buf)?;
                    let dim = u32::from_le_bytes(dim_buf) as usize;
                    let mut vec_bytes = vec![0u8; dim * 4];
                    reader.read_exact(&mut vec_bytes)?;
                    let vector: Vec<f32> = bytemuck::cast_slice(&vec_bytes).to_vec();
                    off += 8u64 + 4 + (dim * 4) as u64;
                    f(WalEntry::Insert { id, vector });
                }
                0x02 => {
                    let mut id_buf = [0u8; 8];
                    reader.read_exact(&mut id_buf)?;
                    let id = u64::from_le_bytes(id_buf);
                    off += 8;
                    f(WalEntry::Delete { id });
                }
                0xFF => {
                    f(WalEntry::Flush);
                }
                tag => {
                    warn!("WAL replay: unknown tag {:#04x} at offset {}, stopping", tag, off - 1);
                    return Err(WalError::Corrupt(off - 1));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_wal_roundtrip() {
        let dir = std::env::temp_dir().join("vivy-wal-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.log");

        {
            let mut wal = WalWriter::open(&path).unwrap();
            wal.append(&WalEntry::Insert { id: 1, vector: vec![1.0, 2.0, 3.0] }).unwrap();
            wal.append(&WalEntry::Delete { id: 2 }).unwrap();
            wal.append(&WalEntry::Flush).unwrap();
            wal.commit().unwrap();
        }

        let mut entries = Vec::new();
        WalWriter::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 3);
        match &entries[0] {
            WalEntry::Insert { id, vector } => {
                assert_eq!(*id, 1);
                assert_eq!(vector, &[1.0, 2.0, 3.0]);
            }
            _ => panic!("expected Insert"),
        }
        match &entries[1] {
            WalEntry::Delete { id } => assert_eq!(*id, 2),
            _ => panic!("expected Delete"),
        }
        match &entries[2] {
            WalEntry::Flush => {}
            _ => panic!("expected Flush"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
