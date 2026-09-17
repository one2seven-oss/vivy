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

///single recorded operation in the WAL.
#[derive(Debug, Clone)]
pub enum WalEntry {
    Insert { id: u64, vector: Vec<f32> },
}

/// append-only, crash-safe.
pub struct WalWriter {
    file: BufWriter<File>,
    #[allow(dead_code)]
    path: Box<Path>,
    committed: u64,
}

impl WalWriter {
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
            path: path.into(),
            committed,
        })
    }

    pub fn append(&mut self, entry: &WalEntry) -> Result<(), WalError> {
        let WalEntry::Insert { id, vector } = entry;
        self.file.write_all(&[0x01])?;
        self.file.write_all(&id.to_le_bytes())?;
        let dim = vector.len() as u32;
        self.file.write_all(&dim.to_le_bytes())?;
        let bytes: &[u8] = bytemuck::cast_slice(vector.as_slice());
        self.file.write_all(bytes)?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<(), WalError> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        self.committed = self.file.get_ref().metadata()?.len();
        Ok(())
    }

    /// Truncate WAL to 0 bytes after delta segments are compacted.
    pub fn reset(&mut self) -> Result<(), WalError> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .read(true)
            .open(&self.path)?;
        self.file = BufWriter::new(file);
        self.committed = 0;
        Ok(())
    }

    /// Replay all entries from the WAL, calling `f` for each.
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
                tag => {
                    warn!(
                        "WAL replay: unknown tag {:#04x} at offset {}, stopping",
                        tag,
                        off - 1
                    );
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
            wal.append(&WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0],
            })
            .unwrap();
            wal.commit().unwrap();
        }

        let mut entries = Vec::new();
        WalWriter::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 1);
        let WalEntry::Insert { id, vector } = &entries[0];
        assert_eq!(*id, 1);
        assert_eq!(vector, &[1.0, 2.0, 3.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
