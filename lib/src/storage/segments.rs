use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::path::Path;
use thiserror::Error;

const MAGIC: [u8; 8] = *b"VIVYSEG\0";
const CURRENT_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version: u32,
    num_nodes: u32,
    dims: u32,
    m: u32,
    m_max: u32,
    pq_subvectors: u32,
    pq_enabled: u8,
    _reserved: [u8; 31],
}

#[derive(Debug, Error)]
pub enum SegmentError {
    #[error("invalid magic bytes")]
    BadMagic,
    #[error("unsupported version {0}")]
    UnsupportedVersion(u32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("file too small")]
    Truncated,
    #[error("dimension mismatch")]
    DimensionMismatch,
}

#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: u64,
    pub level: u32,
    pub neighbors: Vec<Vec<u32>>,
    pub pq_code: Option<Vec<u8>>,
    pub vector: Option<Vec<f32>>,
}

pub struct SealedSegment {
    mmap: Mmap,
    offset_table_off: usize,
    data_off: usize,
    num_nodes: usize,
    dims: usize,
    #[allow(dead_code)]
    m: usize,
    #[allow(dead_code)]
    m_max: usize,
    pq_enabled: bool,
    pq_subvectors: usize,
    has_codebook: bool,
    codebook_off: usize,
}

impl SealedSegment {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SegmentError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_file(file: std::fs::File) -> Result<Self, SegmentError> {
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self, SegmentError> {
        if mmap.len() < size_of::<Header>() {
            return Err(SegmentError::Truncated);
        }
        // Copy the entire header to avoid borrow on mmap
        let h: Header = bytemuck::pod_read_unaligned(&mmap[..size_of::<Header>()]);
        if h.magic != MAGIC {
            return Err(SegmentError::BadMagic);
        }
        if h.version != CURRENT_VERSION {
            return Err(SegmentError::UnsupportedVersion(h.version));
        }
        let num_nodes = h.num_nodes as usize;
        let dims = h.dims as usize;
        let pq_enabled = h.pq_enabled != 0;
        let pq_subvectors = h.pq_subvectors as usize;
        let m = h.m as usize;
        let m_max = h.m_max as usize;

        let offset_table_off = size_of::<Header>();
        let offset_table_len = num_nodes * size_of::<u64>();
        let mut data_off = offset_table_off + offset_table_len;

        let codebook_off = data_off;
        if pq_enabled {
            data_off += pq_subvectors * 256 * size_of::<f32>();
        }

        let has_codebook = pq_enabled;
        let min_size = data_off + if num_nodes > 0 { 12 } else { 0 };
        if mmap.len() < min_size {
            return Err(SegmentError::Truncated);
        }

        Ok(Self {
            mmap,
            offset_table_off,
            data_off,
            num_nodes,
            dims,
            m,
            m_max,
            pq_enabled,
            pq_subvectors,
            has_codebook,
            codebook_off,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn pq_enabled(&self) -> bool {
        self.pq_enabled
    }

    fn offset_of(&self, idx: usize) -> Result<usize, SegmentError> {
        if idx >= self.num_nodes {
            return Err(SegmentError::Truncated);
        }
        let off = self.offset_table_off + idx * size_of::<u64>();
        let rel = u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap());
        Ok(self.data_off + rel as usize)
    }

    pub fn codebook(&self) -> Option<&[f32]> {
        if !self.has_codebook {
            return None;
        }
        let len = self.pq_subvectors * 256 * size_of::<f32>();
        Some(bytemuck::cast_slice(
            &self.mmap[self.codebook_off..self.codebook_off + len],
        ))
    }

    pub fn read_node(&self, idx: usize) -> Result<NodeRecord, SegmentError> {
        let start = self.offset_of(idx)?;
        let buf = &self.mmap[start..];
        let mut pos = 0usize;

        let read_u32 = |p: &mut usize, b: &[u8]| -> u32 {
            let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v
        };
        let read_u64 = |p: &mut usize, b: &[u8]| -> u64 {
            let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
            *p += 8;
            v
        };

        let id = read_u64(&mut pos, buf);
        let level = read_u32(&mut pos, buf) as usize;

        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            let n = read_u32(&mut pos, buf) as usize;
            let mut layer = Vec::with_capacity(n);
            for _ in 0..n {
                layer.push(read_u32(&mut pos, buf));
            }
            neighbors.push(layer);
        }

        let (pq_code, vector) = if self.pq_enabled {
            let code = buf[pos..pos + self.pq_subvectors].to_vec();
            (Some(code), None)
        } else {
            let v: Vec<f32> = bytemuck::cast_slice(&buf[pos..pos + self.dims * 4]).to_vec();
            (None, Some(v))
        };

        Ok(NodeRecord {
            id,
            level: level as u32,
            neighbors,
            pq_code,
            vector,
        })
    }

    pub fn read_pq_code(&self, idx: usize) -> Result<&[u8], SegmentError> {
        let start = self.offset_of(idx)?;
        let buf = &self.mmap[start..];
        let level = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let mut pos = 12usize;
        for _ in 0..=level {
            let n_neigh = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + n_neigh * 4;
        }
        Ok(&buf[pos..pos + self.pq_subvectors])
    }

    pub fn id_at(&self, idx: usize) -> Result<u64, SegmentError> {
        let start = self.offset_of(idx)?;
        if start + 8 > self.mmap.len() {
            return Err(SegmentError::Truncated);
        }
        let id = u64::from_le_bytes(self.mmap[start..start + 8].try_into().unwrap());
        Ok(id)
    }

    pub fn vector_at(&self, idx: usize) -> Result<&[f32], SegmentError> {
        let start = self.offset_of(idx)?;
        let buf = &self.mmap[start..];
        if buf.len() < 12 {
            return Err(SegmentError::Truncated);
        }
        let level = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let mut pos = 12usize;
        for _ in 0..=level {
            if pos + 4 > buf.len() {
                return Err(SegmentError::Truncated);
            }
            let n_neigh = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + n_neigh * 4 > buf.len() {
                return Err(SegmentError::Truncated);
            }
            pos += n_neigh * 4;
        }
        if self.pq_enabled {
            return Err(SegmentError::DimensionMismatch);
        }
        if pos + self.dims * 4 > buf.len() {
            return Err(SegmentError::Truncated);
        }
        let v: &[f32] = bytemuck::cast_slice(&buf[pos..pos + self.dims * 4]);
        Ok(v)
    }
}

#[allow(clippy::type_complexity)]
pub struct SegmentWriter<W: std::io::Write + std::io::Seek> {
    inner: W,
    header: Header,
    nodes: Vec<(u64, u32, Vec<Vec<u32>>, Vec<f32>)>,
}

impl<W: std::io::Write + std::io::Seek> SegmentWriter<W> {
    pub fn new(writer: W, dims: u32, m: u32, m_max: u32) -> Self {
        let header = Header {
            magic: MAGIC,
            version: CURRENT_VERSION,
            num_nodes: 0,
            dims,
            m,
            m_max,
            pq_subvectors: 0,
            pq_enabled: 0,
            _reserved: [0u8; 31],
        };
        Self {
            inner: writer,
            header,
            nodes: Vec::new(),
        }
    }

    pub fn push(&mut self, id: u64, level: u32, neighbors: Vec<Vec<u32>>, vector: Vec<f32>) -> Result<(), SegmentError> {
        if vector.len() != self.header.dims as usize {
            return Err(SegmentError::DimensionMismatch);
        }
        self.nodes.push((id, level, neighbors, vector));
        Ok(())
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn write(mut self) -> Result<(), std::io::Error> {
        self.header.num_nodes = self.nodes.len() as u32;
        let hdr_bytes: &[u8] = bytemuck::bytes_of(&self.header);
        self.inner.write_all(hdr_bytes)?;

        let offset_pos = self.inner.stream_position()?;
        let num_nodes = self.nodes.len();
        for _ in 0..num_nodes {
            self.inner.write_all(&[0u8; 8])?;
        }

        let node_data_start = self.inner.stream_position()?;
        let mut offsets = Vec::with_capacity(num_nodes);
        for (id, level, neighbors, vector) in &self.nodes {
            let off = self.inner.stream_position()? - node_data_start;
            offsets.push(off);
            self.inner.write_all(&id.to_le_bytes())?;
            self.inner.write_all(&level.to_le_bytes())?;
            for layer in neighbors {
                self.inner.write_all(&(layer.len() as u32).to_le_bytes())?;
                for &n in layer {
                    self.inner.write_all(&n.to_le_bytes())?;
                }
            }
            let vbytes: &[u8] = bytemuck::cast_slice(vector.as_slice());
            self.inner
                .write_all(&vbytes[..self.header.dims as usize * 4])?;
        }

        let end_pos = self.inner.stream_position()?;
        self.inner.seek(std::io::SeekFrom::Start(offset_pos))?;
        for off in offsets {
            self.inner.write_all(&off.to_le_bytes())?;
        }
        self.inner.seek(std::io::SeekFrom::Start(end_pos))?;
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufWriter, Seek, SeekFrom};
    use tempfile::tempfile;

    #[test]
    fn test_roundtrip() {
        let mut file = tempfile().unwrap();
        {
            let writer = BufWriter::new(file.try_clone().unwrap());
            let mut seg_w = SegmentWriter::new(writer, 4, 16, 32);
            seg_w.push(42, 1, vec![vec![1, 2], vec![3]], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            seg_w.push(7, 0, vec![vec![0]], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
            seg_w.write().unwrap();
        }
        file.seek(SeekFrom::Start(0)).unwrap();

        let seg = SealedSegment::from_file(file).unwrap();
        assert_eq!(seg.num_nodes(), 2);
        assert_eq!(seg.dims(), 4);

        let n0 = seg.read_node(0).unwrap();
        assert_eq!(n0.id, 42);
        assert_eq!(n0.level, 1);
        assert_eq!(n0.neighbors.len(), 2);
        assert_eq!(n0.neighbors[0], vec![1, 2]);
        assert_eq!(n0.vector.as_deref(), Some(&[1.0, 2.0, 3.0, 4.0][..]));

        let n1 = seg.read_node(1).unwrap();
        assert_eq!(n1.id, 7);
        assert_eq!(n1.level, 0);
    }
}
