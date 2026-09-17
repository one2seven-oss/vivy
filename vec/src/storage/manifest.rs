use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const MANIFEST_MAGIC: &str = "VIVY_MANIFEST_V1";

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid manifest magic")]
    InvalidMagic,
    #[error("corrupt manifest entry")]
    Corrupt,
}

/// Represents the active committed set of sealed segment files in a data directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub segments: Vec<String>,
}

impl Manifest {
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    pub fn manifest_path(dir: impl AsRef<Path>) -> PathBuf {
        dir.as_ref().join("manifest.idx")
    }

    /// Load the manifest from a data directory. If the manifest file does not exist, returns None.
    pub fn load(dir: impl AsRef<Path>) -> Result<Option<Self>, ManifestError> {
        let path = Self::manifest_path(dir);
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header = match lines.next() {
            Some(Ok(h)) => h,
            Some(Err(e)) => return Err(e.into()),
            None => return Err(ManifestError::Corrupt),
        };

        if header != MANIFEST_MAGIC {
            return Err(ManifestError::InvalidMagic);
        }

        let mut segments = Vec::new();
        for line in lines {
            let seg = line?;
            let trimmed = seg.trim();
            if !trimmed.is_empty() {
                segments.push(trimmed.to_string());
            }
        }

        Ok(Some(Self { segments }))
    }

    /// Atomically save the manifest using write-to-temp + fsync + atomic rename.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<(), ManifestError> {
        let dir = dir.as_ref();
        let target_path = Self::manifest_path(dir);
        let temp_path = dir.join(format!("manifest.{}.tmp", std::process::id()));

        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_path)?;

            writeln!(file, "{}", MANIFEST_MAGIC)?;
            for seg in &self.segments {
                writeln!(file, "{}", seg)?;
            }
            file.flush()?;
            file.sync_all()?;
        }

        std::fs::rename(&temp_path, &target_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_manifest_roundtrip() {
        let dir = tempdir().unwrap();
        let manifest = Manifest::new(vec!["seg-1.vivy".into(), "seg-2.vivy".into()]);
        manifest.save(dir.path()).unwrap();

        let loaded = Manifest::load(dir.path()).unwrap().expect("manifest exists");
        assert_eq!(loaded, manifest);
    }
}
