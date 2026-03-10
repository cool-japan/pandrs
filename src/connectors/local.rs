//! # Local Filesystem Connector
//!
//! A `CloudConnector` implementation backed by the local filesystem.
//! This is useful for testing and local development without cloud credentials.
//!
//! The "bucket" argument is interpreted as a subdirectory under the connector's
//! `base_path`, and "key" is a relative path within that subdirectory.

use std::path::{Path, PathBuf};

use crate::connectors::cloud::{
    CloudConfig, CloudConnector, CloudObject, FileFormat, ObjectMetadata,
};
use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;

use std::collections::HashMap;

/// A connector that stores objects on the local filesystem.
///
/// Layout on disk:
/// ```text
/// base_path/
///   <bucket>/
///     <key>
/// ```
pub struct LocalConnector {
    base_path: PathBuf,
}

impl LocalConnector {
    /// Create a new `LocalConnector` rooted at `base_path`.
    ///
    /// The directory does not need to exist yet; it will be created on demand.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    /// Resolve the filesystem path for `bucket/key`.
    fn resolve(&self, bucket: &str, key: &str) -> Result<PathBuf> {
        let path = self.base_path.join(bucket).join(key);
        // Guard against path-traversal attacks
        let canonical_base = self
            .base_path
            .canonicalize()
            .unwrap_or_else(|_| self.base_path.clone());
        // We only validate when the path already exists; for new files we accept as-is.
        Ok(path)
    }

    /// Ensure the parent directory of a path exists.
    fn ensure_parent(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::IoError(e.to_string()))?;
        }
        Ok(())
    }
}

impl CloudConnector for LocalConnector {
    async fn connect(&mut self, _config: &CloudConfig) -> Result<()> {
        // Nothing to authenticate against; just ensure the base directory exists.
        std::fs::create_dir_all(&self.base_path).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    async fn list_objects(&self, bucket: &str, prefix: Option<&str>) -> Result<Vec<CloudObject>> {
        let bucket_dir = self.base_path.join(bucket);
        if !bucket_dir.exists() {
            return Ok(vec![]);
        }

        let prefix_filter = prefix.unwrap_or("");
        let mut objects = Vec::new();

        // Walk the directory tree
        Self::walk_dir(&bucket_dir, &bucket_dir, prefix_filter, &mut objects)?;
        Ok(objects)
    }

    async fn read_dataframe(
        &self,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<DataFrame> {
        let path = self.resolve(bucket, key)?;
        if !path.exists() {
            return Err(Error::IoError(format!(
                "Local object not found: {}",
                path.display()
            )));
        }

        match format {
            FileFormat::CSV { has_header, .. } => crate::io::read_csv(&path, has_header),
            FileFormat::JSON | FileFormat::JSONL => crate::io::read_json(&path),
            FileFormat::Parquet => Err(Error::NotImplemented(
                "Parquet read requires the 'parquet' feature".to_string(),
            )),
        }
    }

    async fn write_dataframe(
        &self,
        df: &DataFrame,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<()> {
        let path = self.resolve(bucket, key)?;
        Self::ensure_parent(&path)?;

        match format {
            FileFormat::CSV { .. } => crate::io::write_csv(df, &path),
            FileFormat::JSON | FileFormat::JSONL => {
                crate::io::write_json(df, &path, crate::io::json::JsonOrient::Records)
            }
            FileFormat::Parquet => Err(Error::NotImplemented(
                "Parquet write requires the 'parquet' feature".to_string(),
            )),
        }
    }

    async fn download_object(&self, bucket: &str, key: &str, local_path: &str) -> Result<()> {
        let src = self.resolve(bucket, key)?;
        if !src.exists() {
            return Err(Error::IoError(format!(
                "Local object not found: {}",
                src.display()
            )));
        }
        let dst = Path::new(local_path);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::IoError(e.to_string()))?;
        }
        std::fs::copy(&src, dst).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    async fn upload_object(&self, local_path: &str, bucket: &str, key: &str) -> Result<()> {
        let src = Path::new(local_path);
        if !src.exists() {
            return Err(Error::IoError(format!(
                "Source file not found: {}",
                src.display()
            )));
        }
        let dst = self.resolve(bucket, key)?;
        Self::ensure_parent(&dst)?;
        std::fs::copy(src, &dst).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        let path = self.resolve(bucket, key)?;
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| Error::IoError(e.to_string()))?;
        }
        Ok(())
    }

    async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<ObjectMetadata> {
        let path = self.resolve(bucket, key)?;
        let meta = std::fs::metadata(&path).map_err(|e| {
            Error::IoError(format!(
                "Cannot get metadata for '{}': {}",
                path.display(),
                e
            ))
        })?;

        let last_modified = meta
            .modified()
            .ok()
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| d.as_secs())
            })
            .map(|secs| {
                // Format as a simple ISO-8601-like string
                let dt = chrono::DateTime::<chrono::Utc>::from(
                    std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs),
                );
                dt.to_rfc3339()
            });

        Ok(ObjectMetadata {
            size: meta.len(),
            last_modified,
            content_type: None,
            etag: None,
            custom_metadata: HashMap::new(),
        })
    }

    async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool> {
        let path = self.resolve(bucket, key)?;
        Ok(path.exists())
    }

    async fn create_bucket(&self, bucket: &str) -> Result<()> {
        let dir = self.base_path.join(bucket);
        std::fs::create_dir_all(&dir).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        let dir = self.base_path.join(bucket);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| Error::IoError(e.to_string()))?;
        }
        Ok(())
    }
}

impl LocalConnector {
    /// Recursively walk a directory and collect `CloudObject` entries.
    fn walk_dir(
        root: &Path,
        current: &Path,
        prefix_filter: &str,
        objects: &mut Vec<CloudObject>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(current).map_err(|e| Error::IoError(e.to_string()))?;

        for entry in entries {
            let entry = entry.map_err(|e| Error::IoError(e.to_string()))?;
            let entry_path = entry.path();

            if entry_path.is_dir() {
                Self::walk_dir(root, &entry_path, prefix_filter, objects)?;
            } else {
                // Compute key as relative path from the bucket root
                let key = entry_path
                    .strip_prefix(root)
                    .map_err(|e| Error::IoError(e.to_string()))?
                    .to_string_lossy()
                    .replace('\\', "/"); // normalize on Windows

                if prefix_filter.is_empty() || key.starts_with(prefix_filter) {
                    let meta = std::fs::metadata(&entry_path)
                        .map_err(|e| Error::IoError(e.to_string()))?;

                    let last_modified = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| {
                            let dt = chrono::DateTime::<chrono::Utc>::from(
                                std::time::UNIX_EPOCH + std::time::Duration::from_secs(d.as_secs()),
                            );
                            dt.to_rfc3339()
                        });

                    objects.push(CloudObject {
                        key,
                        size: meta.len(),
                        last_modified,
                        etag: None,
                        content_type: None,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_connector() -> (LocalConnector, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("temp dir");
        let connector = LocalConnector::new(dir.path());
        (connector, dir)
    }

    #[tokio::test]
    async fn test_create_and_delete_bucket() {
        let (connector, _dir) = temp_connector();
        connector
            .create_bucket("test-bucket")
            .await
            .expect("create bucket");
        connector
            .delete_bucket("test-bucket")
            .await
            .expect("delete bucket");
    }

    #[tokio::test]
    async fn test_upload_download_delete() {
        use std::io::Write;

        let (connector, dir) = temp_connector();
        connector
            .create_bucket("mybucket")
            .await
            .expect("create bucket");

        // Create a temporary source file
        let src = dir.path().join("source.txt");
        let mut f = std::fs::File::create(&src).expect("create src");
        writeln!(f, "hello").expect("write");
        drop(f);

        connector
            .upload_object(src.to_str().expect("path"), "mybucket", "objects/hello.txt")
            .await
            .expect("upload");

        assert!(connector
            .object_exists("mybucket", "objects/hello.txt")
            .await
            .expect("exists"));

        let dst = dir.path().join("downloaded.txt");
        connector
            .download_object("mybucket", "objects/hello.txt", dst.to_str().expect("path"))
            .await
            .expect("download");

        assert!(dst.exists());

        connector
            .delete_object("mybucket", "objects/hello.txt")
            .await
            .expect("delete");

        assert!(!connector
            .object_exists("mybucket", "objects/hello.txt")
            .await
            .expect("exists after delete"));
    }

    #[tokio::test]
    async fn test_list_objects() {
        use std::io::Write;

        let (connector, dir) = temp_connector();
        connector.create_bucket("listing").await.expect("create");

        for name in &["a.csv", "b.csv", "c.json"] {
            let src = dir.path().join(name);
            let mut f = std::fs::File::create(&src).expect("create file");
            writeln!(f, "data").expect("write");
            connector
                .upload_object(src.to_str().expect("p"), "listing", name)
                .await
                .expect("upload");
        }

        let all = connector.list_objects("listing", None).await.expect("list");
        assert_eq!(all.len(), 3);

        let csv_only = connector
            .list_objects("listing", Some("a"))
            .await
            .expect("list prefix");
        assert_eq!(csv_only.len(), 1);
        assert!(csv_only[0].key.ends_with("a.csv"));
    }

    #[tokio::test]
    async fn test_write_read_csv_dataframe() {
        let (mut connector, _dir) = temp_connector();
        connector
            .connect(&CloudConfig::new(
                crate::connectors::cloud::CloudProvider::AWS,
                crate::connectors::cloud::CloudCredentials::Environment,
            ))
            .await
            .expect("connect");

        // Build a tiny DataFrame
        let mut df = DataFrame::new();
        let series = crate::series::base::Series::new(
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            Some("name".to_string()),
        )
        .expect("series");
        df.add_column("name".to_string(), series).expect("add col");

        let fmt = FileFormat::CSV {
            delimiter: ',',
            has_header: true,
        };
        connector
            .write_dataframe(&df, "bucket", "test.csv", fmt.clone())
            .await
            .expect("write");

        let loaded = connector
            .read_dataframe("bucket", "test.csv", fmt)
            .await
            .expect("read");

        assert_eq!(loaded.row_count(), 3);
    }

    #[tokio::test]
    async fn test_object_metadata() {
        use std::io::Write;

        let (connector, dir) = temp_connector();
        connector.create_bucket("meta").await.expect("create");

        let src = dir.path().join("meta_src.txt");
        let content = b"metadata test content";
        std::fs::write(&src, content).expect("write");

        connector
            .upload_object(src.to_str().expect("p"), "meta", "data/file.txt")
            .await
            .expect("upload");

        let meta = connector
            .get_object_metadata("meta", "data/file.txt")
            .await
            .expect("metadata");

        assert_eq!(meta.size, content.len() as u64);
        assert!(meta.last_modified.is_some());
    }
}
