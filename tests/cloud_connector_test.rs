#![allow(clippy::result_large_err)]
//! Integration tests for cloud connector implementations.
//!
//! `LocalConnector` tests run unconditionally (no cloud credentials needed).
//! S3 / GCS / Azure tests compile in all configurations but are skipped at
//! runtime when the required environment variables are absent.

use pandrs::connectors::cloud::{
    CloudConfig, CloudConnector, CloudConnectorFactory, CloudCredentials, CloudObject,
    CloudProvider, FileFormat, GCSConnector, S3Connector,
};
use pandrs::connectors::local::LocalConnector;
use pandrs::dataframe::DataFrame;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build a small DataFrame for round-trip tests.
fn make_test_dataframe() -> DataFrame {
    let mut df = DataFrame::new();
    let series = pandrs::series::base::Series::new(
        vec!["foo".to_string(), "bar".to_string(), "baz".to_string()],
        Some("label".to_string()),
    )
    .expect("series creation");
    df.add_column("label".to_string(), series)
        .expect("add column");
    df
}

/// Return a unique temp directory for each test.
fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir creation")
}

// ──────────────────────────────────────────────────────────────────────────────
// FileFormat tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_file_format_from_extension_csv() {
    let fmt = FileFormat::from_extension("data.csv").expect("csv");
    assert!(matches!(fmt, FileFormat::CSV { .. }));
}

#[test]
fn test_file_format_from_extension_parquet() {
    let fmt = FileFormat::from_extension("data.parquet").expect("parquet");
    assert!(matches!(fmt, FileFormat::Parquet));
}

#[test]
fn test_file_format_from_extension_pq() {
    let fmt = FileFormat::from_extension("data.pq").expect("pq");
    assert!(matches!(fmt, FileFormat::Parquet));
}

#[test]
fn test_file_format_from_extension_json() {
    let fmt = FileFormat::from_extension("data.json").expect("json");
    assert!(matches!(fmt, FileFormat::JSON));
}

#[test]
fn test_file_format_from_extension_jsonl() {
    let fmt = FileFormat::from_extension("data.jsonl").expect("jsonl");
    assert!(matches!(fmt, FileFormat::JSONL));
}

#[test]
fn test_file_format_from_extension_ndjson() {
    let fmt = FileFormat::from_extension("data.ndjson").expect("ndjson");
    assert!(matches!(fmt, FileFormat::JSONL));
}

#[test]
fn test_file_format_from_extension_unknown_returns_none() {
    assert!(FileFormat::from_extension("data.bin").is_none());
    assert!(FileFormat::from_extension("noextension").is_none());
}

// ──────────────────────────────────────────────────────────────────────────────
// CloudConfig builder tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_cloud_config_builder_aws() {
    let config = CloudConfig::new(
        CloudProvider::AWS,
        CloudCredentials::AWS {
            access_key_id: "AKIATEST".to_string(),
            secret_access_key: "secretkey".to_string(),
            session_token: None,
        },
    )
    .with_region("eu-west-1")
    .with_timeout(120)
    .with_parameter("extra", "value");

    assert!(matches!(config.provider, CloudProvider::AWS));
    assert_eq!(config.region, Some("eu-west-1".to_string()));
    assert_eq!(config.timeout, Some(120));
    assert_eq!(config.parameters.get("extra"), Some(&"value".to_string()));
}

#[test]
fn test_cloud_config_builder_gcs() {
    let config = CloudConfig::new(
        CloudProvider::GCS,
        CloudCredentials::GCS {
            service_account_key: r#"{"type":"service_account"}"#.to_string(),
            project_id: "my-project".to_string(),
        },
    )
    .with_parameter("bucket", "my-bucket");

    assert!(matches!(config.provider, CloudProvider::GCS));
    assert_eq!(
        config.parameters.get("bucket"),
        Some(&"my-bucket".to_string())
    );
}

#[test]
fn test_cloud_config_builder_azure() {
    let config = CloudConfig::new(
        CloudProvider::Azure,
        CloudCredentials::Azure {
            account_name: "myaccount".to_string(),
            account_key: "mykey==".to_string(),
        },
    )
    .with_endpoint("https://myaccount.blob.core.windows.net");

    assert!(matches!(config.provider, CloudProvider::Azure));
    assert!(config.endpoint.is_some());
}

// ──────────────────────────────────────────────────────────────────────────────
// Factory tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_factory_creates_connectors() {
    let _s3 = CloudConnectorFactory::s3();
    let _gcs = CloudConnectorFactory::gcs();
    let _azure = CloudConnectorFactory::azure();
}

// ──────────────────────────────────────────────────────────────────────────────
// LocalConnector tests (no cloud credentials required)
// ──────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_local_connector_connect() {
    let dir = temp_dir();
    let mut connector = LocalConnector::new(dir.path());
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await.expect("connect");
}

#[tokio::test]
async fn test_local_connector_create_delete_bucket() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());

    connector
        .create_bucket("test-bucket")
        .await
        .expect("create bucket");
    assert!(dir.path().join("test-bucket").is_dir());

    connector
        .delete_bucket("test-bucket")
        .await
        .expect("delete bucket");
    assert!(!dir.path().join("test-bucket").exists());
}

#[tokio::test]
async fn test_local_connector_upload_exists_download_delete() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());
    connector.create_bucket("mybucket").await.expect("create");

    // Create source file
    let src = dir.path().join("source.txt");
    std::fs::write(&src, b"test content").expect("write source");

    connector
        .upload_object(src.to_str().expect("path"), "mybucket", "data/file.txt")
        .await
        .expect("upload");

    // Exists
    assert!(connector
        .object_exists("mybucket", "data/file.txt")
        .await
        .expect("exists check"));

    // Download
    let dst = dir.path().join("downloaded.txt");
    connector
        .download_object("mybucket", "data/file.txt", dst.to_str().expect("dst path"))
        .await
        .expect("download");
    assert_eq!(std::fs::read(&dst).expect("read dst"), b"test content");

    // Delete
    connector
        .delete_object("mybucket", "data/file.txt")
        .await
        .expect("delete");
    assert!(!connector
        .object_exists("mybucket", "data/file.txt")
        .await
        .expect("exists after delete"));
}

#[tokio::test]
async fn test_local_connector_object_not_exists() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());

    let exists = connector
        .object_exists("nonexistent-bucket", "no-file.csv")
        .await
        .expect("exists check should not error");
    assert!(!exists);
}

#[tokio::test]
async fn test_local_connector_list_objects_empty_bucket() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());
    connector.create_bucket("empty").await.expect("create");

    let objects = connector
        .list_objects("empty", None)
        .await
        .expect("list empty");
    assert!(objects.is_empty());
}

#[tokio::test]
async fn test_local_connector_list_objects_with_prefix() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());
    connector.create_bucket("bkt").await.expect("create");

    // Upload three files
    for name in &["alpha.csv", "alpha_2.csv", "beta.csv"] {
        let src = dir.path().join(name);
        std::fs::write(&src, b"data").expect("write");
        connector
            .upload_object(src.to_str().expect("p"), "bkt", name)
            .await
            .expect("upload");
    }

    let all = connector.list_objects("bkt", None).await.expect("list all");
    assert_eq!(all.len(), 3);

    let alpha_only = connector
        .list_objects("bkt", Some("alpha"))
        .await
        .expect("list prefix");
    assert_eq!(alpha_only.len(), 2);
    for obj in &alpha_only {
        assert!(obj.key.starts_with("alpha"));
    }
}

#[tokio::test]
async fn test_local_connector_metadata() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());
    connector.create_bucket("meta").await.expect("create");

    let content = b"metadata test";
    let src = dir.path().join("meta_src.txt");
    std::fs::write(&src, content).expect("write");

    connector
        .upload_object(src.to_str().expect("p"), "meta", "file.txt")
        .await
        .expect("upload");

    let meta = connector
        .get_object_metadata("meta", "file.txt")
        .await
        .expect("metadata");

    assert_eq!(meta.size, content.len() as u64);
    assert!(meta.last_modified.is_some());
}

#[tokio::test]
async fn test_local_connector_write_read_csv_dataframe() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());

    let df = make_test_dataframe();
    let fmt = FileFormat::CSV {
        delimiter: ',',
        has_header: true,
    };

    connector
        .write_dataframe(&df, "bucket", "output.csv", fmt.clone())
        .await
        .expect("write dataframe");

    let loaded = connector
        .read_dataframe("bucket", "output.csv", fmt)
        .await
        .expect("read dataframe");

    assert_eq!(loaded.row_count(), df.row_count());
    assert_eq!(loaded.column_names().len(), df.column_names().len());
}

#[tokio::test]
async fn test_local_connector_write_read_json_dataframe() {
    let dir = temp_dir();
    let connector = LocalConnector::new(dir.path());

    let df = make_test_dataframe();
    let fmt = FileFormat::JSON;

    connector
        .write_dataframe(&df, "bucket", "output.json", fmt.clone())
        .await
        .expect("write json dataframe");

    let loaded = connector
        .read_dataframe("bucket", "output.json", fmt)
        .await
        .expect("read json dataframe");

    assert_eq!(loaded.row_count(), df.row_count());
}

// ──────────────────────────────────────────────────────────────────────────────
// S3 / GCS compile-only tests (skipped without credentials at runtime)
// ──────────────────────────────────────────────────────────────────────────────

/// Verify that the S3Connector type can be instantiated without the feature.
#[test]
fn test_s3_connector_instantiates() {
    let _connector = S3Connector::new();
}

/// Verify that the GCSConnector type can be instantiated without the feature.
#[test]
fn test_gcs_connector_instantiates() {
    let _connector = GCSConnector::new();
}

/// Attempt to connect to S3 using environment credentials.
/// This test is skipped (returns Ok immediately) when the env vars are absent;
/// it only performs a real connection attempt when they are set.
#[tokio::test]
async fn test_s3_connect_environment_credentials() {
    // Skip if no AWS credentials are configured
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        return;
    }

    let mut connector = S3Connector::new();
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment)
        .with_region("us-east-1")
        .with_parameter("bucket", "test-bucket");

    // Just test that connect() doesn't panic
    let result = connector.connect(&config).await;
    // We accept either Ok or a connection error (the bucket might not exist)
    match result {
        Ok(_) => {}
        Err(e) => eprintln!("S3 connect returned error (expected without real bucket): {e}"),
    }
}

/// Attempt to connect to GCS using environment credentials.
#[tokio::test]
async fn test_gcs_connect_environment_credentials() {
    // Skip if no GCS credentials configured
    if std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_err() {
        return;
    }

    let mut connector = GCSConnector::new();
    let config = CloudConfig::new(CloudProvider::GCS, CloudCredentials::Environment)
        .with_parameter("bucket", "test-bucket");

    let result = connector.connect(&config).await;
    match result {
        Ok(_) => {}
        Err(e) => eprintln!("GCS connect returned error (expected without real bucket): {e}"),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DataConnector integration (local://)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_data_connector_from_local_connection_string() {
    let local_store_path = std::env::temp_dir().join("test_store");
    let conn_str = format!("local:///{}", local_store_path.display());
    let connector = pandrs::connectors::DataConnector::from_connection_string(&conn_str);
    assert!(connector.is_ok());
    assert!(matches!(
        connector.expect("connector"),
        pandrs::connectors::DataConnector::Local(_)
    ));
}

#[tokio::test]
async fn test_data_source_read_write_local() {
    use pandrs::connectors::{DataConnector, DataSource};

    let dir = temp_dir();
    let connector = DataConnector::local(dir.path());
    let source = DataSource::new(connector);

    let df = make_test_dataframe();

    // Write
    source
        .write_cloud(&df, "mybucket", "output.csv")
        .await
        .expect("write cloud");

    // Read
    let loaded = source
        .read_cloud("mybucket", "output.csv")
        .await
        .expect("read cloud");

    assert_eq!(loaded.row_count(), df.row_count());
}
