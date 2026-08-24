//! Wave-4 regression test for A1: cloud HTTPS `CryptoProvider` panic.
//!
//! Background
//! ----------
//! The `object_store` 0.13.2 → 0.14.1 upgrade switched reqwest to the
//! `rustls-no-provider` feature. Under that feature reqwest's
//! `ClientBuilder::build()` **panics** with
//! `"No rustls crypto provider is configured"` unless a process-level
//! `rustls::crypto::CryptoProvider` has been installed first — and
//! `object_store`'s S3/GCS/Azure builders construct that reqwest client
//! *eagerly* inside `build()`. Before the fix, the first cloud client
//! construction (S3/GCS/Azure/MinIO) therefore aborted the process.
//!
//! The fix installs the `ring` provider once (`ensure_crypto_provider`) at the
//! top of every `build_store` in `src/connectors/cloud.rs`.
//!
//! What these tests prove
//! ----------------------
//! Each test drives a real connector `connect()` with a config whose credential
//! path reaches `reqwest::ClientBuilder::build()` **without any network I/O**
//! (client construction never contacts a server):
//!
//! * S3 — static AWS keys reach the unconditional reqwest build in
//!   `AmazonS3Builder::build`.
//! * MinIO — the S3 connector plus a custom HTTP endpoint hits the same
//!   unconditional reqwest build.
//! * GCS — a service-account key with `disable_oauth:true` (object_store's own
//!   test fixture) short-circuits the OAuth token provider and still reaches the
//!   unconditional reqwest build in `GoogleCloudStorageBuilder::build`.
//! * Azure — account name plus a *valid base64* access key reaches the
//!   unconditional reqwest build in `MicrosoftAzureBuilder::build`.
//!
//! Reaching the final assertion at all proves no panic occurred (a panic would
//! abort the test process); `is_ok()` additionally proves the client built
//! cleanly once the provider is installed. Without the `ensure_crypto_provider`
//! guard, every one of these four tests aborts with reqwest's panic message —
//! that is the property under regression.

#![cfg(feature = "cloud-storage")]

use pandrs::connectors::cloud::{
    AzureConnector, CloudConfig, CloudConnector, CloudCredentials, CloudProvider, GCSConnector,
    S3Connector,
};

/// GCS service-account key that parses but disables OAuth, so the builder skips
/// the token provider (which would otherwise need a real RSA private key) and
/// still reaches the eager reqwest client build. This is object_store's own
/// `FAKE_KEY` fixture from `gcp/builder.rs`.
const GCS_FAKE_KEY_DISABLE_OAUTH: &str = r#"{"private_key": "private_key", "private_key_id": "private_key_id", "client_email":"client_email", "disable_oauth":true}"#;

/// Known-good base64 Azure access key (Azurite's `EMULATOR_ACCOUNT_KEY`). Azure's
/// builder base64-decodes the access key while resolving credentials, so a
/// non-base64 placeholder would error out *before* the reqwest client is built
/// and silently defeat the regression.
const AZURE_VALID_BASE64_KEY: &str =
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==";

/// S3: the config path that previously panicked at first client construction.
#[tokio::test]
async fn cloud_s3_client_build_installs_crypto_provider_no_panic() {
    let config = CloudConfig::new(
        CloudProvider::AWS,
        CloudCredentials::AWS {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        },
    )
    .with_region("us-east-1")
    .with_parameter("bucket", "regression-bucket");

    let mut connector = S3Connector::new();
    let result = connector.connect(&config).await;
    assert!(
        result.is_ok(),
        "S3 connect should build a client without panicking once the rustls \
         provider is installed; got: {:?}",
        result.err()
    );
}

/// MinIO: an S3-compatible custom HTTP endpoint (routes through the same S3
/// `build_store` guard site, exercising a distinct config path).
#[tokio::test]
async fn cloud_minio_client_build_installs_crypto_provider_no_panic() {
    let config = CloudConfig::new(
        CloudProvider::MinIO,
        CloudCredentials::AWS {
            access_key_id: "minioadmin".to_string(),
            secret_access_key: "minioadmin".to_string(),
            session_token: None,
        },
    )
    .with_region("us-east-1")
    .with_endpoint("http://127.0.0.1:9000")
    .with_parameter("bucket", "regression-bucket");

    let mut connector = S3Connector::new();
    let result = connector.connect(&config).await;
    assert!(
        result.is_ok(),
        "MinIO (S3 endpoint) connect should build a client without panicking; \
         got: {:?}",
        result.err()
    );
}

/// GCS: the config path that previously panicked at first client construction.
#[tokio::test]
async fn cloud_gcs_client_build_installs_crypto_provider_no_panic() {
    let config = CloudConfig::new(
        CloudProvider::GCS,
        CloudCredentials::GCS {
            service_account_key: GCS_FAKE_KEY_DISABLE_OAUTH.to_string(),
            project_id: "regression-project".to_string(),
        },
    )
    .with_parameter("bucket", "regression-bucket");

    let mut connector = GCSConnector::new();
    let result = connector.connect(&config).await;
    assert!(
        result.is_ok(),
        "GCS connect should build a client without panicking once the rustls \
         provider is installed; got: {:?}",
        result.err()
    );
}

/// Azure: the config path that previously panicked at first client construction.
#[tokio::test]
async fn cloud_azure_client_build_installs_crypto_provider_no_panic() {
    let config = CloudConfig::new(
        CloudProvider::Azure,
        CloudCredentials::Azure {
            account_name: "regressionaccount".to_string(),
            account_key: AZURE_VALID_BASE64_KEY.to_string(),
        },
    )
    .with_parameter("container", "regression-container");

    let mut connector = AzureConnector::new();
    let result = connector.connect(&config).await;
    assert!(
        result.is_ok(),
        "Azure connect should build a client without panicking once the rustls \
         provider is installed; got: {:?}",
        result.err()
    );
}
