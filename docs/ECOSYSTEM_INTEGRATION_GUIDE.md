# PandRS Ecosystem Integration Guide

A concise, source-verified guide to how PandRS talks to the outside world:
cloud storage, Apache Arrow, and Python. **There is no database
connectivity** (no SQL feature, no PostgreSQL/MySQL/SQLite support) — if you
need that, you'll have to build it on top of the cloud/local connectors
described below, or use `py_bindings` from Python alongside a real Python
DB driver.

## Cloud Storage (`cloud-storage` feature)

`pandrs::connectors::cloud` provides real, working connectors for S3, GCS,
and Azure Blob (plus MinIO as an S3-compatible provider):

```rust
use pandrs::connectors::cloud::{CloudConfig, CloudProvider, CloudCredentials, CloudConnectorFactory};

let config = CloudConfig::new(
    CloudProvider::AWS,
    CloudCredentials::AWS {
        access_key_id: "...".to_string(),
        secret_access_key: "...".to_string(),
        session_token: None,
    },
)
.with_region("us-east-1")
.with_timeout(30);

let connector = CloudConnectorFactory::s3(); // or ::gcs() / ::azure()
```

(`CloudCredentials::Environment` is also available — it reads the standard
env vars in the table below instead of taking explicit key material.)

*(Check `src/connectors/cloud.rs` for `CloudCredentials`' exact variant
shape per provider — it differs between AWS/GCS/Azure — and for the
`CloudConnector` trait's methods, e.g. `get_object`/`put_object`/`list_objects`.)*

### Environment Variables

The connectors read standard cloud-provider environment variables directly
(no PandRS-specific env var scheme):

| Provider | Variables |
|---|---|
| AWS | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (optional) |
| GCS | `GOOGLE_APPLICATION_CREDENTIALS` (path to a service-account JSON, Application Default Credentials) |
| Azure | `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY` |

### Local Filesystem

`pandrs::connectors::local::LocalConnector::new(base_path)` implements the
same `CloudConnector` trait as the cloud backends, backed by the local
filesystem — useful for testing cloud-shaped code without real cloud
credentials. (Its internal Parquet read/write helpers are a private
implementation detail, not part of the public API.)

## Apache Arrow (`src/arrow_integration.rs`)

**Every method here requires the `distributed` feature** — `pandrs =
{ features = ["distributed"] }` — even though the trait/struct
declarations themselves aren't individually feature-gated, each method body
is:

```rust
use pandrs::arrow_integration::ArrowIntegration; // trait, brings .to_arrow()/.from_arrow()/.compute_arrow() into scope

let batch = df.to_arrow()?;                    // DataFrame -> arrow::record_batch::RecordBatch
let df_back = DataFrame::from_arrow(&batch)?;   // RecordBatch -> DataFrame
```

(Equivalently, `ArrowConverter::dataframe_to_record_batch(&df)` /
`ArrowConverter::record_batch_to_dataframe(&batch)` — `ArrowConverter` is
the unit struct these trait methods delegate to.)

**This is a converting bridge, not zero-copy.** PandRS's internal columnar
representation (`Int64Column`/`Float64Column`/`StringColumn`/`BooleanColumn`)
is not the same memory layout as Arrow's buffers, so both directions copy
data. Treat "Arrow interoperability" as "you can get your data into/out of
Arrow's `RecordBatch`," not as a zero-allocation fast path.

`df.compute_arrow(operation: ArrowOperation)` runs one of a small, fixed set
of operations through Arrow's compute kernels — as of this writing that's
`ArrowOperation::Sum(column)`,
`ArrowOperation::Filter { column, predicate }` (with `FilterPredicate::{GreaterThan, LessThan, EqualTo, NotEqualTo}`),
and `ArrowOperation::Sort { columns, ascending }`. Same `distributed`-feature
requirement as above.

## Python Bindings (`py_bindings/`)

This is a real, separate PyO3 crate (built with `maturin`, not a Cargo
feature of the main `pandrs` crate) with a substantial, real pandas-shaped
surface — verified real `#[pyclass]` types in `py_bindings/src/`:

- `DataFrame`, `Series`, `NASeries` (`lib.rs`)
- `OptimizedDataFrame`, `LazyFrame` (`py_optimized.rs`)
- `PandasDataFrame`, `PandasSeries`, `ILocIndexer` — a pandas-compatibility layer (`pandas_compat.rs`)
- `GpuConfig`, `GpuDeviceStatus`, `GpuMatrix` — real GPU bindings covering PCA, k-means, linear regression, and correlation (`py_gpu.rs`), backed by the same `src/ml/gpu.rs`/`src/gpu` code described in [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md)

Build it with `maturin develop` (or `maturin build --release`) from
`py_bindings/`; see that directory's own docs/tests for exact usage, since
the Python-side API surface is out of scope for this Rust-crate-focused
guide to hand-copy without a Python interpreter to verify against.

## Security Note

There is no built-in secrets manager. Credentials for cloud connectors come
from the environment variables above (or the `CloudCredentials` enum
constructed however you like — including hardcoding, which you should not
do). Treat credential handling the same way you would for any Rust
application: don't commit them, prefer environment/IAM-role-based auth over
long-lived static keys where your cloud provider supports it.

## What This Guide Intentionally Does Not Cover

Earlier versions of this document described a database-connectivity
chapter (`DatabaseConnectorFactory`, transaction management, a
`DatabaseConfig` security-control builder), a monitoring/observability
chapter, and a testing-framework chapter. **None of that exists in the
current codebase** — those sections have been removed rather than kept as
placeholders, per the project's policy against documenting fabricated APIs.
If you need database access, use a real Rust DB crate (e.g. one of the
COOLJAPAN `oxisql-*` crates) directly alongside PandRS; there is no PandRS
wrapper for it today.
