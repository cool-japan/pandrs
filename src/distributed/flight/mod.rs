//! Arrow Flight RPC support for distributed data transfer.
//!
//! This module provides high-performance DataFrame transfer between PandRS
//! processes using the [Apache Arrow Flight](https://arrow.apache.org/docs/format/Flight.html)
//! protocol.  It builds on the `arrow-flight` and `tonic` crates to expose a
//! gRPC service that any Arrow Flight client – in Rust, Python, Java, Go, …
//! – can talk to.
//!
//! ## Feature flags
//!
//! | Feature      | What it enables                                              |
//! |-------------|--------------------------------------------------------------|
//! | `distributed` | [`conversion`] module (DataFrame ↔ RecordBatch utilities)  |
//! | `flight`      | `server` and `client` modules (full gRPC transport)         |
//!
//! ## Quick start (server side)
//!
//! ```no_run
//! # #[tokio::main]
//! # async fn main() -> pandrs::Result<()> {
//! use pandrs::distributed::flight::server::PandRsFlightServer;
//! use pandrs::{DataFrame, Series};
//!
//! let mut df = DataFrame::new();
//! df.add_column("x".to_string(),
//!     Series::new(vec![1i64, 2, 3], Some("x".to_string()))?.into()).expect("add");
//!
//! let server = PandRsFlightServer::new(50051);
//! server.register_dataframe("my_dataset", &df)?;
//! server.serve().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Quick start (client side)
//!
//! ```no_run
//! # #[tokio::main]
//! # async fn main() -> pandrs::Result<()> {
//! use pandrs::distributed::flight::client::PandRsFlightClient;
//!
//! let mut client = PandRsFlightClient::new("http://localhost:50051");
//! client.connect().await?;
//!
//! for name in client.list_datasets().await? {
//!     println!("Dataset: {name}");
//! }
//!
//! let df = client.get_dataframe("my_dataset").await?;
//! # Ok(())
//! # }
//! ```

/// DataFrame ↔ Arrow RecordBatch conversion utilities.
///
/// Available whenever the `distributed` feature is enabled.
pub mod conversion;

/// Arrow Flight gRPC server ([`PandRsFlightServer`](server::PandRsFlightServer)).
///
/// Requires the `flight` feature.
#[cfg(feature = "flight")]
pub mod server;

/// Arrow Flight gRPC client ([`PandRsFlightClient`](client::PandRsFlightClient)).
///
/// Requires the `flight` feature.
#[cfg(feature = "flight")]
pub mod client;

// Re-export the main public types for ergonomic access via the parent module.
#[cfg(feature = "flight")]
pub use client::PandRsFlightClient;
#[cfg(feature = "flight")]
pub use server::PandRsFlightServer;

pub use conversion::{
    dataframe_to_record_batch, record_batch_to_dataframe, record_batches_to_dataframe,
};
