//! Arrow Flight RPC client for fetching PandRS DataFrames from remote servers.
//!
//! [`PandRsFlightClient`] connects to a `PandRsFlightServer`
//! (or any conformant Arrow Flight service) and can list available datasets,
//! fetch them as DataFrames, and push local DataFrames to the server.

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    flight_service_client::FlightServiceClient, FlightData, FlightDescriptor, Ticket,
};
use futures::StreamExt;
use std::io::Cursor;
use tonic::transport::Channel;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;

use super::conversion::{dataframe_to_record_batch, record_batch_to_dataframe};

// ---------------------------------------------------------------------------
// PandRsFlightClient
// ---------------------------------------------------------------------------

/// gRPC Flight client for communicating with a `PandRsFlightServer`.
///
/// # Example
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> pandrs::Result<()> {
/// use pandrs::distributed::flight::client::PandRsFlightClient;
///
/// let mut client = PandRsFlightClient::new("http://localhost:50051");
/// client.connect().await?;
///
/// let datasets = client.list_datasets().await?;
/// println!("Available: {datasets:?}");
///
/// let df = client.get_dataframe("my_df").await?;
/// println!("{df:?}");
/// # Ok(())
/// # }
/// ```
pub struct PandRsFlightClient {
    endpoint: String,
    inner: Option<FlightServiceClient<Channel>>,
}

impl PandRsFlightClient {
    /// Create a new client pointing at the given endpoint URL.
    ///
    /// Call [`connect`](Self::connect) before making any RPC calls.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            inner: None,
        }
    }

    /// Establish the gRPC connection to the Flight server.
    pub async fn connect(&mut self) -> Result<()> {
        let channel = Channel::from_shared(self.endpoint.clone())
            .map_err(|e| Error::InvalidOperation(format!("Invalid endpoint URI: {e}")))?
            .connect()
            .await
            .map_err(|e| Error::InvalidOperation(format!("gRPC connect failed: {e}")))?;

        self.inner = Some(FlightServiceClient::new(channel));
        Ok(())
    }

    /// Return a mutable reference to the inner client, or an error if not connected.
    fn client(&mut self) -> Result<&mut FlightServiceClient<Channel>> {
        self.inner
            .as_mut()
            .ok_or_else(|| Error::InvalidOperation("Not connected – call connect() first".into()))
    }

    // ------------------------------------------------------------------
    // List
    // ------------------------------------------------------------------

    /// List the names of all datasets available on the server.
    pub async fn list_datasets(&mut self) -> Result<Vec<String>> {
        let request = arrow_flight::Criteria {
            expression: bytes::Bytes::new(),
        };

        let mut stream = self
            .client()?
            .list_flights(request)
            .await
            .map_err(|e| Error::InvalidOperation(format!("ListFlights RPC failed: {e}")))?
            .into_inner();

        let mut names = Vec::new();
        while let Some(info) = stream
            .next()
            .await
            .transpose()
            .map_err(|e| Error::InvalidOperation(format!("ListFlights stream error: {e}")))?
        {
            if let Some(descriptor) = info.flight_descriptor {
                if let Some(first) = descriptor.path.into_iter().next() {
                    names.push(first);
                }
            }
        }

        Ok(names)
    }

    // ------------------------------------------------------------------
    // Get
    // ------------------------------------------------------------------

    /// Fetch a dataset by name and return it as a [`DataFrame`].
    pub async fn get_dataframe(&mut self, name: &str) -> Result<DataFrame> {
        let ticket = Ticket {
            ticket: bytes::Bytes::from(name.as_bytes().to_vec()),
        };

        let mut stream = self
            .client()?
            .do_get(ticket)
            .await
            .map_err(|e| Error::InvalidOperation(format!("DoGet RPC failed: {e}")))?
            .into_inner();

        // Collect all FlightData messages
        let mut flight_data_msgs: Vec<FlightData> = Vec::new();
        while let Some(msg) = stream
            .next()
            .await
            .transpose()
            .map_err(|e| Error::InvalidOperation(format!("DoGet stream error: {e}")))?
        {
            flight_data_msgs.push(msg);
        }

        if flight_data_msgs.is_empty() {
            return Err(Error::InvalidOperation(format!(
                "Server returned no data for dataset '{name}'"
            )));
        }

        let batch = flight_data_to_record_batch(&flight_data_msgs)?;
        record_batch_to_dataframe(&batch)
    }

    // ------------------------------------------------------------------
    // Put
    // ------------------------------------------------------------------

    /// Push a local [`DataFrame`] to the server under `name`.
    ///
    /// The server will register it for subsequent [`get_dataframe`](Self::get_dataframe) calls.
    pub async fn put_dataframe(&mut self, name: &str, df: &DataFrame) -> Result<()> {
        let batch = dataframe_to_record_batch(df)?;
        let flight_data_msgs = record_batch_to_flight_data_with_descriptor(name, &batch)?;

        // DoPut expects a stream of FlightData (items are plain FlightData, not Result)
        let request_stream = futures::stream::iter(flight_data_msgs.into_iter());

        self.client()?
            .do_put(request_stream)
            .await
            .map_err(|e| Error::InvalidOperation(format!("DoPut RPC failed: {e}")))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IPC encoding helpers (mirrored from server.rs to avoid cross-module deps)
// ---------------------------------------------------------------------------

/// Encode a [`RecordBatch`] as a `Vec<FlightData>`, attaching a [`FlightDescriptor`]
/// on the first message so the server knows the dataset name.
fn record_batch_to_flight_data_with_descriptor(
    name: &str,
    batch: &RecordBatch,
) -> Result<Vec<FlightData>> {
    use arrow::ipc::writer::IpcWriteOptions;

    let mut buf = Vec::new();
    let options = IpcWriteOptions::default();
    {
        let mut writer = StreamWriter::try_new_with_options(&mut buf, batch.schema_ref(), options)
            .map_err(|e| Error::InvalidOperation(format!("IPC writer init failed: {e}")))?;
        writer
            .write(batch)
            .map_err(|e| Error::InvalidOperation(format!("IPC write failed: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::InvalidOperation(format!("IPC finish failed: {e}")))?;
    }

    let descriptor = FlightDescriptor::new_path(vec![name.to_string()]);
    let flight_data = FlightData {
        flight_descriptor: Some(descriptor),
        data_header: bytes::Bytes::new(),
        app_metadata: bytes::Bytes::new(),
        data_body: bytes::Bytes::from(buf),
    };

    Ok(vec![flight_data])
}

/// Decode a slice of [`FlightData`] into a single [`RecordBatch`].
fn flight_data_to_record_batch(msgs: &[FlightData]) -> Result<RecordBatch> {
    let mut combined: Vec<u8> = Vec::new();
    for msg in msgs {
        combined.extend_from_slice(&msg.data_body);
    }

    let cursor = Cursor::new(combined);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Error::InvalidOperation(format!("IPC reader init failed: {e}")))?;

    let batch = reader
        .next()
        .ok_or_else(|| Error::InvalidOperation("No RecordBatch in IPC stream".into()))?
        .map_err(|e| Error::InvalidOperation(format!("IPC read failed: {e}")))?;

    Ok(batch)
}
