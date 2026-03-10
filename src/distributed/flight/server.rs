//! Arrow Flight RPC server for serving PandRS DataFrames to remote clients.
//!
//! The [`PandRsFlightServer`] wraps the Arrow Flight gRPC service (implemented
//! via `tonic`) and allows callers to register DataFrames by name.  Remote
//! clients (e.g., `PandRsFlightClient`) can then list
//! available datasets and fetch them as Arrow IPC streams.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use arrow::record_batch::RecordBatch;
use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::Stream;
use tokio::task::JoinHandle;
use tonic::{Request, Response, Status, Streaming};

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;

use super::conversion::dataframe_to_record_batch;

// ---------------------------------------------------------------------------
// Internal dataset registry
// ---------------------------------------------------------------------------

/// Thread-safe map from dataset name to serialised Arrow RecordBatch.
#[derive(Clone, Default)]
struct DatasetRegistry {
    inner: Arc<RwLock<HashMap<String, RecordBatch>>>,
}

impl DatasetRegistry {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&self, name: String, batch: RecordBatch) -> Result<()> {
        self.inner
            .write()
            .map_err(|_| Error::InvalidOperation("Registry lock poisoned".into()))?
            .insert(name, batch);
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<()> {
        self.inner
            .write()
            .map_err(|_| Error::InvalidOperation("Registry lock poisoned".into()))?
            .remove(name);
        Ok(())
    }

    fn get(&self, name: &str) -> Result<Option<RecordBatch>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| Error::InvalidOperation("Registry lock poisoned".into()))?;
        Ok(guard.get(name).cloned())
    }

    fn list(&self) -> Result<Vec<String>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| Error::InvalidOperation("Registry lock poisoned".into()))?;
        Ok(guard.keys().cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// PandRsFlightServer
// ---------------------------------------------------------------------------

/// A gRPC Flight server that serves registered PandRS DataFrames.
///
/// # Example
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> pandrs::Result<()> {
/// use pandrs::distributed::flight::server::PandRsFlightServer;
/// use pandrs::DataFrame;
///
/// let server = PandRsFlightServer::new(50051);
/// // server.register_dataframe("my_df", &df)?;
/// server.serve().await?;
/// # Ok(())
/// # }
/// ```
pub struct PandRsFlightServer {
    registry: DatasetRegistry,
    port: u16,
}

impl PandRsFlightServer {
    /// Create a new Flight server bound to the given port.
    pub fn new(port: u16) -> Self {
        Self {
            registry: DatasetRegistry::new(),
            port,
        }
    }

    /// Register a [`DataFrame`] under `name` so Flight clients can request it.
    ///
    /// The DataFrame is converted to an Arrow [`RecordBatch`] immediately and
    /// stored in memory.
    pub fn register_dataframe(&self, name: &str, df: &DataFrame) -> Result<()> {
        let batch = dataframe_to_record_batch(df)?;
        self.registry.insert(name.to_string(), batch)
    }

    /// Remove a previously registered dataset.  Silently succeeds if absent.
    pub fn unregister(&self, name: &str) -> Result<()> {
        self.registry.remove(name)
    }

    /// Return the names of all registered datasets.
    pub fn list_datasets(&self) -> Vec<String> {
        self.registry.list().unwrap_or_default()
    }

    /// Start the Flight gRPC server and block until it shuts down.
    pub async fn serve(self) -> Result<()> {
        let addr: SocketAddr = format!("0.0.0.0:{}", self.port)
            .parse()
            .map_err(|e| Error::InvalidOperation(format!("Invalid address: {e}")))?;

        let service = PandRsFlightServiceImpl {
            registry: self.registry,
        };

        tonic::transport::Server::builder()
            .add_service(FlightServiceServer::new(service))
            .serve(addr)
            .await
            .map_err(|e| Error::InvalidOperation(format!("Flight server error: {e}")))
    }

    /// Spawn the Flight server on a background tokio task.
    ///
    /// Returns a [`JoinHandle`] that resolves when the server exits.
    pub fn serve_background(self) -> Result<JoinHandle<Result<()>>> {
        let handle = tokio::spawn(async move { self.serve().await });
        Ok(handle)
    }
}

// ---------------------------------------------------------------------------
// tonic FlightService implementation
// ---------------------------------------------------------------------------

/// Internal tonic service implementation.
struct PandRsFlightServiceImpl {
    registry: DatasetRegistry,
}

/// Boxed stream alias used by several Flight RPC methods.
type BoxedStream<T> = Pin<Box<dyn Stream<Item = std::result::Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl FlightService for PandRsFlightServiceImpl {
    type HandshakeStream = BoxedStream<HandshakeResponse>;
    type ListFlightsStream = BoxedStream<FlightInfo>;
    type DoGetStream = BoxedStream<FlightData>;
    type DoPutStream = BoxedStream<PutResult>;
    type DoExchangeStream = BoxedStream<FlightData>;
    type DoActionStream = BoxedStream<arrow_flight::Result>;
    type ListActionsStream = BoxedStream<ActionType>;

    // ------------------------------------------------------------------
    // Handshake (no-auth: just echo the request payload)
    // ------------------------------------------------------------------
    async fn handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> std::result::Result<Response<Self::HandshakeStream>, Status> {
        let response = HandshakeResponse {
            protocol_version: 0,
            payload: bytes::Bytes::new(),
        };
        let stream = futures::stream::once(async move { Ok(response) });
        Ok(Response::new(Box::pin(stream)))
    }

    // ------------------------------------------------------------------
    // ListFlights – returns FlightInfo for every registered dataset
    // ------------------------------------------------------------------
    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> std::result::Result<Response<Self::ListFlightsStream>, Status> {
        let names = self
            .registry
            .list()
            .map_err(|e| Status::internal(e.to_string()))?;

        let infos: Vec<std::result::Result<FlightInfo, Status>> = names
            .into_iter()
            .map(|name| {
                let descriptor = FlightDescriptor::new_path(vec![name.clone()]);
                Ok(FlightInfo {
                    schema: bytes::Bytes::new(),
                    flight_descriptor: Some(descriptor),
                    endpoint: vec![],
                    total_records: -1,
                    total_bytes: -1,
                    ordered: false,
                    app_metadata: bytes::Bytes::new(),
                })
            })
            .collect();

        let stream = futures::stream::iter(infos);
        Ok(Response::new(Box::pin(stream)))
    }

    // ------------------------------------------------------------------
    // GetFlightInfo – return metadata for a single named dataset
    // ------------------------------------------------------------------
    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let name = descriptor.path.first().cloned().unwrap_or_default();

        let _batch = self
            .registry
            .get(&name)
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found(format!("Dataset '{name}' not found")))?;

        let ticket_bytes = bytes::Bytes::from(name.as_bytes().to_vec());
        let endpoint = arrow_flight::FlightEndpoint {
            ticket: Some(Ticket {
                ticket: ticket_bytes,
            }),
            location: vec![],
            expiration_time: None,
            app_metadata: bytes::Bytes::new(),
        };

        Ok(Response::new(FlightInfo {
            schema: bytes::Bytes::new(),
            flight_descriptor: Some(descriptor),
            endpoint: vec![endpoint],
            total_records: -1,
            total_bytes: -1,
            ordered: false,
            app_metadata: bytes::Bytes::new(),
        }))
    }

    // ------------------------------------------------------------------
    // PollFlightInfo – poll a long-running query (not implemented)
    // ------------------------------------------------------------------
    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("PollFlightInfo is not implemented"))
    }

    // ------------------------------------------------------------------
    // GetSchema
    // ------------------------------------------------------------------
    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<SchemaResult>, Status> {
        let descriptor = request.into_inner();
        let name = descriptor.path.first().cloned().unwrap_or_default();

        let batch = self
            .registry
            .get(&name)
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found(format!("Dataset '{name}' not found")))?;

        let schema_bytes =
            schema_to_ipc_bytes(batch.schema_ref()).map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(SchemaResult {
            schema: schema_bytes,
        }))
    }

    // ------------------------------------------------------------------
    // DoGet – stream a dataset as Arrow IPC FlightData
    // ------------------------------------------------------------------
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> std::result::Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let name = String::from_utf8(ticket.ticket.to_vec())
            .map_err(|e| Status::invalid_argument(format!("Invalid ticket: {e}")))?;

        let batch = self
            .registry
            .get(&name)
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found(format!("Dataset '{name}' not found")))?;

        // Encode the schema and the single batch as FlightData messages.
        let flight_data_list =
            record_batch_to_flight_data(&batch).map_err(|e| Status::internal(e.to_string()))?;

        let stream =
            futures::stream::iter(flight_data_list.into_iter().map(Ok::<FlightData, Status>));
        Ok(Response::new(Box::pin(stream)))
    }

    // ------------------------------------------------------------------
    // DoPut – accept and store a dataset from the client
    // ------------------------------------------------------------------
    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoPutStream>, Status> {
        let mut stream = request.into_inner();
        let mut flight_data_msgs: Vec<FlightData> = Vec::new();

        while let Some(msg) = stream.message().await? {
            flight_data_msgs.push(msg);
        }

        if flight_data_msgs.is_empty() {
            return Err(Status::invalid_argument("No FlightData received"));
        }

        // First message contains the descriptor and (optionally) the schema.
        let descriptor = flight_data_msgs[0]
            .flight_descriptor
            .clone()
            .ok_or_else(|| Status::invalid_argument("Missing FlightDescriptor in first message"))?;

        let name = descriptor
            .path
            .first()
            .cloned()
            .unwrap_or_else(|| "unnamed".to_string());

        // Decode the RecordBatch from IPC.
        let batch = flight_data_to_record_batch(&flight_data_msgs)
            .map_err(|e| Status::internal(e.to_string()))?;

        self.registry
            .insert(name, batch)
            .map_err(|e| Status::internal(e.to_string()))?;

        let result_stream = futures::stream::empty::<std::result::Result<PutResult, Status>>();
        Ok(Response::new(Box::pin(result_stream)))
    }

    // ------------------------------------------------------------------
    // DoExchange – bidirectional exchange (not implemented)
    // ------------------------------------------------------------------
    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("DoExchange is not implemented"))
    }

    // ------------------------------------------------------------------
    // DoAction – generic action (not implemented)
    // ------------------------------------------------------------------
    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> std::result::Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("DoAction is not implemented"))
    }

    // ------------------------------------------------------------------
    // ListActions
    // ------------------------------------------------------------------
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> std::result::Result<Response<Self::ListActionsStream>, Status> {
        let stream = futures::stream::empty::<std::result::Result<ActionType, Status>>();
        Ok(Response::new(Box::pin(stream)))
    }
}

// ---------------------------------------------------------------------------
// IPC encoding / decoding helpers
// ---------------------------------------------------------------------------

/// Encode an Arrow schema as IPC bytes (for SchemaResult).
fn schema_to_ipc_bytes(schema: &arrow::datatypes::Schema) -> Result<bytes::Bytes> {
    use arrow::ipc::writer::IpcWriteOptions;
    use arrow_flight::SchemaAsIpc;

    let options = IpcWriteOptions::default();
    let schema_ipc = SchemaAsIpc::new(schema, &options);
    let flight_data: FlightData = schema_ipc.into();
    Ok(flight_data.data_header)
}

/// Encode a [`RecordBatch`] as a `Vec<FlightData>` (schema msg + data msg).
fn record_batch_to_flight_data(batch: &RecordBatch) -> Result<Vec<FlightData>> {
    use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
    use std::io::Cursor;

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

    // Wrap the entire IPC stream in a single FlightData message.
    let flight_data = FlightData {
        flight_descriptor: None,
        data_header: bytes::Bytes::new(),
        app_metadata: bytes::Bytes::new(),
        data_body: bytes::Bytes::from(buf),
    };
    Ok(vec![flight_data])
}

/// Decode a slice of [`FlightData`] messages back into a [`RecordBatch`].
fn flight_data_to_record_batch(msgs: &[FlightData]) -> Result<RecordBatch> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    // Collect the data_body bytes from all messages.
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
