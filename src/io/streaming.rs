//! Comprehensive I/O and Streaming Traits for PandRS
//!
//! This module provides unified streaming data processing capabilities,
//! asynchronous I/O operations, and high-performance data pipeline support
//! for real-time analytics and large-scale data processing.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Trait for streaming data sources
pub trait StreamingDataSource: Send + Sync {
    type Item: Send + Sync + Clone;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Get the next batch of data
    fn next_batch(
        &mut self,
    ) -> Pin<
        Box<
            dyn Future<Output = std::result::Result<Option<Vec<Self::Item>>, Self::Error>>
                + Send
                + '_,
        >,
    >;

    /// Check if the stream has more data
    fn has_more(&self) -> bool;

    /// Get stream metadata
    fn metadata(&self) -> StreamMetadata;

    /// Set batch size for reading
    fn set_batch_size(&mut self, size: usize);

    /// Reset the stream to the beginning (if supported)
    fn reset(&mut self) -> std::result::Result<(), Self::Error>;

    /// Estimate total number of items (if known)
    fn estimated_size(&self) -> Option<usize>;
}

/// Trait for streaming data sinks
pub trait StreamingDataSink: Send + Sync {
    type Item: Send + Sync + Clone;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Write a batch of data to the sink
    fn write_batch(
        &mut self,
        batch: Vec<Self::Item>,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Self::Error>> + Send + '_>>;

    /// Flush any pending writes
    fn flush(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Self::Error>> + Send + '_>>;

    /// Close the sink and release resources
    fn close(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Self::Error>> + Send + '_>>;

    /// Get sink metadata
    fn metadata(&self) -> SinkMetadata;

    /// Set maximum batch size for writing
    fn set_max_batch_size(&mut self, size: usize);
}

/// Stream metadata information
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    /// Stream identifier
    pub id: String,
    /// Stream type
    pub stream_type: StreamType,
    /// Schema information
    pub schema: Option<StreamSchema>,
    /// Estimated data size
    pub estimated_size: Option<usize>,
    /// Stream creation time
    pub created_at: Instant,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Sink metadata information
#[derive(Debug, Clone)]
pub struct SinkMetadata {
    /// Sink identifier
    pub id: String,
    /// Sink type
    pub sink_type: SinkType,
    /// Schema information
    pub schema: Option<StreamSchema>,
    /// Sink creation time
    pub created_at: Instant,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Stream schema definition
#[derive(Debug, Clone)]
pub struct StreamSchema {
    /// Field definitions
    pub fields: Vec<StreamField>,
    /// Schema metadata
    pub metadata: HashMap<String, String>,
}

/// Stream field definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamField {
    /// Field name
    pub name: String,
    /// Field data type
    pub data_type: StreamDataType,
    /// Whether the field is nullable
    pub nullable: bool,
    /// Field metadata
    pub metadata: HashMap<String, String>,
}

/// Stream data type enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDataType {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    String,
    Binary,
    Date,
    DateTime,
    Timestamp,
    Decimal { precision: u8, scale: i8 },
    List(Box<StreamDataType>),
    Struct(Vec<StreamField>),
}

/// Stream type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    File,
    Network,
    Database,
    Queue,
    Memory,
    Kafka,
    Custom,
}

/// Sink type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkType {
    File,
    Network,
    Database,
    Queue,
    Memory,
    Kafka,
    Custom,
}

/// Streaming processor for data transformations
pub trait StreamProcessor: Send + Sync {
    type Input: Send + Sync + Clone;
    type Output: Send + Sync + Clone;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Process a batch of data
    fn process_batch(
        &mut self,
        batch: Vec<Self::Input>,
    ) -> Pin<
        Box<dyn Future<Output = std::result::Result<Vec<Self::Output>, Self::Error>> + Send + '_>,
    >;

    /// Get processor metadata
    fn metadata(&self) -> ProcessorMetadata;

    /// Configure the processor
    fn configure(&mut self, config: ProcessorConfig) -> std::result::Result<(), Self::Error>;

    /// Get processing statistics
    fn stats(&self) -> ProcessorStats;
}

/// Processor metadata
#[derive(Debug, Clone)]
pub struct ProcessorMetadata {
    /// Processor name
    pub name: String,
    /// Processor type
    pub processor_type: ProcessorType,
    /// Input schema
    pub input_schema: Option<StreamSchema>,
    /// Output schema
    pub output_schema: Option<StreamSchema>,
    /// Processor version
    pub version: String,
}

/// Processor type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorType {
    Filter,
    Map,
    Reduce,
    Aggregate,
    Join,
    Window,
    Custom,
}

/// Processor configuration
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Parallelism settings
    pub parallelism: usize,
    /// Buffer size settings
    pub buffer_size: usize,
    /// Timeout settings
    pub timeout: Option<Duration>,
}

/// Processor statistics
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    /// Total items processed
    pub items_processed: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per item
    pub avg_processing_time: Duration,
    /// Error count
    pub error_count: u64,
    /// Last processing time
    pub last_processed: Option<Instant>,
}

/// Data pipeline for streaming operations
///
/// # Concurrency
///
/// `execute` connects a source and a sink through two `tokio::spawn`ed
/// tasks joined by a `tokio::sync::mpsc` channel. This module previously
/// used `crossbeam_channel`'s *blocking* `recv()`/`send()` for that
/// hand-off, from inside `tokio::spawn`ed futures: on a `current_thread`
/// runtime that deadlocks immediately (the one worker thread blocks
/// forever waiting on a channel that only a second task -- which never gets
/// to run -- could service), and on a multi-thread runtime it silently
/// burns a whole worker thread per blocked call and can still deadlock once
/// enough workers are blocked this way. `tokio::sync::mpsc` is async
/// end-to-end, so the tasks genuinely yield instead of blocking a worker.
pub struct StreamingPipeline<T> {
    /// Pipeline stages
    stages: Vec<Box<dyn PipelineStage<T>>>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Pipeline statistics
    stats: Arc<Mutex<PipelineStats>>,
    /// Error handler
    error_handler: Option<Box<dyn ErrorHandler>>,
}

impl<T> StreamingPipeline<T>
where
    T: Send + Sync + Clone + 'static,
{
    /// Create a new streaming pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
            stats: Arc::new(Mutex::new(PipelineStats::new())),
            error_handler: None,
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage<S: PipelineStage<T> + 'static>(&mut self, stage: S) {
        self.stages.push(Box::new(stage));
    }

    /// Set error handler
    pub fn set_error_handler<H: ErrorHandler + 'static>(&mut self, handler: H) {
        self.error_handler = Some(Box::new(handler));
    }

    /// Execute the pipeline: read batches from `source`, run each batch
    /// through every registered stage in order, and write the result to
    /// `sink`.
    ///
    /// Previously, registered stages were never actually invoked --
    /// `add_stage` recorded them, but `execute`'s processor task forwarded
    /// each batch from source to sink completely unmodified, silently
    /// discarding whatever the stages were supposed to do. This also honors
    /// `config.error_strategy` (and, if set, `error_handler`, which takes
    /// precedence) for both source-read and stage-processing errors, and
    /// `PipelineStats.error_count` is incremented on every recorded error
    /// via `record_error()`.
    ///
    /// `execute` is single-shot with respect to the registered stages and
    /// error handler: it moves them into the spawned processor task (so the
    /// task can own them across `.await` points without borrowing `self`
    /// for its whole lifetime) and does not restore them afterward. Only
    /// `self.stats` -- a shared `Arc<Mutex<..>>` -- remains valid to read
    /// via `stats()` after `execute` returns; add stages again before
    /// calling `execute` a second time on the same pipeline instance.
    pub async fn execute<S, K>(&mut self, source: S, sink: K) -> Result<()>
    where
        S: StreamingDataSource<Item = T> + 'static,
        K: StreamingDataSink<Item = T> + 'static,
    {
        let (tx, rx) = mpsc::channel(self.config.buffer_size.max(1));

        // Start source reader
        let source_handle = self.spawn_source_reader(source, tx).await?;

        // Start pipeline processor
        let processor_handle = self.spawn_pipeline_processor(rx, sink).await?;

        // Wait for completion
        let (source_result, processor_result) = tokio::join!(source_handle, processor_handle);
        source_result
            .map_err(|e| Error::InvalidOperation(format!("Source task failed: {}", e)))??;
        processor_result
            .map_err(|e| Error::InvalidOperation(format!("Processor task failed: {}", e)))??;

        Ok(())
    }

    /// Spawn source reader task.
    ///
    /// On a read error, `config.error_strategy` decides what happens:
    /// `FailFast` propagates the error (terminating the pipeline instead of
    /// silently truncating the stream with an `Ok(())`), `SkipErrors` moves
    /// on to the next batch, and `RetryWithBackoff` retries the same
    /// `next_batch()` call after a short delay, up to a bounded number of
    /// attempts (a source read is naturally retriable: unlike a stage
    /// error, no already-consumed batch data needs to be recovered).
    async fn spawn_source_reader<S>(
        &self,
        mut source: S,
        tx: mpsc::Sender<Vec<T>>,
    ) -> Result<tokio::task::JoinHandle<Result<()>>>
    where
        S: StreamingDataSource<Item = T> + 'static,
    {
        let stats = Arc::clone(&self.stats);
        let error_strategy = self.config.error_strategy;
        const MAX_SOURCE_RETRIES: u32 = 5;

        let handle = tokio::spawn(async move {
            let mut consecutive_errors = 0u32;

            while source.has_more() {
                match source.next_batch().await {
                    Ok(Some(batch)) => {
                        consecutive_errors = 0;
                        if let Ok(mut pipeline_stats) = stats.lock() {
                            pipeline_stats.record_batch_read(batch.len());
                        }

                        if tx.send(batch).await.is_err() {
                            break; // Pipeline closed (processor task exited)
                        }
                    }
                    Ok(None) => break, // End of stream
                    Err(e) => {
                        if let Ok(mut pipeline_stats) = stats.lock() {
                            pipeline_stats.record_error();
                        }

                        match error_strategy {
                            ErrorStrategy::FailFast => {
                                return Err(Error::InvalidOperation(format!(
                                    "Source read error: {}",
                                    e
                                )));
                            }
                            ErrorStrategy::SkipErrors => {
                                continue;
                            }
                            ErrorStrategy::RetryWithBackoff => {
                                consecutive_errors += 1;
                                if consecutive_errors > MAX_SOURCE_RETRIES {
                                    return Err(Error::InvalidOperation(format!(
                                        "Source read error persisted after {} retries: {}",
                                        MAX_SOURCE_RETRIES, e
                                    )));
                                }
                                let shift: u32 = consecutive_errors.min(6);
                                let backoff =
                                    Duration::from_millis(50u64.saturating_mul(1u64 << shift));
                                tokio::time::sleep(backoff).await;
                                continue;
                            }
                        }
                    }
                }
            }
            Ok(())
        });

        Ok(handle)
    }

    /// Spawn pipeline processor task: runs every registered stage over each
    /// batch in order, then writes the result to `sink`.
    ///
    /// On a stage error, `error_handler` (if set via `set_error_handler`)
    /// decides the [`ErrorAction`]; otherwise `config.error_strategy`
    /// implies one (`FailFast` -> `Abort`, `SkipErrors` -> `Continue`,
    /// `RetryWithBackoff` -> `Continue`). Note that `Retry` is not
    /// meaningfully implementable for a *stage* error with the current
    /// [`PipelineStage::process`] signature: on error, the input batch that
    /// was passed in is gone (the trait only returns the transformed batch
    /// on success), so there is nothing left to retry with -- this is
    /// therefore treated the same as `Continue` (skip this batch) rather
    /// than silently pretending to retry. A sink write/flush/close error is
    /// always terminal and propagated (previously, both were silently
    /// swallowed via `let _ = ...`, and the loop `break`ing on a write error
    /// still returned `Ok(())`, i.e. truncating the stream while reporting
    /// success).
    async fn spawn_pipeline_processor<K>(
        &mut self,
        mut rx: mpsc::Receiver<Vec<T>>,
        mut sink: K,
    ) -> Result<tokio::task::JoinHandle<Result<()>>>
    where
        K: StreamingDataSink<Item = T> + 'static,
    {
        let stats = Arc::clone(&self.stats);
        let mut stages = std::mem::take(&mut self.stages);
        let error_handler = self.error_handler.take();
        let error_strategy = self.config.error_strategy;

        let handle = tokio::spawn(async move {
            while let Some(batch) = rx.recv().await {
                // `Option<Vec<T>>` (rather than a bare `Vec<T>` reassigned
                // in a loop with a separate `bool` "skip" flag) so the
                // borrow checker can see, from the type alone, that using
                // the batch below is only ever reached in the well-defined
                // `Some` case -- a bare `Vec<T>` moved into a failing
                // `stage.process(...)` call with no reassignment on that
                // path left it in a statically "possibly moved" state at
                // the point of the sink write, even though a runtime-only
                // `bool` flag guaranteed that path was always skipped.
                let mut current_batch: Option<Vec<T>> = Some(batch);
                let mut abort_error: Option<Error> = None;

                for stage in stages.iter_mut() {
                    let Some(input) = current_batch.take() else {
                        break;
                    };
                    match stage.process(input).await {
                        Ok(next_batch) => current_batch = Some(next_batch),
                        Err(e) => {
                            if let Ok(mut pipeline_stats) = stats.lock() {
                                pipeline_stats.record_error();
                            }

                            let action = match &error_handler {
                                Some(handler) => handler.handle_error(&e),
                                None => match error_strategy {
                                    ErrorStrategy::FailFast => ErrorAction::Abort,
                                    ErrorStrategy::SkipErrors | ErrorStrategy::RetryWithBackoff => {
                                        ErrorAction::Continue
                                    }
                                },
                            };

                            match action {
                                ErrorAction::Abort => abort_error = Some(e),
                                // `Retry` cannot recover the input batch
                                // that `stage.process` just consumed (see
                                // this method's doc comment), so it is
                                // treated the same as `Continue`: skip this
                                // batch. `current_batch` is left `None`.
                                ErrorAction::Continue | ErrorAction::Retry => {}
                            }
                            break;
                        }
                    }
                }

                if let Some(e) = abort_error {
                    return Err(e);
                }

                let Some(current_batch) = current_batch else {
                    // A stage declined to continue (and it wasn't an
                    // abort): skip this batch entirely.
                    continue;
                };

                // Write to sink
                if let Err(e) = sink.write_batch(current_batch.clone()).await {
                    if let Ok(mut pipeline_stats) = stats.lock() {
                        pipeline_stats.record_error();
                    }
                    return Err(Error::InvalidOperation(format!("Sink write error: {}", e)));
                }

                if let Ok(mut pipeline_stats) = stats.lock() {
                    pipeline_stats.record_batch_processed(current_batch.len());
                }
            }

            sink.flush()
                .await
                .map_err(|e| Error::InvalidOperation(format!("Sink flush error: {}", e)))?;
            sink.close()
                .await
                .map_err(|e| Error::InvalidOperation(format!("Sink close error: {}", e)))?;
            Ok(())
        });

        Ok(handle)
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> Result<PipelineStats> {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))
    }
}

/// Pipeline stage trait
pub trait PipelineStage<T>: Send + Sync {
    /// Process data through this stage
    fn process(
        &mut self,
        data: Vec<T>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<T>>> + Send + '_>>;

    /// Get stage name
    fn name(&self) -> &str;

    /// Get stage statistics
    fn stats(&self) -> StageStats;
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Buffer size for inter-stage communication
    pub buffer_size: usize,
    /// Maximum parallelism
    pub max_parallelism: usize,
    /// Timeout for operations
    pub timeout: Duration,
    /// Error handling strategy
    pub error_strategy: ErrorStrategy,
    /// Checkpoint interval
    pub checkpoint_interval: Option<Duration>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_parallelism: num_cpus::get(),
            timeout: Duration::from_secs(30),
            error_strategy: ErrorStrategy::FailFast,
            checkpoint_interval: Some(Duration::from_secs(60)),
        }
    }
}

/// Error handling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorStrategy {
    FailFast,
    SkipErrors,
    RetryWithBackoff,
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total batches read
    pub batches_read: u64,
    /// Total batches processed
    pub batches_processed: u64,
    /// Total items read
    pub items_read: u64,
    /// Total items processed
    pub items_processed: u64,
    /// Processing start time
    pub start_time: Option<Instant>,
    /// Processing end time
    pub end_time: Option<Instant>,
    /// Error count
    pub error_count: u64,
}

impl PipelineStats {
    pub fn new() -> Self {
        Self {
            batches_read: 0,
            batches_processed: 0,
            items_read: 0,
            items_processed: 0,
            start_time: None,
            end_time: None,
            error_count: 0,
        }
    }

    pub fn record_batch_read(&mut self, items: usize) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }
        self.batches_read += 1;
        self.items_read += items as u64;
    }

    pub fn record_batch_processed(&mut self, items: usize) {
        self.batches_processed += 1;
        self.items_processed += items as u64;
        self.end_time = Some(Instant::now());
    }

    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    pub fn duration(&self) -> Option<Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    pub fn throughput(&self) -> Option<f64> {
        self.duration().map(|d| {
            if d.as_secs_f64() > 0.0 {
                self.items_processed as f64 / d.as_secs_f64()
            } else {
                0.0
            }
        })
    }
}

/// Stage statistics
#[derive(Debug, Clone)]
pub struct StageStats {
    /// Items processed by this stage
    pub items_processed: u64,
    /// Processing time
    pub processing_time: Duration,
    /// Error count
    pub error_count: u64,
}

/// Error handler trait
pub trait ErrorHandler: Send + Sync {
    /// Handle an error
    fn handle_error(&self, error: &dyn std::error::Error) -> ErrorAction;

    /// Get error statistics
    fn error_stats(&self) -> ErrorStats;
}

/// Error action enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorAction {
    Continue,
    Retry,
    Abort,
}

/// Error statistics
#[derive(Debug, Clone)]
pub struct ErrorStats {
    /// Total errors handled
    pub total_errors: u64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Last error time
    pub last_error_time: Option<Instant>,
}

/// Memory-based streaming data source for testing
pub struct MemoryStreamSource<T> {
    data: Vec<Vec<T>>,
    current_index: usize,
    batch_size: usize,
    metadata: StreamMetadata,
}

impl<T> MemoryStreamSource<T>
where
    T: Clone + Send + Sync,
{
    pub fn new(data: Vec<T>, batch_size: usize) -> Self {
        let total_items = data.len();
        let batches: Vec<Vec<T>> = data
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let metadata = StreamMetadata {
            id: "memory_stream".to_string(),
            stream_type: StreamType::Memory,
            schema: None,
            estimated_size: Some(total_items),
            created_at: Instant::now(),
            properties: HashMap::new(),
        };

        Self {
            data: batches,
            current_index: 0,
            batch_size,
            metadata,
        }
    }
}

impl<T> StreamingDataSource for MemoryStreamSource<T>
where
    T: Clone + Send + Sync + 'static,
{
    type Item = T;
    type Error = Error;

    fn next_batch(
        &mut self,
    ) -> Pin<
        Box<
            dyn Future<Output = std::result::Result<Option<Vec<Self::Item>>, Self::Error>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            if self.current_index < self.data.len() {
                let batch = self.data[self.current_index].clone();
                self.current_index += 1;
                Ok(Some(batch))
            } else {
                Ok(None)
            }
        })
    }

    fn has_more(&self) -> bool {
        self.current_index < self.data.len()
    }

    fn metadata(&self) -> StreamMetadata {
        self.metadata.clone()
    }

    fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    fn reset(&mut self) -> std::result::Result<(), Self::Error> {
        self.current_index = 0;
        Ok(())
    }

    fn estimated_size(&self) -> Option<usize> {
        self.metadata.estimated_size
    }
}

/// Memory-based streaming data sink for testing
pub struct MemoryStreamSink<T> {
    data: Arc<Mutex<Vec<T>>>,
    max_batch_size: usize,
    metadata: SinkMetadata,
}

// Manual `Clone` (rather than `#[derive(Clone)]`, which would add an
// unnecessary `T: Clone` bound): all fields are cheap to duplicate as
// *handles* onto the same underlying data (`Arc::clone`, not a deep copy of
// the buffered `Vec<T>`), which is exactly what makes this useful for tests
// that need to inspect what a sink received after moving one clone of it
// into `StreamingPipeline::execute` (which consumes its sink by value).
impl<T> Clone for MemoryStreamSink<T> {
    fn clone(&self) -> Self {
        MemoryStreamSink {
            data: Arc::clone(&self.data),
            max_batch_size: self.max_batch_size,
            metadata: self.metadata.clone(),
        }
    }
}

impl<T> MemoryStreamSink<T>
where
    T: Clone + Send + Sync,
{
    pub fn new() -> Self {
        let metadata = SinkMetadata {
            id: "memory_sink".to_string(),
            sink_type: SinkType::Memory,
            schema: None,
            created_at: Instant::now(),
            properties: HashMap::new(),
        };

        Self {
            data: Arc::new(Mutex::new(Vec::new())),
            max_batch_size: 1000,
            metadata,
        }
    }

    pub fn get_data(&self) -> Result<Vec<T>> {
        self.data
            .lock()
            .map(|data| data.clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire data lock".to_string()))
    }
}

impl<T> StreamingDataSink for MemoryStreamSink<T>
where
    T: Clone + Send + Sync + 'static,
{
    type Item = T;
    type Error = Error;

    fn write_batch(
        &mut self,
        batch: Vec<Self::Item>,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Self::Error>> + Send + '_>> {
        Box::pin(async move {
            if let Ok(mut data) = self.data.lock() {
                data.extend(batch);
                Ok(())
            } else {
                Err(Error::InvalidOperation(
                    "Failed to acquire data lock".to_string(),
                ))
            }
        })
    }

    fn flush(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Self::Error>> + Send + '_>> {
        Box::pin(async move {
            // No-op for memory sink
            Ok(())
        })
    }

    fn close(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Self::Error>> + Send + '_>> {
        Box::pin(async move {
            // No-op for memory sink
            Ok(())
        })
    }

    fn metadata(&self) -> SinkMetadata {
        self.metadata.clone()
    }

    fn set_max_batch_size(&mut self, size: usize) {
        self.max_batch_size = size;
    }
}

/// DataFrame streaming operations
pub trait DataFrameStreaming {
    /// Create a streaming source from the DataFrame
    fn to_streaming_source(
        &self,
        batch_size: usize,
    ) -> Result<Box<dyn StreamingDataSource<Item = DataFrame, Error = Error>>>;

    /// Create a DataFrame from a streaming source
    fn from_streaming_source<S>(source: S) -> Pin<Box<dyn Future<Output = Result<Self>> + Send>>
    where
        S: StreamingDataSource<Item = DataFrame, Error = Error> + 'static,
        Self: Sized;

    /// Process DataFrame in streaming chunks
    fn process_streaming<F>(
        &self,
        batch_size: usize,
        processor: F,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>
    where
        F: Fn(DataFrame) -> Pin<Box<dyn Future<Output = Result<DataFrame>> + Send + 'static>>
            + Send
            + Sync
            + 'static;
}

/// Windowing operations for streaming data
pub struct StreamWindow<T> {
    /// Window type
    window_type: WindowType,
    /// Window size
    window_size: Duration,
    /// Slide interval
    slide_interval: Duration,
    /// Current window data
    current_window: Vec<(Instant, T)>,
    /// Window statistics
    stats: WindowStats,
}

impl<T> StreamWindow<T>
where
    T: Clone + Send + Sync,
{
    pub fn new(window_type: WindowType, window_size: Duration, slide_interval: Duration) -> Self {
        Self {
            window_type,
            window_size,
            slide_interval,
            current_window: Vec::new(),
            stats: WindowStats::new(),
        }
    }

    /// Add data to the window
    pub fn add(&mut self, timestamp: Instant, data: T) {
        self.current_window.push((timestamp, data));
        self.evict_expired(timestamp);
        self.stats.items_added += 1;
    }

    /// Get current window data
    pub fn current_data(&self) -> Vec<&T> {
        self.current_window.iter().map(|(_, data)| data).collect()
    }

    /// Check if window is ready to emit
    pub fn is_ready(&self, current_time: Instant) -> bool {
        match self.window_type {
            WindowType::Tumbling => {
                if let Some((first_time, _)) = self.current_window.first() {
                    current_time.duration_since(*first_time) >= self.window_size
                } else {
                    false
                }
            }
            WindowType::Sliding => {
                if let Some(last_emit) = self.stats.last_emit_time {
                    current_time.duration_since(last_emit) >= self.slide_interval
                } else {
                    !self.current_window.is_empty()
                }
            }
            WindowType::Session => {
                // Session windows close when no data arrives for the slide_interval
                if let Some((last_time, _)) = self.current_window.last() {
                    current_time.duration_since(*last_time) >= self.slide_interval
                } else {
                    false
                }
            }
        }
    }

    /// Emit window data and reset for next window
    pub fn emit(&mut self, current_time: Instant) -> Vec<T> {
        let data = self
            .current_window
            .iter()
            .map(|(_, data)| data.clone())
            .collect();

        match self.window_type {
            WindowType::Tumbling => {
                self.current_window.clear();
            }
            WindowType::Sliding => {
                // Keep data that's still within the window
                self.evict_expired(current_time);
            }
            WindowType::Session => {
                self.current_window.clear();
            }
        }

        self.stats.windows_emitted += 1;
        self.stats.last_emit_time = Some(current_time);

        data
    }

    /// Remove expired data from the window
    fn evict_expired(&mut self, current_time: Instant) {
        let cutoff_time = current_time - self.window_size;
        self.current_window
            .retain(|(timestamp, _)| *timestamp >= cutoff_time);
    }

    /// Get window statistics
    pub fn stats(&self) -> &WindowStats {
        &self.stats
    }
}

/// Window type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Tumbling,
    Sliding,
    Session,
}

/// Window statistics
#[derive(Debug, Clone)]
pub struct WindowStats {
    /// Total items added to windows
    pub items_added: u64,
    /// Total windows emitted
    pub windows_emitted: u64,
    /// Last emit time
    pub last_emit_time: Option<Instant>,
}

impl WindowStats {
    pub fn new() -> Self {
        Self {
            items_added: 0,
            windows_emitted: 0,
            last_emit_time: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_memory_stream_source() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut source = MemoryStreamSource::new(data, 3);

        assert!(source.has_more());
        assert_eq!(source.estimated_size(), Some(10));

        let batch1 = source
            .next_batch()
            .await
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert_eq!(batch1, vec![1, 2, 3]);

        let batch2 = source
            .next_batch()
            .await
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert_eq!(batch2, vec![4, 5, 6]);

        let batch3 = source
            .next_batch()
            .await
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert_eq!(batch3, vec![7, 8, 9]);

        let batch4 = source
            .next_batch()
            .await
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert_eq!(batch4, vec![10]);

        let batch5 = source.next_batch().await.expect("operation should succeed");
        assert!(batch5.is_none());
        assert!(!source.has_more());
    }

    #[tokio::test]
    async fn test_memory_stream_sink() {
        let mut sink = MemoryStreamSink::new();

        sink.write_batch(vec![1, 2, 3])
            .await
            .expect("operation should succeed");
        sink.write_batch(vec![4, 5, 6])
            .await
            .expect("operation should succeed");
        sink.flush().await.expect("operation should succeed");

        let data = sink.get_data().expect("operation should succeed");
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6]);

        sink.close().await.expect("operation should succeed");
    }

    #[tokio::test]
    async fn test_streaming_pipeline() {
        let config = PipelineConfig::default();
        let mut pipeline = StreamingPipeline::new(config);

        let source_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let source = MemoryStreamSource::new(source_data, 3);
        let sink = MemoryStreamSink::new();

        pipeline
            .execute(source, sink)
            .await
            .expect("operation should succeed");

        let stats = pipeline.stats().expect("operation should succeed");
        assert!(stats.items_read > 0);
        assert!(stats.items_processed > 0);
    }

    /// A trivial stage that doubles every value, used to prove that
    /// registered stages actually run (previously `add_stage` recorded a
    /// stage but `execute` never invoked it, so the sink always received
    /// the source's data completely unmodified).
    struct DoublingStage;

    impl PipelineStage<i32> for DoublingStage {
        fn process(
            &mut self,
            data: Vec<i32>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<i32>>> + Send + '_>> {
            Box::pin(async move { Ok(data.into_iter().map(|v| v * 2).collect()) })
        }

        fn name(&self) -> &str {
            "doubling"
        }

        fn stats(&self) -> StageStats {
            StageStats {
                items_processed: 0,
                processing_time: Duration::from_secs(0),
                error_count: 0,
            }
        }
    }

    #[tokio::test]
    async fn test_streaming_pipeline_executes_registered_stages() {
        let config = PipelineConfig::default();
        let mut pipeline = StreamingPipeline::new(config);
        pipeline.add_stage(DoublingStage);

        let source_data = vec![1, 2, 3, 4, 5];
        let source = MemoryStreamSource::new(source_data, 2);
        let sink = MemoryStreamSink::new();
        let sink_handle = sink.clone();

        pipeline
            .execute(source, sink)
            .await
            .expect("operation should succeed");

        let data = sink_handle.get_data().expect("operation should succeed");
        assert_eq!(data, vec![2, 4, 6, 8, 10]);
    }

    /// A sink whose `write_batch` always fails, used to verify that a
    /// terminal sink error is propagated out of `execute` as an `Err`
    /// rather than being swallowed (`let _ = sink.write_batch(...)`) while
    /// the pipeline reports `Ok(())` after silently truncating the stream.
    struct FailingSink;

    impl StreamingDataSink for FailingSink {
        type Item = i32;
        type Error = Error;

        fn write_batch(
            &mut self,
            _batch: Vec<i32>,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Error>> + Send + '_>> {
            Box::pin(async move { Err(Error::IoError("simulated sink failure".into())) })
        }

        fn flush(
            &mut self,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Error>> + Send + '_>> {
            Box::pin(async move { Ok(()) })
        }

        fn close(
            &mut self,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<(), Error>> + Send + '_>> {
            Box::pin(async move { Ok(()) })
        }

        fn metadata(&self) -> SinkMetadata {
            SinkMetadata {
                id: "failing".to_string(),
                sink_type: SinkType::Memory,
                schema: None,
                created_at: Instant::now(),
                properties: HashMap::new(),
            }
        }

        fn set_max_batch_size(&mut self, _size: usize) {}
    }

    #[tokio::test]
    async fn test_streaming_pipeline_propagates_sink_error() {
        let config = PipelineConfig {
            error_strategy: ErrorStrategy::FailFast,
            ..PipelineConfig::default()
        };
        let mut pipeline = StreamingPipeline::new(config);

        let source = MemoryStreamSource::new(vec![1, 2, 3], 1);
        let sink = FailingSink;

        let result = pipeline.execute(source, sink).await;
        assert!(
            result.is_err(),
            "a terminal sink error must propagate as Err, not be swallowed as Ok(())"
        );

        let stats = pipeline.stats().expect("operation should succeed");
        assert!(
            stats.error_count > 0,
            "the sink error should have been recorded via record_error()"
        );
    }

    #[test]
    fn test_stream_window() {
        let mut window = StreamWindow::new(
            WindowType::Tumbling,
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        let base_time = Instant::now();

        window.add(base_time, 1);
        window.add(base_time + Duration::from_secs(1), 2);
        window.add(base_time + Duration::from_secs(2), 3);

        assert_eq!(window.current_data(), vec![&1, &2, &3]);
        assert!(!window.is_ready(base_time + Duration::from_secs(3)));
        assert!(window.is_ready(base_time + Duration::from_secs(6)));

        let emitted = window.emit(base_time + Duration::from_secs(6));
        assert_eq!(emitted, vec![1, 2, 3]);
        assert!(window.current_data().is_empty());
    }

    #[test]
    fn test_stream_schema() {
        let schema = StreamSchema {
            fields: vec![
                StreamField {
                    name: "id".to_string(),
                    data_type: StreamDataType::Int64,
                    nullable: false,
                    metadata: HashMap::new(),
                },
                StreamField {
                    name: "name".to_string(),
                    data_type: StreamDataType::String,
                    nullable: true,
                    metadata: HashMap::new(),
                },
                StreamField {
                    name: "scores".to_string(),
                    data_type: StreamDataType::List(Box::new(StreamDataType::Float64)),
                    nullable: false,
                    metadata: HashMap::new(),
                },
            ],
            metadata: HashMap::new(),
        };

        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.fields[0].name, "id");
        assert_eq!(schema.fields[0].data_type, StreamDataType::Int64);
        assert!(!schema.fields[0].nullable);

        assert_eq!(schema.fields[1].name, "name");
        assert_eq!(schema.fields[1].data_type, StreamDataType::String);
        assert!(schema.fields[1].nullable);

        if let StreamDataType::List(inner_type) = &schema.fields[2].data_type {
            assert_eq!(**inner_type, StreamDataType::Float64);
        } else {
            panic!("Expected List type");
        }
    }

    #[test]
    fn test_pipeline_stats() {
        let mut stats = PipelineStats::new();

        stats.record_batch_read(100);
        stats.record_batch_read(150);
        stats.record_batch_processed(90);
        stats.record_batch_processed(140);

        assert_eq!(stats.batches_read, 2);
        assert_eq!(stats.batches_processed, 2);
        assert_eq!(stats.items_read, 250);
        assert_eq!(stats.items_processed, 230);

        assert!(stats.start_time.is_some());
        assert!(stats.end_time.is_some());
        assert!(stats.duration().is_some());
    }
}
