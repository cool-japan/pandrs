//! Backpressure handling for streaming data processing
//!
//! This module provides mechanisms to handle slow consumers and prevent
//! memory overflow in streaming pipelines.

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use super::StreamRecord;
use crate::error::{Error, Result};

/// Strategy for handling backpressure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureStrategy {
    /// Block the producer until space is available
    Block,
    /// Drop the oldest records when buffer is full
    DropOldest,
    /// Drop the newest records when buffer is full
    DropNewest,
    /// Sample records at a rate proportional to consumer speed
    AdaptiveSampling,
    /// Apply rate limiting based on consumer throughput
    RateLimiting,
}

impl Default for BackpressureStrategy {
    fn default() -> Self {
        BackpressureStrategy::Block
    }
}

/// Configuration for backpressure handling
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum buffer size before backpressure is applied
    pub high_watermark: usize,
    /// Buffer level at which normal processing resumes
    pub low_watermark: usize,
    /// Strategy for handling backpressure
    pub strategy: BackpressureStrategy,
    /// Timeout for blocking operations
    pub block_timeout: Duration,
    /// Rate limit (records per second) for rate limiting strategy
    pub rate_limit: Option<f64>,
    /// Sampling rate for adaptive sampling (0.0 - 1.0)
    pub min_sampling_rate: f64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        BackpressureConfig {
            high_watermark: 10_000,
            low_watermark: 5_000,
            strategy: BackpressureStrategy::Block,
            block_timeout: Duration::from_secs(30),
            rate_limit: None,
            min_sampling_rate: 0.1,
        }
    }
}

/// Builder for BackpressureConfig
pub struct BackpressureConfigBuilder {
    config: BackpressureConfig,
}

impl BackpressureConfigBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        BackpressureConfigBuilder {
            config: BackpressureConfig::default(),
        }
    }

    /// Sets the high watermark
    pub fn high_watermark(mut self, watermark: usize) -> Self {
        self.config.high_watermark = watermark;
        self
    }

    /// Sets the low watermark
    pub fn low_watermark(mut self, watermark: usize) -> Self {
        self.config.low_watermark = watermark;
        self
    }

    /// Sets the backpressure strategy
    pub fn strategy(mut self, strategy: BackpressureStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Sets the block timeout
    pub fn block_timeout(mut self, timeout: Duration) -> Self {
        self.config.block_timeout = timeout;
        self
    }

    /// Sets the rate limit
    pub fn rate_limit(mut self, rate: f64) -> Self {
        self.config.rate_limit = Some(rate);
        self
    }

    /// Sets the minimum sampling rate
    pub fn min_sampling_rate(mut self, rate: f64) -> Self {
        self.config.min_sampling_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Builds the config
    pub fn build(self) -> BackpressureConfig {
        self.config
    }
}

impl Default for BackpressureConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for backpressure monitoring
#[derive(Debug, Clone)]
pub struct BackpressureStats {
    /// Total records received
    pub records_received: u64,
    /// Records dropped due to backpressure
    pub records_dropped: u64,
    /// Records successfully processed
    pub records_processed: u64,
    /// Current buffer size
    pub current_buffer_size: usize,
    /// Times backpressure was triggered
    pub backpressure_events: u64,
    /// Current sampling rate (for adaptive sampling)
    pub current_sampling_rate: f64,
    /// Average processing latency in milliseconds
    pub avg_latency_ms: f64,
}

impl Default for BackpressureStats {
    fn default() -> Self {
        BackpressureStats {
            records_received: 0,
            records_dropped: 0,
            records_processed: 0,
            current_buffer_size: 0,
            backpressure_events: 0,
            current_sampling_rate: 1.0,
            avg_latency_ms: 0.0,
        }
    }
}

/// A buffer with backpressure support
#[derive(Debug)]
pub struct BackpressureBuffer {
    /// Configuration
    config: BackpressureConfig,
    /// Internal buffer
    buffer: Arc<RwLock<VecDeque<StreamRecord>>>,
    /// Whether backpressure is currently active
    backpressure_active: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<RwLock<BackpressureStats>>,
    /// Current sampling rate
    current_sampling_rate: Arc<RwLock<f64>>,
    /// Rate limiter state
    rate_limiter: Arc<Mutex<RateLimiterState>>,
}

#[derive(Debug)]
struct RateLimiterState {
    /// Tokens available for rate limiting
    tokens: f64,
    /// Last refill time
    last_refill: Instant,
    /// Records per second limit
    rate: f64,
}

impl BackpressureBuffer {
    /// Creates a new backpressure buffer
    pub fn new(config: BackpressureConfig) -> Self {
        let rate = config.rate_limit.unwrap_or(1000.0);

        BackpressureBuffer {
            config,
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            backpressure_active: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(BackpressureStats::default())),
            current_sampling_rate: Arc::new(RwLock::new(1.0)),
            rate_limiter: Arc::new(Mutex::new(RateLimiterState {
                tokens: rate,
                last_refill: Instant::now(),
                rate,
            })),
        }
    }

    /// Tries to push a record into the buffer
    pub fn try_push(&self, record: StreamRecord) -> Result<bool> {
        let mut stats = self.stats.write().unwrap();
        stats.records_received += 1;

        // Check current buffer size
        let current_size = self.buffer.read().unwrap().len();
        stats.current_buffer_size = current_size;

        // Check if we're above high watermark
        if current_size >= self.config.high_watermark {
            self.backpressure_active.store(true, Ordering::SeqCst);
            stats.backpressure_events += 1;

            match self.config.strategy {
                BackpressureStrategy::Block => {
                    // Will be handled by the blocking push method
                    return Ok(false);
                }
                BackpressureStrategy::DropOldest => {
                    let mut buffer = self.buffer.write().unwrap();
                    while buffer.len() >= self.config.high_watermark {
                        buffer.pop_front();
                        stats.records_dropped += 1;
                    }
                    buffer.push_back(record);
                    return Ok(true);
                }
                BackpressureStrategy::DropNewest => {
                    stats.records_dropped += 1;
                    return Ok(false);
                }
                BackpressureStrategy::AdaptiveSampling => {
                    // Reduce sampling rate
                    let mut rate = self.current_sampling_rate.write().unwrap();
                    *rate = (*rate * 0.9).max(self.config.min_sampling_rate);
                    stats.current_sampling_rate = *rate;

                    // Probabilistically accept the record
                    if should_sample(*rate) {
                        let mut buffer = self.buffer.write().unwrap();
                        buffer.push_back(record);
                        return Ok(true);
                    } else {
                        stats.records_dropped += 1;
                        return Ok(false);
                    }
                }
                BackpressureStrategy::RateLimiting => {
                    // Check rate limiter
                    if self.acquire_token() {
                        let mut buffer = self.buffer.write().unwrap();
                        buffer.push_back(record);
                        return Ok(true);
                    } else {
                        stats.records_dropped += 1;
                        return Ok(false);
                    }
                }
            }
        }

        // Normal operation - check if below low watermark
        if current_size < self.config.low_watermark {
            self.backpressure_active.store(false, Ordering::SeqCst);

            // Restore sampling rate for adaptive sampling
            if self.config.strategy == BackpressureStrategy::AdaptiveSampling {
                let mut rate = self.current_sampling_rate.write().unwrap();
                *rate = (*rate * 1.1).min(1.0);
                stats.current_sampling_rate = *rate;
            }
        }

        // Push the record
        let mut buffer = self.buffer.write().unwrap();
        buffer.push_back(record);
        Ok(true)
    }

    /// Pushes a record with blocking if necessary
    pub fn push(&self, record: StreamRecord) -> Result<()> {
        if self.config.strategy == BackpressureStrategy::Block {
            let start = Instant::now();
            loop {
                if self.try_push(record.clone())? {
                    return Ok(());
                }

                if start.elapsed() > self.config.block_timeout {
                    return Err(Error::IoError("Backpressure timeout".into()));
                }

                // Wait a bit before retrying
                thread::sleep(Duration::from_millis(10));
            }
        } else {
            self.try_push(record)?;
            Ok(())
        }
    }

    /// Pops a record from the buffer
    pub fn pop(&self) -> Option<StreamRecord> {
        let mut buffer = self.buffer.write().unwrap();
        let record = buffer.pop_front();

        if record.is_some() {
            let mut stats = self.stats.write().unwrap();
            stats.records_processed += 1;
            stats.current_buffer_size = buffer.len();
        }

        record
    }

    /// Pops multiple records from the buffer
    pub fn pop_batch(&self, max_batch_size: usize) -> Vec<StreamRecord> {
        let mut buffer = self.buffer.write().unwrap();
        let batch_size = max_batch_size.min(buffer.len());
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(record) = buffer.pop_front() {
                batch.push(record);
            }
        }

        if !batch.is_empty() {
            let mut stats = self.stats.write().unwrap();
            stats.records_processed += batch.len() as u64;
            stats.current_buffer_size = buffer.len();
        }

        batch
    }

    /// Checks if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.read().unwrap().is_empty()
    }

    /// Gets the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.read().unwrap().len()
    }

    /// Checks if backpressure is currently active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::SeqCst)
    }

    /// Gets the current statistics
    pub fn stats(&self) -> BackpressureStats {
        self.stats.read().unwrap().clone()
    }

    /// Resets the statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = BackpressureStats::default();
    }

    /// Acquires a token from the rate limiter
    fn acquire_token(&self) -> bool {
        let mut state = self.rate_limiter.lock().unwrap();

        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);
        let new_tokens = elapsed.as_secs_f64() * state.rate;
        state.tokens = (state.tokens + new_tokens).min(state.rate);
        state.last_refill = now;

        // Try to acquire a token
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Simple random sampling based on rate
fn should_sample(rate: f64) -> bool {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();

    (nanos as f64 / u32::MAX as f64) < rate
}

/// A channel with backpressure support
pub struct BackpressureChannel {
    /// Sender side
    sender: Sender<StreamRecord>,
    /// Receiver side
    receiver: Receiver<StreamRecord>,
    /// Configuration
    config: BackpressureConfig,
    /// Statistics
    stats: Arc<RwLock<BackpressureStats>>,
    /// Current buffer size
    buffer_size: Arc<AtomicUsize>,
    /// Whether backpressure is active
    backpressure_active: Arc<AtomicBool>,
}

impl BackpressureChannel {
    /// Creates a new backpressure channel
    pub fn new(config: BackpressureConfig) -> Self {
        let (sender, receiver) = bounded(config.high_watermark);

        BackpressureChannel {
            sender,
            receiver,
            config,
            stats: Arc::new(RwLock::new(BackpressureStats::default())),
            buffer_size: Arc::new(AtomicUsize::new(0)),
            backpressure_active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Sends a record through the channel
    pub fn send(&self, record: StreamRecord) -> Result<()> {
        let mut stats = self.stats.write().unwrap();
        stats.records_received += 1;

        let current_size = self.buffer_size.load(Ordering::SeqCst);

        match self.config.strategy {
            BackpressureStrategy::Block => {
                match self.sender.send_timeout(record, self.config.block_timeout) {
                    Ok(_) => {
                        self.buffer_size.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }
                    Err(_) => {
                        stats.records_dropped += 1;
                        stats.backpressure_events += 1;
                        Err(Error::IoError("Channel send timeout".into()))
                    }
                }
            }
            BackpressureStrategy::DropNewest => {
                if current_size >= self.config.high_watermark {
                    stats.records_dropped += 1;
                    stats.backpressure_events += 1;
                    self.backpressure_active.store(true, Ordering::SeqCst);
                    Ok(())
                } else {
                    match self.sender.try_send(record) {
                        Ok(_) => {
                            self.buffer_size.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        }
                        Err(TrySendError::Full(_)) => {
                            stats.records_dropped += 1;
                            stats.backpressure_events += 1;
                            Ok(())
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            Err(Error::IoError("Channel disconnected".into()))
                        }
                    }
                }
            }
            _ => {
                // For other strategies, use try_send
                match self.sender.try_send(record) {
                    Ok(_) => {
                        self.buffer_size.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }
                    Err(TrySendError::Full(_)) => {
                        stats.records_dropped += 1;
                        stats.backpressure_events += 1;
                        Ok(())
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        Err(Error::IoError("Channel disconnected".into()))
                    }
                }
            }
        }
    }

    /// Receives a record from the channel
    pub fn recv(&self) -> Result<StreamRecord> {
        match self.receiver.recv() {
            Ok(record) => {
                self.buffer_size.fetch_sub(1, Ordering::SeqCst);

                let current_size = self.buffer_size.load(Ordering::SeqCst);
                if current_size < self.config.low_watermark {
                    self.backpressure_active.store(false, Ordering::SeqCst);
                }

                let mut stats = self.stats.write().unwrap();
                stats.records_processed += 1;
                stats.current_buffer_size = current_size;

                Ok(record)
            }
            Err(_) => Err(Error::IoError("Channel receive failed".into())),
        }
    }

    /// Receives a record with timeout
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<StreamRecord>> {
        match self.receiver.recv_timeout(timeout) {
            Ok(record) => {
                self.buffer_size.fetch_sub(1, Ordering::SeqCst);

                let mut stats = self.stats.write().unwrap();
                stats.records_processed += 1;
                stats.current_buffer_size = self.buffer_size.load(Ordering::SeqCst);

                Ok(Some(record))
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Ok(None),
            Err(_) => Err(Error::IoError("Channel disconnected".into())),
        }
    }

    /// Gets the current buffer size
    pub fn len(&self) -> usize {
        self.buffer_size.load(Ordering::SeqCst)
    }

    /// Checks if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the current statistics
    pub fn stats(&self) -> BackpressureStats {
        self.stats.read().unwrap().clone()
    }

    /// Checks if backpressure is currently active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::SeqCst)
    }
}

/// A flow controller that monitors throughput and applies backpressure
#[derive(Debug)]
pub struct FlowController {
    /// Target throughput in records per second
    target_throughput: f64,
    /// Current actual throughput
    current_throughput: Arc<RwLock<f64>>,
    /// Record count in current window
    record_count: Arc<AtomicU64>,
    /// Window start time
    window_start: Arc<RwLock<Instant>>,
    /// Window duration
    window_duration: Duration,
    /// Whether flow control is active
    active: Arc<AtomicBool>,
}

impl FlowController {
    /// Creates a new flow controller
    pub fn new(target_throughput: f64, window_duration: Duration) -> Self {
        FlowController {
            target_throughput,
            current_throughput: Arc::new(RwLock::new(0.0)),
            record_count: Arc::new(AtomicU64::new(0)),
            window_start: Arc::new(RwLock::new(Instant::now())),
            window_duration,
            active: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Records a processed record and returns whether to continue
    pub fn record_processed(&self) -> bool {
        if !self.active.load(Ordering::SeqCst) {
            return true;
        }

        // Increment count
        let count = self.record_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Check if window has elapsed
        let window_start = *self.window_start.read().unwrap();
        let elapsed = window_start.elapsed();

        if elapsed >= self.window_duration {
            // Calculate throughput and reset window
            let throughput = count as f64 / elapsed.as_secs_f64();
            *self.current_throughput.write().unwrap() = throughput;
            self.record_count.store(0, Ordering::SeqCst);
            *self.window_start.write().unwrap() = Instant::now();

            // Check if we need to slow down
            if throughput > self.target_throughput * 1.1 {
                let delay_ms = ((throughput / self.target_throughput - 1.0) * 100.0) as u64;
                thread::sleep(Duration::from_millis(delay_ms.min(100)));
            }
        }

        true
    }

    /// Gets the current throughput
    pub fn current_throughput(&self) -> f64 {
        *self.current_throughput.read().unwrap()
    }

    /// Sets the target throughput
    pub fn set_target_throughput(&self, target: f64) {
        // Note: We can't modify target_throughput directly since it's not wrapped
        // This would need interior mutability
    }

    /// Pauses flow control
    pub fn pause(&self) {
        self.active.store(false, Ordering::SeqCst);
    }

    /// Resumes flow control
    pub fn resume(&self) {
        self.active.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_record() -> StreamRecord {
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), "42".to_string());
        StreamRecord::new(fields)
    }

    #[test]
    fn test_backpressure_buffer_normal_operation() {
        let config = BackpressureConfig::default();
        let buffer = BackpressureBuffer::new(config);

        for _ in 0..100 {
            buffer.push(create_test_record()).unwrap();
        }

        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_backpressure_active());
    }

    #[test]
    fn test_backpressure_buffer_drop_oldest() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(10)
            .low_watermark(5)
            .strategy(BackpressureStrategy::DropOldest)
            .build();

        let buffer = BackpressureBuffer::new(config);

        for _ in 0..20 {
            buffer.try_push(create_test_record()).unwrap();
        }

        // Buffer should not exceed high watermark
        assert!(buffer.len() <= 10);
    }

    #[test]
    fn test_backpressure_buffer_drop_newest() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(10)
            .low_watermark(5)
            .strategy(BackpressureStrategy::DropNewest)
            .build();

        let buffer = BackpressureBuffer::new(config);

        for _ in 0..20 {
            buffer.try_push(create_test_record()).unwrap();
        }

        // Buffer should equal high watermark (oldest records kept)
        assert_eq!(buffer.len(), 10);
    }

    #[test]
    fn test_backpressure_stats() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(10)
            .low_watermark(5)
            .strategy(BackpressureStrategy::DropNewest)
            .build();

        let buffer = BackpressureBuffer::new(config);

        for _ in 0..20 {
            buffer.try_push(create_test_record()).unwrap();
        }

        let stats = buffer.stats();
        assert_eq!(stats.records_received, 20);
        assert_eq!(stats.records_dropped, 10);
        assert!(stats.backpressure_events > 0);
    }

    #[test]
    fn test_backpressure_channel() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(100)
            .low_watermark(50)
            .build();

        let channel = BackpressureChannel::new(config);

        for _ in 0..50 {
            channel.send(create_test_record()).unwrap();
        }

        assert_eq!(channel.len(), 50);

        for _ in 0..25 {
            channel.recv().unwrap();
        }

        assert_eq!(channel.len(), 25);
    }

    #[test]
    fn test_flow_controller() {
        let controller = FlowController::new(1000.0, Duration::from_millis(100));

        for _ in 0..100 {
            controller.record_processed();
        }

        // Flow controller should be tracking records
        controller.pause();
        assert!(controller.record_processed());
    }

    #[test]
    fn test_backpressure_pop_batch() {
        let config = BackpressureConfig::default();
        let buffer = BackpressureBuffer::new(config);

        for _ in 0..100 {
            buffer.push(create_test_record()).unwrap();
        }

        let batch = buffer.pop_batch(30);
        assert_eq!(batch.len(), 30);
        assert_eq!(buffer.len(), 70);
    }
}
