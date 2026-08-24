//! Backpressure handling for streaming data processing
//!
//! This module provides mechanisms to handle slow consumers and prevent
//! memory overflow in streaming pipelines.
//!
//! # Locking discipline
//!
//! [`BackpressureBuffer`] guards two independent pieces of state -- the
//! record buffer (`buffer: RwLock<VecDeque<...>>`) and the reporting
//! counters (`stats: RwLock<BackpressureStats>`) -- with two separate
//! locks. Earlier versions of `try_push` acquired `stats` first and then
//! `buffer` while still holding it, while `pop`/`pop_batch` acquired
//! `buffer` first and then (for a successful pop) `stats` -- classic ABBA
//! lock ordering between concurrent pushers and poppers. Every method here
//! is written to hold at most one of these two locks at a time (compute
//! everything that needs `buffer`, release it, *then* separately acquire
//! `stats` to record the outcome), which makes that deadlock structurally
//! impossible rather than merely unlikely.

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use scirs2_core::random::{rng, RngExt};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use super::StreamRecord;
use crate::error::{Error, Result};
use crate::{lock_safe, read_lock_safe, write_lock_safe};

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
    /// Total records received (counted once per logical push call, even if
    /// a `Block`-strategy push internally retries several times).
    pub records_received: u64,
    /// Records dropped due to backpressure
    pub records_dropped: u64,
    /// Records successfully processed
    pub records_processed: u64,
    /// Current buffer size
    pub current_buffer_size: usize,
    /// Number of times backpressure transitioned from inactive to active
    /// (edge-triggered: a run of many consecutive drops while backpressure
    /// stays continuously active counts as one event, not one per drop).
    pub backpressure_events: u64,
    /// Current sampling rate (for adaptive sampling)
    pub current_sampling_rate: f64,
    /// Average processing latency in milliseconds: the mean time between a
    /// record's `StreamRecord` creation timestamp and it being dequeued via
    /// `pop`/`pop_batch`/`recv`/`recv_timeout`, updated incrementally.
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

impl BackpressureStats {
    /// Incorporates one more latency sample into `avg_latency_ms` via an
    /// incremental (streaming) mean, given `records_processed` has already
    /// been incremented to include this sample.
    fn record_latency_sample(&mut self, latency_ms: f64) {
        let n = self.records_processed as f64;
        if n > 0.0 {
            self.avg_latency_ms += (latency_ms - self.avg_latency_ms) / n;
        } else {
            self.avg_latency_ms = latency_ms;
        }
    }
}

/// Outcome of attempting to push a record into a [`BackpressureBuffer`].
///
/// Replaces the previous `bool`-returning `try_push`, whose `false` result
/// conflated two very different situations: "the record was dropped
/// (`DropNewest`/`AdaptiveSampling`/`RateLimiting` declined to admit it)"
/// and, implicitly, "the buffer is full and using `Block` strategy, so the
/// blocking `push` wrapper should keep retrying". Callers that need to
/// distinguish those can match on this enum directly instead of a bare
/// `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// The record was enqueued.
    Enqueued,
    /// The record was dropped by the configured strategy.
    Dropped,
    /// The buffer is at or above the high watermark and the strategy is
    /// `Block`: the record was **not** enqueued and the caller should retry
    /// (this is what `push()` does) or apply its own backoff.
    WouldBlock,
}

impl PushOutcome {
    /// True if the record ended up in the buffer.
    pub fn is_enqueued(self) -> bool {
        matches!(self, PushOutcome::Enqueued)
    }
}

/// A simple token-bucket rate limiter: tokens accumulate at `rate` tokens
/// per second, capped at `rate` (i.e. bursts of at most one second's worth
/// at the configured rate), and each admitted item consumes one token.
///
/// Shared by [`BackpressureBuffer`]'s `RateLimiting` strategy and
/// [`FlowController`]. Previously, `FlowController` implemented its own
/// window-measure-then-sleep throttling instead of reusing this
/// already-correct primitive (which existed, but only inside
/// `BackpressureBuffer`, unused by `FlowController`).
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    rate: f64,
}

impl TokenBucket {
    fn new(rate: f64) -> Self {
        let rate = rate.max(0.0);
        TokenBucket {
            tokens: rate,
            last_refill: Instant::now(),
            rate,
        }
    }

    fn set_rate(&mut self, rate: f64) {
        self.rate = rate.max(0.0);
    }

    /// Refills based on elapsed time since the last call, then tries to
    /// consume one token.
    fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens = elapsed.as_secs_f64() * self.rate;
        self.tokens = (self.tokens + new_tokens).min(self.rate);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// A buffer with backpressure support
#[derive(Debug)]
pub struct BackpressureBuffer {
    /// Configuration
    config: BackpressureConfig,
    /// Internal buffer
    buffer: RwLock<VecDeque<StreamRecord>>,
    /// Whether backpressure is currently active
    backpressure_active: AtomicBool,
    /// Statistics
    stats: RwLock<BackpressureStats>,
    /// Current sampling rate
    current_sampling_rate: RwLock<f64>,
    /// Rate limiter state, for the `RateLimiting` strategy
    rate_limiter: Mutex<TokenBucket>,
}

/// Result of one push attempt, computed entirely under the `buffer` lock
/// and applied to `stats` afterward (see the module-level doc comment on
/// locking discipline).
struct PushAttempt {
    outcome: PushOutcome,
    dropped_count: u64,
    became_backpressure_event: bool,
    sampling_rate_after: Option<f64>,
}

impl BackpressureBuffer {
    /// Creates a new backpressure buffer
    pub fn new(config: BackpressureConfig) -> Self {
        let rate = config.rate_limit.unwrap_or(1000.0);

        BackpressureBuffer {
            config,
            buffer: RwLock::new(VecDeque::new()),
            backpressure_active: AtomicBool::new(false),
            stats: RwLock::new(BackpressureStats::default()),
            current_sampling_rate: RwLock::new(1.0),
            rate_limiter: Mutex::new(TokenBucket::new(rate)),
        }
    }

    /// Performs one push attempt under a single `buffer` write lock
    /// (fixing a check-then-act race between the watermark check and the
    /// push/evict itself), without touching `stats` -- callers apply the
    /// resulting counts to `stats` afterward via a separate, non-nested
    /// lock acquisition.
    fn attempt_push(&self, record: StreamRecord) -> Result<PushAttempt> {
        let mut buffer = write_lock_safe!(self.buffer, "backpressure buffer write")?;
        let current_size = buffer.len();

        if current_size >= self.config.high_watermark {
            let was_active = self.backpressure_active.swap(true, Ordering::SeqCst);
            let became_backpressure_event = !was_active;

            let (outcome, dropped_count, sampling_rate_after) = match self.config.strategy {
                BackpressureStrategy::Block => (PushOutcome::WouldBlock, 0u64, None),
                BackpressureStrategy::DropOldest => {
                    let mut dropped = 0u64;
                    while buffer.len() >= self.config.high_watermark {
                        buffer.pop_front();
                        dropped += 1;
                    }
                    buffer.push_back(record);
                    (PushOutcome::Enqueued, dropped, None)
                }
                BackpressureStrategy::DropNewest => (PushOutcome::Dropped, 1, None),
                BackpressureStrategy::AdaptiveSampling => {
                    let rate_now = {
                        let mut rate =
                            write_lock_safe!(self.current_sampling_rate, "sampling rate write")?;
                        *rate = (*rate * 0.9).max(self.config.min_sampling_rate);
                        *rate
                    };
                    if should_sample(rate_now) {
                        buffer.push_back(record);
                        (PushOutcome::Enqueued, 0, Some(rate_now))
                    } else {
                        (PushOutcome::Dropped, 1, Some(rate_now))
                    }
                }
                BackpressureStrategy::RateLimiting => {
                    let admitted =
                        lock_safe!(self.rate_limiter, "rate limiter lock")?.try_acquire();
                    if admitted {
                        buffer.push_back(record);
                        (PushOutcome::Enqueued, 0, None)
                    } else {
                        (PushOutcome::Dropped, 1, None)
                    }
                }
            };

            Ok(PushAttempt {
                outcome,
                dropped_count,
                became_backpressure_event,
                sampling_rate_after,
            })
        } else {
            let mut sampling_rate_after = None;
            if current_size < self.config.low_watermark {
                self.backpressure_active.store(false, Ordering::SeqCst);

                if self.config.strategy == BackpressureStrategy::AdaptiveSampling {
                    let mut rate =
                        write_lock_safe!(self.current_sampling_rate, "sampling rate write")?;
                    *rate = (*rate * 1.1).min(1.0);
                    sampling_rate_after = Some(*rate);
                }
            }

            buffer.push_back(record);

            Ok(PushAttempt {
                outcome: PushOutcome::Enqueued,
                dropped_count: 0,
                became_backpressure_event: false,
                sampling_rate_after,
            })
        }
    }

    /// Applies one [`PushAttempt`]'s counts to `stats`, under a single
    /// `stats` lock acquisition that never overlaps with the `buffer` lock.
    /// `count_received` controls whether this call also counts toward
    /// `records_received`: exactly one of the (possibly several, for a
    /// blocking retry loop) attempts backing one logical push should set
    /// this.
    fn apply_attempt_to_stats(&self, attempt: &PushAttempt, count_received: bool) -> Result<()> {
        let current_size =
            read_lock_safe!(self.buffer, "backpressure buffer read for stats")?.len();
        let mut stats = write_lock_safe!(self.stats, "backpressure stats write")?;
        if count_received {
            stats.records_received += 1;
        }
        stats.records_dropped += attempt.dropped_count;
        if attempt.became_backpressure_event {
            stats.backpressure_events += 1;
        }
        if let Some(rate) = attempt.sampling_rate_after {
            stats.current_sampling_rate = rate;
        }
        // Eventually consistent: `buffer`'s lock was already released by
        // the time we read `current_size` above, so a concurrent push/pop
        // may have changed it since. That's an acceptable tradeoff for
        // never holding `buffer` and `stats` at once (see module doc
        // comment).
        stats.current_buffer_size = current_size;
        Ok(())
    }

    /// Tries to push a record into the buffer.
    pub fn try_push(&self, record: StreamRecord) -> Result<PushOutcome> {
        let attempt = self.attempt_push(record)?;
        self.apply_attempt_to_stats(&attempt, true)?;
        Ok(attempt.outcome)
    }

    /// Pushes a record with blocking if necessary.
    ///
    /// For `Block` strategy this may internally retry `attempt_push` many
    /// times while waiting for space; each logical call to `push` still
    /// counts as exactly one `records_received`, not one per retry
    /// iteration (the previous implementation called the equivalent of
    /// `try_push` -- which itself counted a receive -- on every ~10ms retry
    /// iteration, inflating `records_received` by roughly
    /// `block_timeout / 10ms` for a single blocked record).
    pub fn push(&self, record: StreamRecord) -> Result<()> {
        if self.config.strategy == BackpressureStrategy::Block {
            let start = Instant::now();
            let mut received_counted = false;

            loop {
                let attempt = self.attempt_push(record.clone())?;
                self.apply_attempt_to_stats(&attempt, !received_counted)?;
                received_counted = true;

                match attempt.outcome {
                    PushOutcome::Enqueued | PushOutcome::Dropped => return Ok(()),
                    PushOutcome::WouldBlock => {}
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
    pub fn pop(&self) -> Result<Option<StreamRecord>> {
        let (record, new_size) = {
            let mut buffer = write_lock_safe!(self.buffer, "backpressure buffer write")?;
            let record = buffer.pop_front();
            (record, buffer.len())
        };

        if let Some(ref rec) = record {
            let latency_ms = rec.timestamp.elapsed().as_secs_f64() * 1000.0;
            let mut stats = write_lock_safe!(self.stats, "backpressure stats write")?;
            stats.records_processed += 1;
            stats.current_buffer_size = new_size;
            stats.record_latency_sample(latency_ms);
        }

        Ok(record)
    }

    /// Pops multiple records from the buffer
    pub fn pop_batch(&self, max_batch_size: usize) -> Result<Vec<StreamRecord>> {
        let (batch, new_size) = {
            let mut buffer = write_lock_safe!(self.buffer, "backpressure buffer write")?;
            let batch_size = max_batch_size.min(buffer.len());
            let mut batch = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                if let Some(record) = buffer.pop_front() {
                    batch.push(record);
                }
            }
            (batch, buffer.len())
        };

        if !batch.is_empty() {
            let mut stats = write_lock_safe!(self.stats, "backpressure stats write")?;
            for record in &batch {
                let latency_ms = record.timestamp.elapsed().as_secs_f64() * 1000.0;
                stats.records_processed += 1;
                stats.record_latency_sample(latency_ms);
            }
            stats.current_buffer_size = new_size;
        }

        Ok(batch)
    }

    /// Checks if the buffer is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(read_lock_safe!(self.buffer, "backpressure buffer read")?.is_empty())
    }

    /// Gets the current buffer size
    pub fn len(&self) -> Result<usize> {
        Ok(read_lock_safe!(self.buffer, "backpressure buffer read")?.len())
    }

    /// Checks if backpressure is currently active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::SeqCst)
    }

    /// Gets the current statistics
    pub fn stats(&self) -> Result<BackpressureStats> {
        Ok(read_lock_safe!(self.stats, "backpressure stats read")?.clone())
    }

    /// Resets the statistics
    pub fn reset_stats(&self) -> Result<()> {
        let mut stats = write_lock_safe!(self.stats, "backpressure stats write")?;
        *stats = BackpressureStats::default();
        Ok(())
    }
}

/// Genuinely random sampling based on `rate` (probability in `[0.0, 1.0]`
/// that a record is kept).
///
/// The previous implementation derived a "random" value from
/// `SystemTime::now().subsec_nanos() as f64 / u32::MAX as f64`. Wall-clock
/// subsecond nanoseconds are not remotely uniform on most platforms (many
/// have coarser-than-nanosecond timer resolution, so the low bits are
/// frequently zero or otherwise patterned) and, worse, `subsec_nanos()` is
/// bounded by `999_999_999`, not `u32::MAX` (`4_294_967_295`) -- so the
/// computed ratio could never exceed roughly `0.2328`, meaning any
/// configured `rate` above that was accepted unconditionally (nothing was
/// ever dropped) while the effective acceptance rate for smaller
/// configured rates was inflated by roughly 4.3x. Using the crate's
/// scirs2_core RNG instead is both actually random and correctly spans the
/// full `[0.0, 1.0)` range.
fn should_sample(rate: f64) -> bool {
    rng().random_range(0.0..1.0) < rate
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
    stats: RwLock<BackpressureStats>,
    /// Current buffer size
    buffer_size: AtomicUsize,
    /// Whether backpressure is active
    backpressure_active: AtomicBool,
}

impl BackpressureChannel {
    /// Creates a new backpressure channel
    pub fn new(config: BackpressureConfig) -> Self {
        let (sender, receiver) = bounded(config.high_watermark);

        BackpressureChannel {
            sender,
            receiver,
            config,
            stats: RwLock::new(BackpressureStats::default()),
            buffer_size: AtomicUsize::new(0),
            backpressure_active: AtomicBool::new(false),
        }
    }

    /// Sends a record through the channel.
    ///
    /// The (potentially blocking, for `Block` strategy) send/try-send call
    /// itself is performed with no lock held; `stats` is only acquired
    /// afterward to record the outcome, so a producer waiting out
    /// `block_timeout` no longer serializes every other producer, the
    /// consumer, and `stats()` callers behind it for the duration of that
    /// wait.
    pub fn send(&self, record: StreamRecord) -> Result<()> {
        let current_size = self.buffer_size.load(Ordering::SeqCst);

        // (sent, dropped)
        let (sent, dropped) = match self.config.strategy {
            BackpressureStrategy::Block => {
                match self.sender.send_timeout(record, self.config.block_timeout) {
                    Ok(_) => {
                        self.buffer_size.fetch_add(1, Ordering::SeqCst);
                        (true, false)
                    }
                    Err(_) => (false, true),
                }
            }
            BackpressureStrategy::DropNewest if current_size >= self.config.high_watermark => {
                self.backpressure_active.store(true, Ordering::SeqCst);
                (false, true)
            }
            _ => match self.sender.try_send(record) {
                Ok(_) => {
                    self.buffer_size.fetch_add(1, Ordering::SeqCst);
                    (true, false)
                }
                Err(TrySendError::Full(_)) => (false, true),
                Err(TrySendError::Disconnected(_)) => {
                    return Err(Error::IoError("Channel disconnected".into()));
                }
            },
        };

        {
            let mut stats = write_lock_safe!(self.stats, "backpressure channel stats write")?;
            stats.records_received += 1;
            if dropped {
                stats.records_dropped += 1;
                stats.backpressure_events += 1;
            }
        }

        if !sent && self.config.strategy == BackpressureStrategy::Block {
            return Err(Error::IoError("Channel send timeout".into()));
        }

        Ok(())
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

                let latency_ms = record.timestamp.elapsed().as_secs_f64() * 1000.0;
                let mut stats = write_lock_safe!(self.stats, "backpressure channel stats write")?;
                stats.records_processed += 1;
                stats.current_buffer_size = current_size;
                stats.record_latency_sample(latency_ms);

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

                let latency_ms = record.timestamp.elapsed().as_secs_f64() * 1000.0;
                let mut stats = write_lock_safe!(self.stats, "backpressure channel stats write")?;
                stats.records_processed += 1;
                stats.current_buffer_size = self.buffer_size.load(Ordering::SeqCst);
                stats.record_latency_sample(latency_ms);

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
    pub fn stats(&self) -> Result<BackpressureStats> {
        Ok(read_lock_safe!(self.stats, "backpressure stats read")?.clone())
    }

    /// Checks if backpressure is currently active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::SeqCst)
    }
}

/// A flow controller that monitors throughput and applies backpressure.
///
/// Throttle *decisions* (`record_processed`'s return value) are driven by a
/// `TokenBucket` sized to `target_throughput`, reused from
/// [`BackpressureBuffer`]'s `RateLimiting` strategy rather than duplicated.
/// The window/`current_throughput()` machinery is retained for reporting
/// only, decoupled from the throttle decision itself.
#[derive(Debug)]
pub struct FlowController {
    /// Target throughput in records per second, stored as the bit pattern
    /// of an `f64` (via `f64::to_bits`/`from_bits`) so `set_target_throughput`
    /// can update it with a single atomic store rather than requiring
    /// `&mut self` or a lock -- the previous implementation's body was
    /// empty specifically because the field wasn't wrapped in anything
    /// that supported interior mutability, making every call a silent
    /// no-op.
    target_throughput_bits: AtomicU64,
    /// Token bucket used to make the actual throttle decision.
    limiter: Mutex<TokenBucket>,
    /// Current measured throughput (records/sec over the last closed
    /// window), for reporting via `current_throughput()`.
    current_throughput: RwLock<f64>,
    /// Record count accumulated in the window currently being measured.
    record_count: AtomicU64,
    /// Fixed instant this controller was constructed at: the epoch for
    /// `window_start_nanos`, so window rollover can be a single `AtomicU64`
    /// compare-and-swap instead of a separately locked `RwLock<Instant>`
    /// whose check-then-reset was a check-then-act race under concurrent
    /// callers.
    anchor: Instant,
    /// Nanoseconds since `anchor` at which the current measurement window
    /// started.
    window_start_nanos: AtomicU64,
    /// Window duration for throughput measurement.
    window_duration: Duration,
    /// Whether flow control is active.
    active: AtomicBool,
}

impl FlowController {
    /// Creates a new flow controller
    pub fn new(target_throughput: f64, window_duration: Duration) -> Self {
        let target = target_throughput.max(0.0);
        FlowController {
            target_throughput_bits: AtomicU64::new(target.to_bits()),
            limiter: Mutex::new(TokenBucket::new(target)),
            current_throughput: RwLock::new(0.0),
            record_count: AtomicU64::new(0),
            anchor: Instant::now(),
            window_start_nanos: AtomicU64::new(0),
            window_duration,
            active: AtomicBool::new(true),
        }
    }

    fn target_throughput_value(&self) -> f64 {
        f64::from_bits(self.target_throughput_bits.load(Ordering::Acquire))
    }

    /// Records a processed record and returns whether the caller should
    /// proceed now (`true`) or apply its own backoff (`false`) because
    /// `target_throughput` is currently exhausted.
    ///
    /// This used to unconditionally `thread::sleep` the calling thread for
    /// up to 100ms directly inside this call whenever the *previous*
    /// window's measured throughput exceeded target -- baking a specific
    /// policy (block the caller's thread) into what should be a
    /// measurement/decision primitive, and only re-evaluating once per
    /// window boundary (zero throttling pressure between boundaries).
    /// Returning a decision lets the caller sleep, drop the record, queue
    /// it, or ignore the signal, and the token bucket applies pressure
    /// continuously rather than only at window edges.
    pub fn record_processed(&self) -> Result<bool> {
        if !self.active.load(Ordering::SeqCst) {
            return Ok(true);
        }

        self.record_count.fetch_add(1, Ordering::SeqCst);

        let now_nanos = self.anchor.elapsed().as_nanos() as u64;
        let window_start = self.window_start_nanos.load(Ordering::Acquire);
        let elapsed_nanos = now_nanos.saturating_sub(window_start);
        let window_nanos = self.window_duration.as_nanos() as u64;

        if window_nanos > 0 && elapsed_nanos >= window_nanos {
            // Only the thread that wins this compare-exchange performs the
            // rollover; losers simply proceed without double-resetting or
            // clobbering another thread's in-flight window.
            if self
                .window_start_nanos
                .compare_exchange(window_start, now_nanos, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let count = self.record_count.swap(0, Ordering::SeqCst);
                let throughput = count as f64 / (elapsed_nanos as f64 / 1_000_000_000.0);
                *write_lock_safe!(self.current_throughput, "flow controller throughput write")? =
                    throughput;
            }
        }

        Ok(lock_safe!(self.limiter, "flow controller limiter lock")?.try_acquire())
    }

    /// Gets the current throughput
    pub fn current_throughput(&self) -> Result<f64> {
        Ok(*read_lock_safe!(
            self.current_throughput,
            "flow controller throughput read"
        )?)
    }

    /// Gets the currently configured target throughput.
    pub fn target_throughput(&self) -> f64 {
        self.target_throughput_value()
    }

    /// Sets the target throughput, updating both the reporting target and
    /// the token bucket's refill rate.
    pub fn set_target_throughput(&self, target: f64) {
        let target = target.max(0.0);
        self.target_throughput_bits
            .store(target.to_bits(), Ordering::Release);
        if let Ok(mut limiter) = lock_safe!(
            self.limiter,
            "flow controller limiter lock for set_target_throughput"
        ) {
            limiter.set_rate(target);
        }
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
            buffer
                .push(create_test_record())
                .expect("operation should succeed");
        }

        assert_eq!(buffer.len().expect("operation should succeed"), 100);
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
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed");
        }

        // Buffer should not exceed high watermark
        assert!(buffer.len().expect("operation should succeed") <= 10);
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
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed");
        }

        // Buffer should equal high watermark (oldest records kept)
        assert_eq!(buffer.len().expect("operation should succeed"), 10);
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
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed");
        }

        let stats = buffer.stats().expect("operation should succeed");
        assert_eq!(stats.records_received, 20);
        assert_eq!(stats.records_dropped, 10);
        assert!(stats.backpressure_events > 0);
    }

    #[test]
    fn test_push_outcome_distinguishes_enqueued_and_dropped() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(2)
            .low_watermark(1)
            .strategy(BackpressureStrategy::DropNewest)
            .build();
        let buffer = BackpressureBuffer::new(config);

        assert_eq!(
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed"),
            PushOutcome::Enqueued
        );
        assert_eq!(
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed"),
            PushOutcome::Enqueued
        );
        // Buffer is now at the high watermark (2); DropNewest declines further pushes.
        assert_eq!(
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed"),
            PushOutcome::Dropped
        );
    }

    #[test]
    fn test_push_outcome_would_block_for_block_strategy() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(1)
            .low_watermark(0)
            .strategy(BackpressureStrategy::Block)
            .build();
        let buffer = BackpressureBuffer::new(config);

        assert_eq!(
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed"),
            PushOutcome::Enqueued
        );
        // try_push (unlike push) never blocks/retries; it reports WouldBlock directly.
        assert_eq!(
            buffer
                .try_push(create_test_record())
                .expect("operation should succeed"),
            PushOutcome::WouldBlock
        );
    }

    #[test]
    fn test_backpressure_channel() {
        let config = BackpressureConfigBuilder::new()
            .high_watermark(100)
            .low_watermark(50)
            .build();

        let channel = BackpressureChannel::new(config);

        for _ in 0..50 {
            channel
                .send(create_test_record())
                .expect("operation should succeed");
        }

        assert_eq!(channel.len(), 50);

        for _ in 0..25 {
            channel.recv().expect("operation should succeed");
        }

        assert_eq!(channel.len(), 25);
    }

    #[test]
    fn test_flow_controller() {
        let controller = FlowController::new(1000.0, Duration::from_millis(100));

        for _ in 0..100 {
            let _ = controller.record_processed();
        }

        // Flow controller should be tracking records
        controller.pause();
        assert!(controller
            .record_processed()
            .expect("operation should succeed"));
    }

    #[test]
    fn test_flow_controller_set_target_throughput_actually_sets() {
        let controller = FlowController::new(10.0, Duration::from_millis(50));
        assert_eq!(controller.target_throughput(), 10.0);

        controller.set_target_throughput(5000.0);
        assert_eq!(controller.target_throughput(), 5000.0);
    }

    #[test]
    fn test_flow_controller_throttles_beyond_target() {
        // A tiny target throughput (2/sec) should throttle a tight loop of
        // `record_processed()` calls almost immediately -- if the token
        // bucket is genuinely wired in, most calls in a fast loop must
        // return `false`.
        let controller = FlowController::new(2.0, Duration::from_millis(50));

        let mut allowed = 0;
        let mut throttled = 0;
        for _ in 0..50 {
            if controller
                .record_processed()
                .expect("operation should succeed")
            {
                allowed += 1;
            } else {
                throttled += 1;
            }
        }

        assert!(
            throttled > 0,
            "expected the token bucket to throttle at least one call out of 50 at a 2/sec target"
        );
        assert!(
            allowed > 0,
            "expected at least the initial burst to be allowed"
        );
    }

    #[test]
    fn test_backpressure_pop_batch() {
        let config = BackpressureConfig::default();
        let buffer = BackpressureBuffer::new(config);

        for _ in 0..100 {
            buffer
                .push(create_test_record())
                .expect("operation should succeed");
        }

        let batch = buffer.pop_batch(30).expect("operation should succeed");
        assert_eq!(batch.len(), 30);
        assert_eq!(buffer.len().expect("operation should succeed"), 70);
    }

    #[test]
    fn test_backpressure_pop_records_latency() {
        let config = BackpressureConfig::default();
        let buffer = BackpressureBuffer::new(config);

        buffer
            .push(create_test_record())
            .expect("operation should succeed");
        std::thread::sleep(Duration::from_millis(5));
        buffer.pop().expect("operation should succeed");

        let stats = buffer.stats().expect("operation should succeed");
        assert!(
            stats.avg_latency_ms > 0.0,
            "expected a measured, nonzero average latency, got {}",
            stats.avg_latency_ms
        );
    }

    #[test]
    fn test_should_sample_spans_full_range() {
        // With the previous subsec_nanos()/u32::MAX implementation, rate
        // values above ~0.2328 always sampled true. Verify a high rate can
        // still occasionally decline and a low rate can still occasionally
        // accept, across enough trials that a broken implementation would
        // almost certainly show either all-true or all-false.
        let mostly_true = (0..500).filter(|_| should_sample(0.9)).count();
        let mostly_false = (0..500).filter(|_| should_sample(0.1)).count();

        assert!(
            mostly_true > 350,
            "rate=0.9 should accept most samples, got {mostly_true}/500"
        );
        assert!(
            mostly_false < 150,
            "rate=0.1 should accept few samples, got {mostly_false}/500"
        );
    }
}
