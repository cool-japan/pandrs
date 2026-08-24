//! Resilience patterns for PandRS connectors
//!
//! This module provides retry mechanisms, circuit breakers, and fault tolerance
//! patterns for database and cloud storage connectors to handle transient failures
//! and improve system reliability.
//!
//! # Feature gating note
//!
//! This module (`config::resilience`) is compiled unconditionally by
//! `src/config/mod.rs` regardless of the `resilience` Cargo feature, so it
//! must not hard-depend on the optional `tokio` crate that the `resilience`
//! feature pulls in. The one place that genuinely needs an async timer
//! (`backoff_sleep`) is therefore internally `#[cfg(feature = "resilience")]`-gated
//! with a blocking fallback for builds where the feature (and thus `tokio`)
//! is unavailable. See `backoff_sleep` for details.

use crate::core::error::{Error, Result};
use crate::lock_safe;
use scirs2_core::random::{rng, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Configuration for retry mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay between retries (in milliseconds)
    pub base_delay_ms: u64,
    /// Maximum delay between retries (in milliseconds)
    pub max_delay_ms: u64,
    /// Backoff strategy for calculating delays
    pub backoff_strategy: BackoffStrategy,
    /// Jitter to add randomness to retry delays
    pub jitter: bool,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Error categories that should trigger retries.
    ///
    /// Each entry is matched as a **prefix of the error's `Display` text**
    /// (not the Rust variant identifier). The defaults are the literal
    /// `Display` prefixes of pandrs's own transient [`Error`] variants
    /// (`"Connection error"`, `"Timeout error"`, `"IO error"`,
    /// `"Executor error"`), so out of the box this correctly matches
    /// `crate::core::error::Error` values produced by connectors, as well as
    /// any other error type whose `Display` message begins with a
    /// recognizable category token (as this crate's own tests exercise with
    /// plain `&str`/`String` errors such as `"TestError: ..."`).
    ///
    /// For classifying pandrs's own [`Error`] type structurally — independent
    /// of message wording — use [`is_retryable`] instead.
    pub retryable_errors: Vec<String>,
}

/// Backoff strategies for retry mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Exponential backoff with multiplier
    Exponential,
    /// Linear increase in delay
    Linear,
    /// Custom backoff with specific delays
    Custom(Vec<u64>),
}

/// Configuration for circuit breaker pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Time window for counting failures (in seconds)
    pub failure_window_seconds: u64,
    /// Minimum number of calls (within `failure_window_seconds`) before the
    /// circuit is allowed to trip
    pub minimum_calls: u32,
    /// Time to wait before attempting to close circuit (in seconds)
    pub timeout_seconds: u64,
    /// Success threshold for closing circuit (percentage)
    pub success_threshold_percentage: f64,
    /// Number of test calls when half-open
    pub half_open_max_calls: u32,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    /// Circuit is closed, allowing calls through
    Closed,
    /// Circuit is open, rejecting calls
    Open,
    /// Circuit is half-open, testing if service has recovered
    HalfOpen,
}

/// Statistics for circuit breaker monitoring
///
/// These counters are lifetime totals (never reset by window pruning), which
/// is what makes them suitable for health reporting via
/// [`ResilienceManager::get_health_status`]. The circuit breaker's internal
/// open/close decisions use a separate, windowed view of calls and failures
/// (see [`CircuitBreakerConfig::failure_window_seconds`]) rather than these
/// cumulative counters.
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub rejected_calls: u64,
    pub last_failure_time: Option<Instant>,
    pub state_changed_time: Instant,
}

/// All circuit-breaker mutable state, guarded by a single [`Mutex`].
///
/// Earlier versions of this type split state across four independently
/// locked fields (`state`, `stats`, `failure_times`, `half_open_calls`).
/// Because different methods acquired those locks in different orders
/// (e.g. `can_execute` took `state` then `stats`, while `record_failure`
/// effectively required `stats` before `state`), concurrent callers could
/// deadlock via classic ABBA lock ordering — most likely exactly when the
/// circuit was flapping open under load. Collapsing everything into one
/// `Mutex<Inner>` makes that class of deadlock structurally impossible.
#[derive(Debug)]
struct Inner {
    state: CircuitState,
    stats: CircuitStats,
    /// Timestamps of failures within the current failure window, used to
    /// evaluate `failure_threshold`. Pruned on every write.
    failure_times: Vec<Instant>,
    /// Timestamps of *all* calls (success and failure) within the current
    /// failure window, used to evaluate `minimum_calls`. This is
    /// intentionally separate from `stats.total_calls`, which is a lifetime
    /// counter used for reporting, not for the trip decision.
    call_times: Vec<Instant>,
    /// Number of probe calls admitted since entering `HalfOpen`.
    half_open_admitted: u32,
    /// Number of probe calls that succeeded since entering `HalfOpen`.
    half_open_successes: u32,
}

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    inner: Mutex<Inner>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 100,
            max_delay_ms: 30_000,
            backoff_strategy: BackoffStrategy::Exponential,
            jitter: true,
            backoff_multiplier: 2.0,
            retryable_errors: vec![
                "Connection error".to_string(),
                "Timeout error".to_string(),
                "IO error".to_string(),
                "Executor error".to_string(),
            ],
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window_seconds: 60,
            minimum_calls: 10,
            timeout_seconds: 60,
            success_threshold_percentage: 50.0,
            half_open_max_calls: 3,
        }
    }
}

/// Prunes timestamps older than `window_seconds` relative to `now`.
///
/// Uses `checked_sub` because `now - Duration::from_secs(window_seconds)`
/// panics on `Instant` underflow whenever the configured window exceeds how
/// long the process has been running (e.g. a 24h window checked in the first
/// 24h of uptime) — a reachable panic for any user-configurable window. When
/// the subtraction would underflow, no timestamp can possibly be older than
/// "the start of the process", so nothing is pruned.
fn prune_window(times: &mut Vec<Instant>, now: Instant, window_seconds: u64) {
    if let Some(window_start) = now.checked_sub(Duration::from_secs(window_seconds)) {
        times.retain(|&t| t >= window_start);
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner {
                state: CircuitState::Closed,
                stats: CircuitStats {
                    total_calls: 0,
                    successful_calls: 0,
                    failed_calls: 0,
                    rejected_calls: 0,
                    last_failure_time: None,
                    state_changed_time: Instant::now(),
                },
                failure_times: Vec::new(),
                call_times: Vec::new(),
                half_open_admitted: 0,
                half_open_successes: 0,
            }),
        }
    }

    /// Check if the circuit breaker allows the call.
    ///
    /// In `HalfOpen` state this also *admits* the call (incrementing the
    /// admission counter under the same lock as the check), bounding the
    /// number of concurrent probe calls to `half_open_max_calls`. The
    /// previous implementation only incremented a counter on *success*,
    /// which let an unbounded number of concurrent callers see "not yet at
    /// the limit" simultaneously and all get admitted at once (a thundering
    /// herd against a service that just failed).
    pub fn can_execute(&self) -> Result<bool> {
        let mut inner = lock_safe!(self.inner, "circuit breaker lock")?;
        let now = Instant::now();

        match inner.state {
            CircuitState::Closed => Ok(true),
            CircuitState::Open => {
                let timeout_elapsed = now.duration_since(inner.stats.state_changed_time).as_secs()
                    >= self.config.timeout_seconds;

                if timeout_elapsed {
                    inner.state = CircuitState::HalfOpen;
                    inner.stats.state_changed_time = now;
                    // This call itself is the first admitted probe.
                    inner.half_open_admitted = 1;
                    inner.half_open_successes = 0;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            CircuitState::HalfOpen => {
                if inner.half_open_admitted < self.config.half_open_max_calls {
                    inner.half_open_admitted += 1;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) -> Result<()> {
        let now = Instant::now();
        let mut inner = lock_safe!(self.inner, "circuit breaker lock for success")?;
        inner.stats.total_calls += 1;
        inner.stats.successful_calls += 1;
        inner.call_times.push(now);
        let window = self.config.failure_window_seconds;
        prune_window(&mut inner.call_times, now, window);

        if inner.state == CircuitState::HalfOpen {
            inner.half_open_successes += 1;
            // `half_open_admitted` should always be >= successes, but guard
            // against callers (including this module's own tests) that
            // invoke `record_success` directly without a matching
            // `can_execute` admission.
            let admitted = inner.half_open_admitted.max(inner.half_open_successes);

            if inner.half_open_successes >= self.config.half_open_max_calls {
                // Genuine success-rate computation: successes divided by the
                // number of probes actually admitted, not by the configured
                // maximum (which the previous implementation used as the
                // denominator, making the ratio trivially >= 100% whenever
                // the admission count happened to reach the configured max —
                // and worse than "trivial" once concurrent over-admission
                // could push the numerator past the configured max too).
                //
                // Note: because any failure while half-open reopens the
                // circuit immediately (see `record_failure`), this ratio can
                // only ever be evaluated after an unbroken run of successes,
                // so it is always exactly 100% in practice today. It is
                // still computed honestly (bounded, from real counters)
                // rather than assumed, and remains meaningful if the
                // fail-fast-on-first-half-open-failure policy is ever
                // relaxed to tolerate a bounded number of probe failures.
                let success_rate = (inner.half_open_successes as f64 / admitted as f64) * 100.0;
                if success_rate >= self.config.success_threshold_percentage {
                    inner.state = CircuitState::Closed;
                    inner.stats.state_changed_time = now;
                    inner.failure_times.clear();
                    inner.call_times.clear();
                    inner.half_open_admitted = 0;
                    inner.half_open_successes = 0;
                }
            }
        }
        Ok(())
    }

    /// Record a failed operation
    pub fn record_failure(&self) -> Result<()> {
        let now = Instant::now();
        let mut inner = lock_safe!(self.inner, "circuit breaker lock for failure")?;
        inner.stats.total_calls += 1;
        inner.stats.failed_calls += 1;
        inner.stats.last_failure_time = Some(now);

        let window = self.config.failure_window_seconds;
        inner.call_times.push(now);
        inner.failure_times.push(now);
        prune_window(&mut inner.call_times, now, window);
        prune_window(&mut inner.failure_times, now, window);

        let failure_count = inner.failure_times.len() as u32;
        let windowed_calls = inner.call_times.len() as u32;

        match inner.state {
            CircuitState::Closed => {
                // Both conditions are evaluated over the same failure
                // window, not against `stats.total_calls` (a lifetime
                // counter that only grows, which would eventually make the
                // `minimum_calls` gate meaningless no matter how quiet the
                // service had been recently).
                if windowed_calls >= self.config.minimum_calls
                    && failure_count >= self.config.failure_threshold
                {
                    inner.state = CircuitState::Open;
                    inner.stats.state_changed_time = now;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure while probing reopens the circuit immediately.
                inner.state = CircuitState::Open;
                inner.stats.state_changed_time = now;
                inner.half_open_admitted = 0;
                inner.half_open_successes = 0;
            }
            CircuitState::Open => {
                // Already open, just update stats
            }
        }
        Ok(())
    }

    /// Record a rejected call (when circuit is open)
    pub fn record_rejection(&self) -> Result<()> {
        lock_safe!(self.inner, "circuit breaker lock for rejection")?
            .stats
            .rejected_calls += 1;
        Ok(())
    }

    /// Get current circuit breaker state
    pub fn state(&self) -> Result<CircuitState> {
        Ok(
            lock_safe!(self.inner, "circuit breaker lock for state query")?
                .state
                .clone(),
        )
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> Result<CircuitStats> {
        Ok(
            lock_safe!(self.inner, "circuit breaker lock for stats query")?
                .stats
                .clone(),
        )
    }
}

/// Structural retryability classification for pandrs's own [`Error`] type.
///
/// Matches on the concrete `Error` variant — connection drop, timeout,
/// low-level I/O hiccup, or executor failure — instead of scanning the
/// human-readable `Display` message. The previous approach compared
/// `retryable_errors` entries like `"ConnectionError"` (a Rust identifier)
/// against the rendered `Display` text (e.g. `"Connection error: refused"`,
/// a free-form sentence), which can never match: `starts_with` on those two
/// strings is always `false`, so retries never actually fired. Use this
/// function when the operation you are retrying returns
/// `crate::core::error::Error` directly and you want classification that
/// does not depend on message wording at all.
///
/// Errors not covered here (configuration, authentication, parse/type
/// errors, ...) are treated as permanent by default: retrying a malformed
/// request or a bad credential does not make it succeed.
pub fn is_retryable(error: &Error) -> bool {
    matches!(
        error,
        Error::ConnectionError(_)
            | Error::TimeoutError(_)
            | Error::IoError(_)
            | Error::Io(_)
            | Error::ExecutorError(_)
    )
}

/// Checks whether `error_display` (an error's rendered `Display` text)
/// matches one of `config.retryable_errors` as a prefix. See the
/// documentation on [`RetryConfig::retryable_errors`] for what "matches"
/// means here and why it is prefix-based.
fn is_retryable_str(config: &RetryConfig, error_display: &str) -> bool {
    config
        .retryable_errors
        .iter()
        .any(|retryable| error_display.starts_with(retryable.as_str()))
}

/// Waits for `duration` before the next retry attempt.
///
/// `src/config/resilience.rs` is compiled unconditionally (see the
/// module-level doc comment), so it cannot unconditionally name `tokio`
/// symbols — the `tokio` crate is only guaranteed present when the
/// `resilience` Cargo feature (or another feature that also pulls it in) is
/// enabled. When it is enabled, this genuinely suspends the calling task via
/// [`tokio::time::sleep`] instead of blocking a worker thread. When it is
/// not, this falls back to a blocking `std::thread::sleep`: still correct
/// (retries still happen, still spaced out), just not non-blocking — a
/// degradation that only affects builds which asked not to depend on tokio
/// in the first place.
async fn backoff_sleep(duration: Duration) {
    #[cfg(feature = "resilience")]
    {
        tokio::time::sleep(duration).await;
    }
    #[cfg(not(feature = "resilience"))]
    {
        std::thread::sleep(duration);
    }
}

/// Retry mechanism implementation
#[derive(Debug)]
pub struct RetryMechanism {
    config: RetryConfig,
}

impl RetryMechanism {
    /// Create a new retry mechanism with the given configuration
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute a function with retry logic.
    ///
    /// Retries are genuinely asynchronous: the delay between attempts is
    /// awaited via `backoff_sleep` rather than blocking the calling thread
    /// with `std::thread::sleep`, so this no longer stalls a `tokio` worker
    /// (and, on a `current_thread` runtime, the whole application) for up to
    /// `max_delay_ms` per retry.
    pub async fn execute<F, T, E>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display + std::fmt::Debug,
    {
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let error_str = format!("{}", error);
                    let is_retryable_now = is_retryable_str(&self.config, &error_str);

                    // Return immediately if error is not retryable or we've reached max attempts
                    if !is_retryable_now || attempt >= self.config.max_attempts {
                        return Err(Error::OperationFailed(format!(
                            "Operation failed after {} attempts. Last error: {}",
                            attempt, error_str
                        )));
                    }

                    // Calculate delay for next attempt
                    let delay = self.calculate_delay(attempt);
                    backoff_sleep(Duration::from_millis(delay)).await;
                }
            }
        }
    }

    /// Calculate delay for the given attempt number
    fn calculate_delay(&self, attempt: u32) -> u64 {
        let base_delay = match &self.config.backoff_strategy {
            BackoffStrategy::Fixed => self.config.base_delay_ms,
            BackoffStrategy::Exponential => {
                let exp_delay = (self.config.base_delay_ms as f64
                    * self.config.backoff_multiplier.powi((attempt - 1) as i32))
                    as u64;
                std::cmp::min(exp_delay, self.config.max_delay_ms)
            }
            BackoffStrategy::Linear => {
                let linear_delay = self.config.base_delay_ms * attempt as u64;
                std::cmp::min(linear_delay, self.config.max_delay_ms)
            }
            BackoffStrategy::Custom(delays) => {
                if attempt > 0 && (attempt as usize - 1) < delays.len() {
                    delays[attempt as usize - 1]
                } else {
                    self.config.max_delay_ms
                }
            }
        };

        // Add jitter if enabled
        if self.config.jitter {
            let jitter_amount = (base_delay as f64 * 0.1) as u64;
            let jitter = rng().random_range(0..=jitter_amount);
            base_delay + jitter
        } else {
            base_delay
        }
    }
}

/// Combined resilience manager for connectors
#[derive(Debug)]
pub struct ResilienceManager {
    circuit_breakers: Mutex<HashMap<String, std::sync::Arc<CircuitBreaker>>>,
    retry_configs: Mutex<HashMap<String, RetryConfig>>,
    default_retry_config: RetryConfig,
    default_circuit_config: CircuitBreakerConfig,
}

impl ResilienceManager {
    /// Create a new resilience manager
    pub fn new() -> Self {
        Self {
            circuit_breakers: Mutex::new(HashMap::new()),
            retry_configs: Mutex::new(HashMap::new()),
            default_retry_config: RetryConfig::default(),
            default_circuit_config: CircuitBreakerConfig::default(),
        }
    }

    /// Get or create a circuit breaker for the given service
    pub fn get_circuit_breaker(
        &self,
        service_name: &str,
    ) -> Result<std::sync::Arc<CircuitBreaker>> {
        let mut breakers = lock_safe!(
            self.circuit_breakers,
            "resilience manager circuit breakers lock"
        )?;

        if !breakers.contains_key(service_name) {
            let breaker =
                std::sync::Arc::new(CircuitBreaker::new(self.default_circuit_config.clone()));
            breakers.insert(service_name.to_string(), breaker);
        }

        // Return the shared circuit breaker instance
        Ok(breakers
            .get(service_name)
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Circuit breaker for {} should exist",
                    service_name
                ))
            })?
            .clone())
    }

    /// Get retry configuration for the given service
    pub fn get_retry_config(&self, service_name: &str) -> Result<RetryConfig> {
        let configs = lock_safe!(self.retry_configs, "resilience manager retry configs lock")?;
        Ok(configs
            .get(service_name)
            .cloned()
            .unwrap_or_else(|| self.default_retry_config.clone()))
    }

    /// Set custom retry configuration for a service
    pub fn set_retry_config(&self, service_name: &str, config: RetryConfig) -> Result<()> {
        let mut configs = lock_safe!(
            self.retry_configs,
            "resilience manager retry configs lock for set"
        )?;
        configs.insert(service_name.to_string(), config);
        Ok(())
    }

    /// Execute an operation with both retry and circuit breaker protection.
    ///
    /// Unlike a naive composition of [`CircuitBreaker`] and
    /// [`RetryMechanism`], this drives its own attempt loop so that the
    /// breaker is consulted and updated *per attempt*:
    ///
    /// - `can_execute()` is re-checked before every attempt, not just once
    ///   before the whole (possibly multi-attempt) retry sequence. Without
    ///   this, a circuit that trips mid-retry-sequence would not stop the
    ///   remaining attempts of that sequence from still hitting the failing
    ///   service.
    /// - `record_success`/`record_failure` is called once per attempt, not
    ///   once for the entire sequence. Without this, a sequence of N retries
    ///   that all fail only ever counted as a single failure toward the
    ///   breaker's failure threshold, making the breaker far slower to trip
    ///   than the configured `failure_threshold` would suggest.
    pub async fn execute_with_resilience<F, T, E>(
        &self,
        service_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> std::result::Result<T, E> + Send + Sync,
        E: std::fmt::Display + std::fmt::Debug + Send + Sync,
    {
        let circuit_breaker = self.get_circuit_breaker(service_name)?;
        let retry_config = self.get_retry_config(service_name)?;
        let retry_mechanism = RetryMechanism::new(retry_config.clone());

        let mut attempt = 0u32;

        loop {
            attempt += 1;

            if !circuit_breaker.can_execute()? {
                circuit_breaker.record_rejection()?;
                return Err(Error::ConnectionError(format!(
                    "Circuit breaker is open for service: {}",
                    service_name
                )));
            }

            match operation() {
                Ok(result) => {
                    circuit_breaker.record_success()?;
                    return Ok(result);
                }
                Err(error) => {
                    circuit_breaker.record_failure()?;

                    let error_str = format!("{}", error);
                    let is_retryable_now = is_retryable_str(&retry_config, &error_str);

                    if !is_retryable_now || attempt >= retry_config.max_attempts {
                        return Err(Error::OperationFailed(format!(
                            "Operation failed after {} attempts for service '{}'. Last error: {}",
                            attempt, service_name, error_str
                        )));
                    }

                    let delay = retry_mechanism.calculate_delay(attempt);
                    backoff_sleep(Duration::from_millis(delay)).await;
                }
            }
        }
    }

    /// Get health status for all services
    pub fn get_health_status(&self) -> Result<HashMap<String, ServiceHealth>> {
        let breakers = lock_safe!(
            self.circuit_breakers,
            "resilience manager circuit breakers lock for health"
        )?;
        let mut health_status = HashMap::new();

        for (service_name, breaker) in breakers.iter() {
            let stats = breaker.stats()?;
            let state = breaker.state()?;

            let health = ServiceHealth {
                service_name: service_name.clone(),
                state,
                total_calls: stats.total_calls,
                successful_calls: stats.successful_calls,
                failed_calls: stats.failed_calls,
                rejected_calls: stats.rejected_calls,
                success_rate: if stats.total_calls > 0 {
                    (stats.successful_calls as f64 / stats.total_calls as f64) * 100.0
                } else {
                    0.0
                },
                last_failure_time: stats.last_failure_time,
            };

            health_status.insert(service_name.clone(), health);
        }

        Ok(health_status)
    }
}

/// Service health information
#[derive(Debug, Clone)]
pub struct ServiceHealth {
    pub service_name: String,
    pub state: CircuitState,
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub rejected_calls: u64,
    pub success_rate: f64,
    pub last_failure_time: Option<Instant>,
}

impl Default for ResilienceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for resilient operations
#[allow(async_fn_in_trait)]
pub trait ResilientOperation<T> {
    /// Execute operation with resilience patterns
    async fn execute_resilient(self, manager: &ResilienceManager, service_name: &str) -> Result<T>;
}

impl<F, T, E> ResilientOperation<T> for F
where
    F: Fn() -> std::result::Result<T, E> + Send + Sync,
    E: std::fmt::Display + std::fmt::Debug + Send + Sync,
{
    async fn execute_resilient(self, manager: &ResilienceManager, service_name: &str) -> Result<T> {
        manager.execute_with_resilience(service_name, self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_circuit_breaker_basic_operations() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            minimum_calls: 5,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Initially closed
        assert_eq!(
            cb.state().expect("operation should succeed"),
            CircuitState::Closed
        );
        assert!(cb.can_execute().expect("operation should succeed"));

        // Record some successes
        for _ in 0..3 {
            cb.record_success().expect("operation should succeed");
        }
        assert_eq!(
            cb.state().expect("operation should succeed"),
            CircuitState::Closed
        );

        // Record failures - not enough to trip circuit yet
        for _ in 0..2 {
            cb.record_failure().expect("operation should succeed");
        }
        assert_eq!(
            cb.state().expect("operation should succeed"),
            CircuitState::Closed
        );

        // One more failure should trip the circuit
        cb.record_failure().expect("operation should succeed");
        assert_eq!(
            cb.state().expect("operation should succeed"),
            CircuitState::Open
        );
        assert!(!cb.can_execute().expect("operation should succeed"));
    }

    /// Regression test for the retry mechanism: the previous version of this
    /// test never invoked `RetryMechanism` at all (it just asserted a
    /// literal `&str` against itself), so it could not have caught the bug
    /// where `is_retryable` compared config strings against the wrong
    /// representation of the error and therefore never actually retried.
    /// This version drives a real counting closure through `execute` and
    /// checks the invocation count directly: 2 simulated transient failures
    /// followed by a success must yield exactly 3 invocations.
    #[tokio::test]
    async fn test_retry_mechanism() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 1,
            backoff_strategy: BackoffStrategy::Fixed,
            jitter: false,
            retryable_errors: vec!["transient".to_string()],
            ..Default::default()
        };
        let retry = RetryMechanism::new(config);

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result = retry
            .execute(move || {
                let n = attempts_clone.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(format!("transient failure #{n}"))
                } else {
                    Ok(42)
                }
            })
            .await
            .expect("operation should eventually succeed after transient failures");

        assert_eq!(result, 42);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "expected exactly 2 failed attempts followed by 1 successful attempt"
        );
    }

    #[tokio::test]
    async fn test_retry_mechanism_stops_on_non_retryable_error() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 1,
            jitter: false,
            retryable_errors: vec!["transient".to_string()],
            ..Default::default()
        };
        let retry = RetryMechanism::new(config);

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result: Result<()> = retry
            .execute(move || {
                attempts_clone.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>("permanent failure".to_string())
            })
            .await;

        assert!(result.is_err());
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "a non-retryable error must not be retried"
        );
    }

    #[test]
    fn test_is_retryable_structural_classification() {
        assert!(is_retryable(&Error::ConnectionError("refused".into())));
        assert!(is_retryable(&Error::TimeoutError("deadline".into())));
        assert!(is_retryable(&Error::IoError("disk busy".into())));
        assert!(is_retryable(&Error::ExecutorError("worker crashed".into())));
        assert!(!is_retryable(&Error::InvalidInput("bad request".into())));
        assert!(!is_retryable(&Error::ColumnNotFound("missing".into())));
    }

    #[test]
    fn test_resilience_manager() {
        let manager = ResilienceManager::new();

        // Test getting circuit breaker
        let _cb1 = manager.get_circuit_breaker("test_service");
        let _cb2 = manager.get_circuit_breaker("test_service");

        // Test retry config
        let config = RetryConfig {
            max_attempts: 5,
            ..Default::default()
        };
        manager
            .set_retry_config("test_service", config.clone())
            .expect("operation should succeed");

        let retrieved_config = manager
            .get_retry_config("test_service")
            .expect("operation should succeed");
        assert_eq!(retrieved_config.max_attempts, 5);
    }
}
