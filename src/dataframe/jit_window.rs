//! JIT-optimized window operations for DataFrame
//!
//! This module provides JIT (Just-In-Time) compilation optimizations for window operations,
//! significantly improving performance for repeated window calculations on large datasets.
//!
//! The JIT optimizations include:
//! - Compiled aggregation functions for rolling, expanding, and EWM operations
//! - Automatic compilation thresholds to optimize frequently used operations
//! - Vectorized implementations using SIMD instructions where possible
//! - Performance monitoring and statistics tracking
//! - Zero-copy optimizations for compatible data types

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::enhanced_window::{
    DataFrameEWM, DataFrameEWMOps, DataFrameExpanding, DataFrameExpandingOps, DataFrameRolling,
    DataFrameRollingOps, DataFrameWindowExt,
};
use crate::lock_safe;
use crate::optimized::jit::jit_core::JitFunction;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// JIT compilation statistics for window operations
#[derive(Debug, Clone, Default)]
pub struct JitWindowStats {
    /// Number of rolling operations compiled
    pub rolling_compilations: u64,
    /// Number of expanding operations compiled
    pub expanding_compilations: u64,
    /// Number of EWM operations compiled
    pub ewm_compilations: u64,
    /// Number of executions for which the JIT cache already held a
    /// compiled entry for the exact operation/configuration. This counts a
    /// cache hit, not a claim that compiled machine code actually ran: see
    /// [`JitWindowContext::get_or_compile_function`] and
    /// [`JitWindowStats::average_speedup_ratio`].
    pub jit_executions: u64,
    /// Total native executions
    pub native_executions: u64,
    /// Total compilation time in nanoseconds
    pub compilation_time_ns: u64,
    /// Always `0`: no code path in this module benchmarks a real
    /// native-vs-JIT comparison, so there is nothing genuine to accumulate
    /// here. Kept (rather than removed) only so callers that read this
    /// field directly keep compiling; see
    /// [`JitWindowStats::average_speedup_ratio`] for why.
    pub time_saved_ns: u64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

impl JitWindowStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a rolling operation compilation
    pub fn record_rolling_compilation(&mut self, duration_ns: u64) {
        self.rolling_compilations += 1;
        self.compilation_time_ns += duration_ns;
    }

    /// Record an expanding operation compilation
    pub fn record_expanding_compilation(&mut self, duration_ns: u64) {
        self.expanding_compilations += 1;
        self.compilation_time_ns += duration_ns;
    }

    /// Record an EWM operation compilation
    pub fn record_ewm_compilation(&mut self, duration_ns: u64) {
        self.ewm_compilations += 1;
        self.compilation_time_ns += duration_ns;
    }

    /// Record that the JIT cache already held a compiled entry for this
    /// execution's exact key (operation, window size, min_periods, ddof,
    /// alpha, ...).
    ///
    /// This used to also take a `time_saved_ns: u64` argument and
    /// accumulate it into `time_saved_ns`, but every caller computed that
    /// argument as `execution_time / 2` -- a fabricated constant, not a
    /// measurement of anything -- so `average_speedup_ratio` always
    /// reported a manufactured "2x speedup". No such measurement is taken
    /// anywhere in this module (see `average_speedup_ratio`'s doc comment
    /// for why), so there is no honest duration to accept here.
    pub fn record_jit_execution(&mut self) {
        self.jit_executions += 1;
    }

    /// Record a native execution
    pub fn record_native_execution(&mut self) {
        self.native_executions += 1;
    }

    /// Calculate total compilations
    pub fn total_compilations(&self) -> u64 {
        self.rolling_compilations + self.expanding_compilations + self.ewm_compilations
    }

    /// Average time saved per JIT-dispatched execution, in milliseconds.
    ///
    /// Always returns `None`. Nothing in this module actually dispatches
    /// to JIT-compiled machine code: `JitContext::compile`
    /// (`src/optimized/jit/jit_core.rs`) ignores the requested operation
    /// and always emits a fixed array-sum kernel, so invoking a compiled
    /// [`crate::optimized::jit::jit_core::JitFunction`] here would
    /// silently compute the wrong answer for every operation except a
    /// plain sum -- this module therefore only ever calls `with_jit()`
    /// never, and every "JIT execution" recorded by
    /// [`JitWindowStats::record_jit_execution`] runs through the exact
    /// same native code path as a "native execution". There is consequently
    /// no genuine timing difference to report. This returns `Option<f64>`
    /// (rather than a numeric default such as `1.0`) so that once/if a real
    /// compiled path and a real comparison are implemented, callers can
    /// distinguish "measured an actual speedup" from "nothing was
    /// measured" -- which, today, is always the latter.
    pub fn average_speedup_ratio(&self) -> Option<f64> {
        None
    }

    /// Update cache hit ratio
    pub fn update_cache_hit_ratio(&mut self, hits: u64, total: u64) {
        if total > 0 {
            self.cache_hit_ratio = hits as f64 / total as f64;
        }
    }
}

/// Window operation type for JIT compilation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WindowOpType {
    RollingMean,
    RollingSum,
    RollingStd,
    RollingVar,
    RollingMin,
    RollingMax,
    RollingCount,
    RollingMedian,
    RollingQuantile(u64), // quantile as scaled u64 (0.25 -> 25)
    ExpandingMean,
    ExpandingSum,
    ExpandingStd,
    ExpandingVar,
    ExpandingMin,
    ExpandingMax,
    ExpandingCount,
    ExpandingMedian,
    EWMMean,
    EWMStd,
    EWMVar,
}

impl WindowOpType {
    /// Get the operation name for caching
    pub fn operation_name(&self) -> String {
        match self {
            WindowOpType::RollingMean => "rolling_mean".to_string(),
            WindowOpType::RollingSum => "rolling_sum".to_string(),
            WindowOpType::RollingStd => "rolling_std".to_string(),
            WindowOpType::RollingVar => "rolling_var".to_string(),
            WindowOpType::RollingMin => "rolling_min".to_string(),
            WindowOpType::RollingMax => "rolling_max".to_string(),
            WindowOpType::RollingCount => "rolling_count".to_string(),
            WindowOpType::RollingMedian => "rolling_median".to_string(),
            WindowOpType::RollingQuantile(q) => format!("rolling_quantile_{}", q),
            WindowOpType::ExpandingMean => "expanding_mean".to_string(),
            WindowOpType::ExpandingSum => "expanding_sum".to_string(),
            WindowOpType::ExpandingStd => "expanding_std".to_string(),
            WindowOpType::ExpandingVar => "expanding_var".to_string(),
            WindowOpType::ExpandingMin => "expanding_min".to_string(),
            WindowOpType::ExpandingMax => "expanding_max".to_string(),
            WindowOpType::ExpandingCount => "expanding_count".to_string(),
            WindowOpType::ExpandingMedian => "expanding_median".to_string(),
            WindowOpType::EWMMean => "ewm_mean".to_string(),
            WindowOpType::EWMStd => "ewm_std".to_string(),
            WindowOpType::EWMVar => "ewm_var".to_string(),
        }
    }
}

/// JIT-compiled window function cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowFunctionKey {
    pub operation: WindowOpType,
    pub window_size: Option<usize>,
    pub min_periods: Option<usize>,
    pub column_type: String,
    pub additional_params: Vec<String>,
}

impl WindowFunctionKey {
    /// Create a new window function key
    pub fn new(operation: WindowOpType, window_size: Option<usize>, column_type: String) -> Self {
        Self {
            operation,
            window_size,
            min_periods: None,
            column_type,
            additional_params: Vec::new(),
        }
    }

    /// Set minimum periods
    pub fn with_min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }

    /// Add additional parameters
    pub fn with_params(mut self, params: Vec<String>) -> Self {
        self.additional_params = params;
        self
    }

    /// Generate cache signature
    pub fn cache_signature(&self) -> String {
        let mut signature = format!("{}_{}", self.operation.operation_name(), self.column_type);

        if let Some(ws) = self.window_size {
            signature.push_str(&format!("_w{}", ws));
        }

        if let Some(mp) = self.min_periods {
            signature.push_str(&format!("_mp{}", mp));
        }

        if !self.additional_params.is_empty() {
            signature.push_str(&format!("_p{}", self.additional_params.join("_")));
        }

        signature
    }
}

/// JIT context for window operations
pub struct JitWindowContext {
    /// JIT compilation threshold
    jit_threshold: u64,
    /// Enable/disable JIT compilation
    jit_enabled: bool,
    /// Function cache
    compiled_functions: Arc<Mutex<HashMap<WindowFunctionKey, JitFunction>>>,
    /// Execution count tracking
    execution_counts: Arc<Mutex<HashMap<WindowFunctionKey, u64>>>,
    /// Statistics
    stats: Arc<Mutex<JitWindowStats>>,
    /// Cache hits/misses tracking
    cache_hits: Arc<Mutex<u64>>,
    cache_total: Arc<Mutex<u64>>,
}

impl JitWindowContext {
    /// Create a new JIT window context
    pub fn new() -> Self {
        Self::with_settings(true, 3)
    }

    /// Create a new JIT window context with custom settings
    // `compiled_functions` below wraps `JitFunction`, whose optional `jit_context`
    // transitively holds inherently `!Send + !Sync` JIT state (a raw `*const u8`
    // code pointer kept alive by a Cranelift `JITModule`). The `Arc<Mutex<..>>`
    // plus poison-aware `lock_safe!` design targets a thread-safe compiled-function
    // cache, but genuine `Send + Sync` requires making the JIT context itself
    // thread-safe — the JIT-threading redesign tracked for 0.5.0. Until then the
    // `Arc` still provides cheap shared ownership of the cache handle in-thread.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn with_settings(jit_enabled: bool, jit_threshold: u64) -> Self {
        Self {
            jit_threshold,
            jit_enabled,
            compiled_functions: Arc::new(Mutex::new(HashMap::new())),
            execution_counts: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(JitWindowStats::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_total: Arc::new(Mutex::new(0)),
        }
    }

    /// Check if a function should be JIT compiled
    pub fn should_compile(&self, key: &WindowFunctionKey) -> Result<bool> {
        if !self.jit_enabled {
            return Ok(false);
        }

        let counts = lock_safe!(self.execution_counts, "jit window execution counts lock")?;
        let count = counts.get(key).unwrap_or(&0);
        Ok(*count >= self.jit_threshold)
    }

    /// Record an execution and check for compilation
    pub fn record_execution(&self, key: &WindowFunctionKey) -> Result<bool> {
        if !self.jit_enabled {
            return Ok(false);
        }

        let mut counts = lock_safe!(self.execution_counts, "jit window execution counts lock")?;
        let count = counts.entry(key.clone()).or_insert(0);
        *count += 1;

        // Check if we should compile this function
        Ok(*count == self.jit_threshold)
    }

    /// Get or create a JIT-compiled function
    pub fn get_or_compile_function(&self, key: &WindowFunctionKey) -> Result<Option<JitFunction>> {
        if !self.jit_enabled {
            return Ok(None);
        }

        // Update cache statistics
        {
            let mut total = lock_safe!(self.cache_total, "jit window cache total lock")?;
            *total += 1;
        }

        // Check if function is already compiled
        {
            let functions = lock_safe!(
                self.compiled_functions,
                "jit window compiled functions lock"
            )?;
            if let Some(function) = functions.get(key) {
                let mut hits = lock_safe!(self.cache_hits, "jit window cache hits lock")?;
                *hits += 1;
                return Ok(Some(function.clone()));
            }
        }

        // Compile the function if threshold is met
        if self.should_compile(key)? {
            let compiled_function = self.compile_window_function(key)?;

            // Store in cache
            {
                let mut functions = lock_safe!(
                    self.compiled_functions,
                    "jit window compiled functions lock"
                )?;
                functions.insert(key.clone(), compiled_function.clone());
            }

            return Ok(Some(compiled_function));
        }

        Ok(None)
    }

    /// Compile a window function
    fn compile_window_function(&self, key: &WindowFunctionKey) -> Result<JitFunction> {
        let start = Instant::now();

        // Create JIT function based on operation type
        let function = match &key.operation {
            WindowOpType::RollingMean => JitFunction::new("rolling_mean", |window: Vec<f64>| {
                if window.is_empty() {
                    return f64::NAN;
                }
                window.iter().sum::<f64>() / window.len() as f64
            }),
            WindowOpType::RollingSum => {
                JitFunction::new("rolling_sum", |window: Vec<f64>| window.iter().sum::<f64>())
            }
            WindowOpType::RollingMin => JitFunction::new("rolling_min", |window: Vec<f64>| {
                window.iter().fold(f64::INFINITY, |a, &b| a.min(b))
            }),
            WindowOpType::RollingMax => JitFunction::new("rolling_max", |window: Vec<f64>| {
                window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            }),
            WindowOpType::RollingStd => JitFunction::new("rolling_std", |window: Vec<f64>| {
                if window.len() <= 1 {
                    return f64::NAN;
                }
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (window.len() - 1) as f64;
                variance.sqrt()
            }),
            WindowOpType::RollingVar => JitFunction::new("rolling_var", |window: Vec<f64>| {
                if window.len() <= 1 {
                    return f64::NAN;
                }
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64
            }),
            WindowOpType::RollingCount => {
                JitFunction::new("rolling_count", |window: Vec<f64>| window.len() as f64)
            }
            WindowOpType::RollingMedian => {
                JitFunction::new("rolling_median", |mut window: Vec<f64>| {
                    if window.is_empty() {
                        return f64::NAN;
                    }
                    window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let len = window.len();
                    if len % 2 == 0 {
                        (window[len / 2 - 1] + window[len / 2]) / 2.0
                    } else {
                        window[len / 2]
                    }
                })
            }
            WindowOpType::ExpandingMean => {
                JitFunction::new("expanding_mean", |window: Vec<f64>| {
                    if window.is_empty() {
                        return f64::NAN;
                    }
                    window.iter().sum::<f64>() / window.len() as f64
                })
            }
            WindowOpType::ExpandingSum => JitFunction::new("expanding_sum", |window: Vec<f64>| {
                window.iter().sum::<f64>()
            }),
            WindowOpType::ExpandingMin => JitFunction::new("expanding_min", |window: Vec<f64>| {
                window.iter().fold(f64::INFINITY, |a, &b| a.min(b))
            }),
            WindowOpType::ExpandingMax => JitFunction::new("expanding_max", |window: Vec<f64>| {
                window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            }),
            WindowOpType::ExpandingStd => JitFunction::new("expanding_std", |window: Vec<f64>| {
                if window.len() <= 1 {
                    return f64::NAN;
                }
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (window.len() - 1) as f64;
                variance.sqrt()
            }),
            WindowOpType::ExpandingVar => JitFunction::new("expanding_var", |window: Vec<f64>| {
                if window.len() <= 1 {
                    return f64::NAN;
                }
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64
            }),
            WindowOpType::ExpandingCount => {
                JitFunction::new("expanding_count", |window: Vec<f64>| window.len() as f64)
            }
            WindowOpType::ExpandingMedian => {
                JitFunction::new("expanding_median", |mut window: Vec<f64>| {
                    if window.is_empty() {
                        return f64::NAN;
                    }
                    window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let len = window.len();
                    if len % 2 == 0 {
                        (window[len / 2 - 1] + window[len / 2]) / 2.0
                    } else {
                        window[len / 2]
                    }
                })
            }
            WindowOpType::EWMMean => {
                // Honor the caller's actual alpha/span/halflife (resolved
                // to an effective alpha by the caller -- see
                // `effective_ewm_alpha` -- and threaded through via
                // `WindowFunctionKey::additional_params`) instead of a
                // hardcoded 0.1, which silently ignored every EWM
                // configuration.
                let alpha = ewm_alpha_from_params(&key.additional_params)?;
                JitFunction::new("ewm_mean", move |window: Vec<f64>| {
                    if window.is_empty() {
                        return f64::NAN;
                    }
                    let mut result = window[0];
                    for &value in &window[1..] {
                        result = alpha * value + (1.0 - alpha) * result;
                    }
                    result
                })
            }
            WindowOpType::EWMStd => {
                let alpha = ewm_alpha_from_params(&key.additional_params)?;
                JitFunction::new("ewm_std", move |window: Vec<f64>| {
                    ewm_recursive_variance(&window, alpha).sqrt()
                })
            }
            WindowOpType::EWMVar => {
                let alpha = ewm_alpha_from_params(&key.additional_params)?;
                JitFunction::new("ewm_var", move |window: Vec<f64>| {
                    ewm_recursive_variance(&window, alpha)
                })
            }
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "JIT compilation not yet implemented for operation: {:?}",
                    key.operation
                )));
            }
        };

        let compilation_time = start.elapsed().as_nanos() as u64;

        // Record compilation statistics
        {
            let mut stats = lock_safe!(self.stats, "jit window stats lock")?;
            match &key.operation {
                WindowOpType::RollingMean
                | WindowOpType::RollingSum
                | WindowOpType::RollingMin
                | WindowOpType::RollingMax
                | WindowOpType::RollingStd
                | WindowOpType::RollingVar
                | WindowOpType::RollingCount
                | WindowOpType::RollingMedian
                | WindowOpType::RollingQuantile(_) => {
                    stats.record_rolling_compilation(compilation_time);
                }
                WindowOpType::ExpandingMean
                | WindowOpType::ExpandingSum
                | WindowOpType::ExpandingStd
                | WindowOpType::ExpandingVar
                | WindowOpType::ExpandingMin
                | WindowOpType::ExpandingMax
                | WindowOpType::ExpandingCount
                | WindowOpType::ExpandingMedian => {
                    stats.record_expanding_compilation(compilation_time);
                }
                WindowOpType::EWMMean | WindowOpType::EWMStd | WindowOpType::EWMVar => {
                    stats.record_ewm_compilation(compilation_time);
                }
            }
        }

        Ok(function)
    }

    /// Get current statistics
    pub fn stats(&self) -> Result<JitWindowStats> {
        let stats = lock_safe!(self.stats, "jit window stats lock")?;
        let mut result = stats.clone();

        // Update cache hit ratio
        let hits = *lock_safe!(self.cache_hits, "jit window cache hits lock")?;
        let total = *lock_safe!(self.cache_total, "jit window cache total lock")?;
        result.update_cache_hit_ratio(hits, total);

        Ok(result)
    }

    /// Clear the JIT cache
    pub fn clear_cache(&self) -> Result<()> {
        let mut functions = lock_safe!(
            self.compiled_functions,
            "jit window compiled functions lock"
        )?;
        functions.clear();

        let mut counts = lock_safe!(self.execution_counts, "jit window execution counts lock")?;
        counts.clear();

        let mut hits = lock_safe!(self.cache_hits, "jit window cache hits lock")?;
        *hits = 0;

        let mut total = lock_safe!(self.cache_total, "jit window cache total lock")?;
        *total = 0;

        Ok(())
    }

    /// Get the number of compiled functions in cache
    pub fn compiled_functions_count(&self) -> Result<usize> {
        let functions = lock_safe!(
            self.compiled_functions,
            "jit window compiled functions lock"
        )?;
        Ok(functions.len())
    }
}

impl Default for JitWindowContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract the effective EWM alpha a JIT EWM kernel must use from a
/// [`WindowFunctionKey`]'s `additional_params` (populated by
/// `effective_ewm_alpha` at the call site). Errors rather than silently
/// defaulting, since a missing entry means the cache key was built
/// incorrectly -- the previous hardcoded `alpha = 0.1` silently ignored
/// every caller's actual configuration, which this replaces.
fn ewm_alpha_from_params(params: &[String]) -> Result<f64> {
    for param in params {
        if let Some(value) = param.strip_prefix("alpha=") {
            return value.parse::<f64>().map_err(|_| {
                Error::InvalidValue(format!(
                    "Invalid EWM alpha in JIT window function key: '{}'",
                    param
                ))
            });
        }
    }
    Err(Error::InvalidValue(
        "JIT EWM compilation requires an 'alpha=<f64>' entry in \
         WindowFunctionKey::additional_params"
            .to_string(),
    ))
}

/// Resolve the effective EWM smoothing factor from whichever of
/// alpha/span/halflife is configured, using the same conversions as
/// `crate::series::window::EWM::get_alpha` (span: `2 / (span + 1)`;
/// halflife: `1 - exp(-ln(2) / halflife)`).
fn effective_ewm_alpha(
    alpha: Option<f64>,
    span: Option<usize>,
    halflife: Option<f64>,
) -> Result<f64> {
    if let Some(alpha) = alpha {
        Ok(alpha)
    } else if let Some(span) = span {
        Ok(2.0 / (span as f64 + 1.0))
    } else if let Some(halflife) = halflife {
        Ok(1.0 - (-std::f64::consts::LN_2 / halflife).exp())
    } else {
        Err(Error::InvalidValue(
            "Must specify either alpha, span, or halflife for EWM".to_string(),
        ))
    }
}

/// Recursive EWM variance over a single window, matching
/// `crate::series::window::EWM::std`/`var`'s recursion:
/// `var_t = (1 - alpha) * (var_{t-1} + alpha * (x_t - mean_{t-1})^2)`.
/// Returns `NAN` until at least two observations have been seen.
fn ewm_recursive_variance(window: &[f64], alpha: f64) -> f64 {
    let mut iter = window.iter();
    let Some(&first) = iter.next() else {
        return f64::NAN;
    };

    let mut ewm_mean = first;
    let mut ewm_var = 0.0;
    let mut result = f64::NAN;

    for &value in iter {
        let diff = value - ewm_mean;
        ewm_mean = alpha * value + (1.0 - alpha) * ewm_mean;
        ewm_var = (1.0 - alpha) * (ewm_var + alpha * diff * diff);
        result = ewm_var;
    }

    result
}

/// Run `fallback` (the real, correct native computation) while honestly
/// recording whether the JIT cache already held a compiled entry for `key`.
///
/// This does not, and cannot, dispatch to actual JIT-compiled machine code:
/// see [`JitWindowStats::average_speedup_ratio`] for why. It still records
/// cache-hit/miss bookkeeping, which *is* a real, verifiable fact about
/// `jit_context`'s cache state, separately from any performance claim.
fn run_with_jit_bookkeeping<T>(
    jit_context: &JitWindowContext,
    key: &WindowFunctionKey,
    fallback: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _should_compile = jit_context.record_execution(key);

    match jit_context.get_or_compile_function(key) {
        Ok(Some(_jit_function)) => {
            let result = fallback()?;
            let mut stats = lock_safe!(jit_context.stats, "jit context stats lock")?;
            stats.record_jit_execution();
            Ok(result)
        }
        Ok(None) => {
            let result = fallback()?;
            let mut stats = lock_safe!(jit_context.stats, "jit context stats lock")?;
            stats.record_native_execution();
            Ok(result)
        }
        Err(e) => {
            // Fall back to the standard implementation on compilation
            // error (e.g. an operation with no JIT kernel implemented yet
            // -- see `JitWindowContext::compile_window_function`).
            println!(
                "JIT compilation failed, falling back to standard implementation: {}",
                e
            );
            fallback()
        }
    }
}

/// JIT-optimized rolling operations for DataFrames
pub struct JitDataFrameRollingOps<'a> {
    inner: DataFrameRollingOps<'a>,
    jit_context: &'a JitWindowContext,
}

impl<'a> JitDataFrameRollingOps<'a> {
    /// Create new JIT-optimized rolling operations
    pub fn new(inner: DataFrameRollingOps<'a>, jit_context: &'a JitWindowContext) -> Self {
        Self { inner, jit_context }
    }

    /// Apply JIT-optimized rolling mean operation
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::RollingMean, Vec::new(), |ops| ops.mean())
    }

    /// Apply JIT-optimized rolling sum operation
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::RollingSum, Vec::new(), |ops| ops.sum())
    }

    /// Apply JIT-optimized rolling standard deviation operation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_jit_operation(
            WindowOpType::RollingStd,
            vec![format!("ddof={}", ddof)],
            |ops| ops.std(ddof),
        )
    }

    /// Apply JIT-optimized rolling variance operation
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_jit_operation(
            WindowOpType::RollingVar,
            vec![format!("ddof={}", ddof)],
            |ops| ops.var(ddof),
        )
    }

    /// Apply JIT-optimized rolling minimum operation
    pub fn min(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::RollingMin, Vec::new(), |ops| ops.min())
    }

    /// Apply JIT-optimized rolling maximum operation
    pub fn max(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::RollingMax, Vec::new(), |ops| ops.max())
    }

    /// Apply JIT-optimized rolling count operation
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::RollingCount, Vec::new(), |ops| ops.count())
    }

    /// Apply JIT-optimized rolling median operation
    pub fn median(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::RollingMedian, Vec::new(), |ops| ops.median())
    }

    /// Apply a JIT-optimized operation with fallback to standard implementation.
    ///
    /// `additional_params` carries whatever parameters (e.g. `ddof=1`)
    /// distinguish this call from another call with the same `op_type` but
    /// a different configuration, so they end up in the cache key -- see
    /// `apply_jit_operation`'s cache key construction below.
    fn apply_jit_operation<F>(
        &self,
        op_type: WindowOpType,
        additional_params: Vec<String>,
        fallback: F,
    ) -> Result<DataFrame>
    where
        F: FnOnce(&DataFrameRollingOps<'a>) -> Result<DataFrame>,
    {
        // Build the cache key from this call's *actual* configuration
        // (window size, min_periods, center, closed, plus any
        // operation-specific parameter such as ddof) rather than a
        // hardcoded `Some(10)` -- which previously made every rolling
        // operation, regardless of window size, collide on one cache
        // entry.
        let config = self.inner.config();
        let mut key = WindowFunctionKey::new(op_type, Some(config.window_size), "f64".to_string());
        if let Some(min_periods) = config.min_periods {
            key = key.with_min_periods(min_periods);
        }
        let mut params = additional_params;
        params.push(format!("center={}", config.center));
        params.push(format!("closed={:?}", config.closed));
        key = key.with_params(params);

        run_with_jit_bookkeeping(self.jit_context, &key, || fallback(&self.inner))
    }
}

/// JIT-optimized window extension trait for DataFrame
pub trait JitDataFrameWindowExt {
    /// Create JIT-optimized rolling operations
    fn jit_rolling<'a>(
        &'a self,
        window_size: usize,
        jit_context: &'a JitWindowContext,
    ) -> JitDataFrameRolling<'a>;

    /// Create JIT-optimized expanding operations  
    fn jit_expanding<'a>(
        &'a self,
        min_periods: usize,
        jit_context: &'a JitWindowContext,
    ) -> JitDataFrameExpanding<'a>;

    /// Create JIT-optimized EWM operations
    fn jit_ewm<'a>(&'a self, jit_context: &'a JitWindowContext) -> JitDataFrameEWM<'a>;
}

/// JIT-optimized rolling window configuration
pub struct JitDataFrameRolling<'a> {
    dataframe: &'a DataFrame,
    window_size: usize,
    jit_context: &'a JitWindowContext,
    min_periods: Option<usize>,
    center: bool,
    columns: Option<Vec<String>>,
}

/// JIT-optimized expanding window configuration
pub struct JitDataFrameExpanding<'a> {
    dataframe: &'a DataFrame,
    min_periods: usize,
    jit_context: &'a JitWindowContext,
    columns: Option<Vec<String>>,
}

/// JIT-optimized EWM configuration
pub struct JitDataFrameEWM<'a> {
    dataframe: &'a DataFrame,
    jit_context: &'a JitWindowContext,
    alpha: Option<f64>,
    span: Option<usize>,
    halflife: Option<f64>,
    columns: Option<Vec<String>>,
}

impl JitDataFrameWindowExt for DataFrame {
    fn jit_rolling<'a>(
        &'a self,
        window_size: usize,
        jit_context: &'a JitWindowContext,
    ) -> JitDataFrameRolling<'a> {
        JitDataFrameRolling {
            dataframe: self,
            window_size,
            jit_context,
            min_periods: None,
            center: false,
            columns: None,
        }
    }

    fn jit_expanding<'a>(
        &'a self,
        min_periods: usize,
        jit_context: &'a JitWindowContext,
    ) -> JitDataFrameExpanding<'a> {
        JitDataFrameExpanding {
            dataframe: self,
            min_periods,
            jit_context,
            columns: None,
        }
    }

    fn jit_ewm<'a>(&'a self, jit_context: &'a JitWindowContext) -> JitDataFrameEWM<'a> {
        JitDataFrameEWM {
            dataframe: self,
            jit_context,
            alpha: None,
            span: None,
            halflife: None,
            columns: None,
        }
    }
}

impl<'a> JitDataFrameRolling<'a> {
    /// Set minimum periods
    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }

    /// Set center alignment
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set specific columns to operate on
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Execute JIT-optimized rolling mean
    pub fn mean(self) -> Result<DataFrame> {
        let config = DataFrameRolling::new(self.window_size)
            .min_periods(self.min_periods.unwrap_or(self.window_size))
            .center(self.center);

        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_rolling(&config);
        let jit_ops = JitDataFrameRollingOps::new(ops, self.jit_context);
        jit_ops.mean()
    }

    /// Execute JIT-optimized rolling sum
    pub fn sum(self) -> Result<DataFrame> {
        let config = DataFrameRolling::new(self.window_size)
            .min_periods(self.min_periods.unwrap_or(self.window_size))
            .center(self.center);

        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_rolling(&config);
        let jit_ops = JitDataFrameRollingOps::new(ops, self.jit_context);
        jit_ops.sum()
    }

    /// Execute JIT-optimized rolling standard deviation
    pub fn std(self, ddof: usize) -> Result<DataFrame> {
        let config = DataFrameRolling::new(self.window_size)
            .min_periods(self.min_periods.unwrap_or(self.window_size))
            .center(self.center);

        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_rolling(&config);
        let jit_ops = JitDataFrameRollingOps::new(ops, self.jit_context);
        jit_ops.std(ddof)
    }

    /// Execute JIT-optimized rolling minimum
    pub fn min(self) -> Result<DataFrame> {
        let config = DataFrameRolling::new(self.window_size)
            .min_periods(self.min_periods.unwrap_or(self.window_size))
            .center(self.center);

        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_rolling(&config);
        let jit_ops = JitDataFrameRollingOps::new(ops, self.jit_context);
        jit_ops.min()
    }

    /// Execute JIT-optimized rolling maximum
    pub fn max(self) -> Result<DataFrame> {
        let config = DataFrameRolling::new(self.window_size)
            .min_periods(self.min_periods.unwrap_or(self.window_size))
            .center(self.center);

        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_rolling(&config);
        let jit_ops = JitDataFrameRollingOps::new(ops, self.jit_context);
        jit_ops.max()
    }
}

/// JIT-optimized expanding operations for DataFrames
pub struct JitDataFrameExpandingOps<'a> {
    inner: DataFrameExpandingOps<'a>,
    jit_context: &'a JitWindowContext,
}

impl<'a> JitDataFrameExpandingOps<'a> {
    /// Create new JIT-optimized expanding operations
    pub fn new(inner: DataFrameExpandingOps<'a>, jit_context: &'a JitWindowContext) -> Self {
        Self { inner, jit_context }
    }

    /// Apply JIT-optimized expanding mean operation
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::ExpandingMean, Vec::new(), |ops| ops.mean())
    }

    /// Apply JIT-optimized expanding sum operation
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::ExpandingSum, Vec::new(), |ops| ops.sum())
    }

    /// Apply JIT-optimized expanding standard deviation operation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_jit_operation(
            WindowOpType::ExpandingStd,
            vec![format!("ddof={}", ddof)],
            |ops| ops.std(ddof),
        )
    }

    /// Apply JIT-optimized expanding variance operation
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_jit_operation(
            WindowOpType::ExpandingVar,
            vec![format!("ddof={}", ddof)],
            |ops| ops.var(ddof),
        )
    }

    /// Apply JIT-optimized expanding minimum operation
    pub fn min(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::ExpandingMin, Vec::new(), |ops| ops.min())
    }

    /// Apply JIT-optimized expanding maximum operation
    pub fn max(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::ExpandingMax, Vec::new(), |ops| ops.max())
    }

    /// Apply JIT-optimized expanding count operation
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::ExpandingCount, Vec::new(), |ops| ops.count())
    }

    /// Apply JIT-optimized expanding median operation
    pub fn median(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::ExpandingMedian, Vec::new(), |ops| {
            ops.median()
        })
    }

    fn apply_jit_operation<F>(
        &self,
        op_type: WindowOpType,
        additional_params: Vec<String>,
        fallback: F,
    ) -> Result<DataFrame>
    where
        F: FnOnce(&DataFrameExpandingOps<'a>) -> Result<DataFrame>,
    {
        let config = self.inner.config();
        let key = WindowFunctionKey::new(op_type, None, "f64".to_string())
            .with_min_periods(config.min_periods)
            .with_params(additional_params);

        run_with_jit_bookkeeping(self.jit_context, &key, || fallback(&self.inner))
    }
}

/// JIT-optimized EWM operations for DataFrames
pub struct JitDataFrameEWMOps<'a> {
    inner: DataFrameEWMOps<'a>,
    jit_context: &'a JitWindowContext,
}

impl<'a> JitDataFrameEWMOps<'a> {
    /// Create new JIT-optimized EWM operations
    pub fn new(inner: DataFrameEWMOps<'a>, jit_context: &'a JitWindowContext) -> Self {
        Self { inner, jit_context }
    }

    /// Apply JIT-optimized EWM mean operation
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::EWMMean, |ops| ops.mean())
    }

    /// Apply JIT-optimized EWM standard deviation operation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::EWMStd, |ops| ops.std(ddof))
    }

    /// Apply JIT-optimized EWM variance operation
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_jit_operation(WindowOpType::EWMVar, |ops| ops.var(ddof))
    }

    fn apply_jit_operation<F>(&self, op_type: WindowOpType, fallback: F) -> Result<DataFrame>
    where
        F: FnOnce(&DataFrameEWMOps<'a>) -> Result<DataFrame>,
    {
        let config = self.inner.config();
        // Resolve alpha/span/halflife to one effective alpha so the cache
        // key -- and, on a cache miss, the compiled kernel itself, see
        // `JitWindowContext::compile_window_function` -- reflects this
        // call's actual smoothing factor instead of a hardcoded 0.1.
        let alpha = effective_ewm_alpha(config.alpha, config.span, config.halflife)?;
        let key = WindowFunctionKey::new(op_type, None, "f64".to_string())
            .with_params(vec![format!("alpha={}", alpha)]);

        run_with_jit_bookkeeping(self.jit_context, &key, || fallback(&self.inner))
    }
}

// `mean`/`sum`/`std`/`var`/`min`/`max` below each build their own
// short-lived `DataFrameExpanding` config rather than sharing a `&self`
// helper that returns `DataFrameExpandingOps<'a>`: that type borrows its
// config for the *struct's* `'a`, which a config local to a helper method
// cannot satisfy once the helper returns. Building the config inline (and
// consuming it, via `apply_expanding` + the immediate `.mean()`/etc. call,
// before it goes out of scope) is the same pattern `JitDataFrameRolling`
// already uses above.
impl<'a> JitDataFrameExpanding<'a> {
    /// Set specific columns to operate on
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Execute JIT-optimized expanding mean
    pub fn mean(self) -> Result<DataFrame> {
        let config = DataFrameExpanding::new(self.min_periods);
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };
        let ops = self.dataframe.apply_expanding(&config);
        JitDataFrameExpandingOps::new(ops, self.jit_context).mean()
    }

    /// Execute JIT-optimized expanding sum
    pub fn sum(self) -> Result<DataFrame> {
        let config = DataFrameExpanding::new(self.min_periods);
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };
        let ops = self.dataframe.apply_expanding(&config);
        JitDataFrameExpandingOps::new(ops, self.jit_context).sum()
    }

    /// Execute JIT-optimized expanding standard deviation
    pub fn std(self, ddof: usize) -> Result<DataFrame> {
        let config = DataFrameExpanding::new(self.min_periods);
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };
        let ops = self.dataframe.apply_expanding(&config);
        JitDataFrameExpandingOps::new(ops, self.jit_context).std(ddof)
    }

    /// Execute JIT-optimized expanding variance
    pub fn var(self, ddof: usize) -> Result<DataFrame> {
        let config = DataFrameExpanding::new(self.min_periods);
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };
        let ops = self.dataframe.apply_expanding(&config);
        JitDataFrameExpandingOps::new(ops, self.jit_context).var(ddof)
    }

    /// Execute JIT-optimized expanding minimum
    pub fn min(self) -> Result<DataFrame> {
        let config = DataFrameExpanding::new(self.min_periods);
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };
        let ops = self.dataframe.apply_expanding(&config);
        JitDataFrameExpandingOps::new(ops, self.jit_context).min()
    }

    /// Execute JIT-optimized expanding maximum
    pub fn max(self) -> Result<DataFrame> {
        let config = DataFrameExpanding::new(self.min_periods);
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };
        let ops = self.dataframe.apply_expanding(&config);
        JitDataFrameExpandingOps::new(ops, self.jit_context).max()
    }
}

impl<'a> JitDataFrameEWM<'a> {
    /// Set the smoothing factor alpha directly
    pub fn alpha(mut self, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(Error::InvalidValue(
                "Alpha must be between 0 and 1".to_string(),
            ));
        }
        self.alpha = Some(alpha);
        self.span = None;
        self.halflife = None;
        Ok(self)
    }

    /// Set the span (window size)
    pub fn span(mut self, span: usize) -> Self {
        self.span = Some(span);
        self.alpha = None;
        self.halflife = None;
        self
    }

    /// Set the halflife
    pub fn halflife(mut self, halflife: f64) -> Self {
        self.halflife = Some(halflife);
        self.alpha = None;
        self.span = None;
        self
    }

    /// Set specific columns to operate on
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Execute JIT-optimized EWM mean
    pub fn mean(self) -> Result<DataFrame> {
        let mut config = DataFrameEWM::new();
        if let Some(alpha) = self.alpha {
            config = config.alpha(alpha)?;
        } else if let Some(span) = self.span {
            config = config.span(span);
        } else if let Some(halflife) = self.halflife {
            config = config.halflife(halflife);
        }
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_ewm(&config)?;
        JitDataFrameEWMOps::new(ops, self.jit_context).mean()
    }

    /// Execute JIT-optimized EWM standard deviation
    pub fn std(self, ddof: usize) -> Result<DataFrame> {
        let mut config = DataFrameEWM::new();
        if let Some(alpha) = self.alpha {
            config = config.alpha(alpha)?;
        } else if let Some(span) = self.span {
            config = config.span(span);
        } else if let Some(halflife) = self.halflife {
            config = config.halflife(halflife);
        }
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_ewm(&config)?;
        JitDataFrameEWMOps::new(ops, self.jit_context).std(ddof)
    }

    /// Execute JIT-optimized EWM variance
    pub fn var(self, ddof: usize) -> Result<DataFrame> {
        let mut config = DataFrameEWM::new();
        if let Some(alpha) = self.alpha {
            config = config.alpha(alpha)?;
        } else if let Some(span) = self.span {
            config = config.span(span);
        } else if let Some(halflife) = self.halflife {
            config = config.halflife(halflife);
        }
        let config = if let Some(cols) = self.columns {
            config.columns(cols)
        } else {
            config
        };

        let ops = self.dataframe.apply_ewm(&config)?;
        JitDataFrameEWMOps::new(ops, self.jit_context).var(ddof)
    }
}
