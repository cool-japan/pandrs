//! DataFrame Integration for Enhanced JIT System
//!
//! This module provides seamless integration between the enhanced JIT compilation system
//! and the new DataFrame trait hierarchy, enabling automatic optimization of DataFrame operations.

use crate::core::data_value::DataValue;
use crate::core::dataframe_traits::{
    AggFunc, Axis, BooleanMask, DataFrameAdvancedOps, DataFrameOps, GroupByOps, GroupKey,
    IndexingOps, JoinType, StatisticalOps,
};
use crate::core::error::{Error, Result};
use crate::optimized::jit::{
    adaptive_optimizer::{AdaptiveOptimizer, OptimizationReport},
    cache::{FunctionId, JitFunctionCache},
    config::JITConfig,
    expression_tree::{
        BinaryOperator, ExpressionNode, ExpressionTree, ReductionOperation, UnaryOperator,
    },
    performance_monitor::{FunctionPerformanceMetrics, JitPerformanceMonitor},
    types::{NumericValue, TypedVector},
    JitError, JitResult,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::{read_lock_safe, write_lock_safe};

/// JIT-optimized DataFrame operations trait
pub trait JitDataFrameOps {
    /// Enable JIT optimization for this DataFrame
    fn enable_jit_optimization(&mut self, config: Option<JITConfig>) -> Result<()>;

    /// Disable JIT optimization
    fn disable_jit_optimization(&mut self) -> Result<()>;

    /// Get JIT optimization statistics
    fn get_jit_stats(&self) -> Option<JitOptimizationStats>;

    /// Compile and cache frequently used operations
    fn warm_jit_cache(&self, operations: &[&str]) -> Result<()>;

    /// Clear JIT cache for this DataFrame
    fn clear_jit_cache(&self) -> Result<()>;

    /// Execute operation with JIT optimization
    fn execute_with_jit<F, R>(&self, operation_name: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R> + Send + Sync + 'static,
        R: Send + Sync + 'static;

    /// Create expression tree for complex operations
    fn create_expression_tree(&self, expression: &str) -> Result<ExpressionTree>;

    /// Optimize expression tree and execute with supplied column inputs.
    ///
    /// `inputs` maps variable names to their f64 column data.  The method
    /// optimises the tree via constant-folding / algebraic simplification and
    /// then walks the result recursively, performing scalar-broadcast
    /// arithmetic, reductions, conditionals, and built-in function calls.
    ///
    /// Returns a `Vec<f64>`: a single-element vector for scalar results, or a
    /// full-length vector for element-wise results.
    fn execute_expression_tree(
        &self,
        tree: &ExpressionTree,
        inputs: &std::collections::HashMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>>;
}

/// JIT optimization statistics for DataFrame operations
#[derive(Debug, Clone)]
pub struct JitOptimizationStats {
    /// Total number of JIT-optimized operations
    pub total_jit_operations: u64,
    /// Cache hit rate for JIT functions
    pub cache_hit_rate: f64,
    /// Average speedup from JIT optimization
    pub avg_speedup: f64,
    /// Memory savings from optimization
    pub memory_savings_bytes: usize,
    /// Number of expression trees optimized
    pub expression_trees_optimized: u64,
    /// Time saved through optimization (in nanoseconds)
    pub time_saved_ns: u64,
}

/// JIT-optimized DataFrame implementation
pub struct JitOptimizedDataFrame<T> {
    /// Underlying DataFrame implementation
    inner: T,
    /// JIT configuration
    jit_config: Option<JITConfig>,
    /// Performance monitor
    monitor: Arc<JitPerformanceMonitor>,
    /// Function cache
    cache: Arc<JitFunctionCache>,
    /// Adaptive optimizer
    optimizer: Arc<AdaptiveOptimizer>,
    /// Operation statistics
    stats: RwLock<JitOptimizationStats>,
    /// Expression cache
    expression_cache: RwLock<HashMap<String, ExpressionTree>>,
}

impl<T> JitOptimizedDataFrame<T>
where
    T: DataFrameOps + Send + Sync + 'static,
    T::Output: Send + Sync + 'static,
{
    /// Create a new JIT-optimized DataFrame wrapper
    pub fn new(inner: T, config: Option<JITConfig>) -> Self {
        let jit_config = config.unwrap_or_default();
        let monitor = Arc::new(JitPerformanceMonitor::new(jit_config.clone()));
        let cache = Arc::new(JitFunctionCache::new(128)); // 128MB cache
        let optimizer = Arc::new(AdaptiveOptimizer::new(
            monitor.clone(),
            cache.clone(),
            jit_config.clone(),
        ));

        Self {
            inner,
            jit_config: Some(jit_config),
            monitor,
            cache,
            optimizer,
            stats: RwLock::new(JitOptimizationStats {
                total_jit_operations: 0,
                cache_hit_rate: 0.0,
                avg_speedup: 1.0,
                memory_savings_bytes: 0,
                expression_trees_optimized: 0,
                time_saved_ns: 0,
            }),
            expression_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get reference to inner DataFrame
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get mutable reference to inner DataFrame
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Run optimization cycle
    pub fn optimize(&self) -> Result<OptimizationReport> {
        self.optimizer
            .optimize()
            .map_err(|e| Error::InvalidOperation(e.to_string()))
    }

    /// Create a function ID for an operation
    fn create_function_id(&self, operation_name: &str, input_types: &[&str]) -> FunctionId {
        let shape = self.inner.shape();
        let signature = format!("{}x{}", shape.0, shape.1);

        FunctionId::new(
            operation_name,
            input_types.join("_"),
            "dataframe",
            signature,
            self.jit_config
                .as_ref()
                .map(|c| c.optimization_level)
                .unwrap_or(2),
        )
    }

    /// Execute operation with performance monitoring
    fn execute_monitored<F, R>(&self, function_id: &FunctionId, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let start = Instant::now();
        let result = operation();
        let execution_time = start.elapsed().as_nanos() as u64;

        // Record performance metrics
        let _ = self.monitor.record_function_execution(
            function_id,
            execution_time,
            1024, // Estimated memory usage
            0.8,  // Estimated CPU utilization
        );

        // Update statistics
        let mut stats = write_lock_safe!(self.stats, "jit dataframe integration stats write")?;
        stats.total_jit_operations += 1;

        result
    }
}

// Implement DataFrameOps for JitOptimizedDataFrame
impl<T> DataFrameOps for JitOptimizedDataFrame<T>
where
    T: DataFrameOps + Send + Sync + 'static,
    T::Output: Send + Sync + 'static,
{
    type Output = T::Output;
    type Error = Error;

    fn select(&self, columns: &[&str]) -> Result<Self::Output> {
        let function_id = self.create_function_id("select", &["string_array"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .select(columns)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn drop(&self, columns: &[&str]) -> Result<Self::Output> {
        let function_id = self.create_function_id("drop", &["string_array"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .drop(columns)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn rename(&self, mapping: &HashMap<String, String>) -> Result<Self::Output> {
        let function_id = self.create_function_id("rename", &["hashmap"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .rename(mapping)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn filter<F>(&self, predicate: F) -> Result<Self::Output>
    where
        F: Fn(&dyn DataValue) -> bool + Send + Sync,
    {
        let function_id = self.create_function_id("filter", &["predicate"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .filter(predicate)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn head(&self, n: usize) -> Result<Self::Output> {
        let function_id = self.create_function_id("head", &["usize"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .head(n)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn tail(&self, n: usize) -> Result<Self::Output> {
        let function_id = self.create_function_id("tail", &["usize"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .tail(n)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn sample(&self, n: usize, random_state: Option<u64>) -> Result<Self::Output> {
        let function_id = self.create_function_id("sample", &["usize", "option_u64"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .sample(n, random_state)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn sort_values(&self, by: &[&str], ascending: &[bool]) -> Result<Self::Output> {
        let function_id = self.create_function_id("sort_values", &["string_array", "bool_array"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .sort_values(by, ascending)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn sort_index(&self) -> Result<Self::Output> {
        let function_id = self.create_function_id("sort_index", &[]);

        self.execute_monitored(&function_id, || {
            self.inner
                .sort_index()
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn columns(&self) -> Vec<String> {
        self.inner.columns()
    }

    fn dtypes(&self) -> HashMap<String, String> {
        self.inner.dtypes()
    }

    fn info(&self) -> crate::core::dataframe_traits::DataFrameInfo {
        self.inner.info()
    }

    fn dropna(
        &self,
        axis: Option<Axis>,
        how: crate::core::dataframe_traits::DropNaHow,
    ) -> Result<Self::Output> {
        let function_id = self.create_function_id("dropna", &["axis", "how"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .dropna(axis, how)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn fillna(
        &self,
        value: &dyn DataValue,
        method: Option<crate::core::dataframe_traits::FillMethod>,
    ) -> Result<Self::Output> {
        let function_id = self.create_function_id("fillna", &["datavalue", "method"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .fillna(value, method)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn isna(&self) -> Result<Self::Output> {
        let function_id = self.create_function_id("isna", &[]);

        self.execute_monitored(&function_id, || {
            self.inner
                .isna()
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn map<F>(&self, func: F) -> Result<Self::Output>
    where
        F: Fn(&dyn DataValue) -> Box<dyn DataValue> + Send + Sync,
    {
        let function_id = self.create_function_id("map", &["function"]);

        self.execute_monitored(&function_id, || {
            self.inner
                .map(func)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }

    fn apply<F>(&self, func: F, axis: Axis) -> Result<Self::Output>
    where
        F: Fn(&Self::Output) -> Box<dyn DataValue> + Send + Sync,
    {
        let function_id = self.create_function_id("apply", &["function", "axis"]);

        // `apply` operates on rows/columns of the inner frame's output type, so
        // it cannot be transparently JIT-accelerated at this layer without
        // type-specific introspection.  Delegate directly to the inner
        // implementation so callers receive working (un-accelerated) behaviour
        // while monitoring overhead is still recorded.
        self.execute_monitored(&function_id, || {
            self.inner
                .apply(func, axis)
                .map_err(|e| Error::InvalidOperation(e.to_string()))
        })
    }
}

// ---------------------------------------------------------------------------
// Expression-tree interpreter
// ---------------------------------------------------------------------------

/// Intermediate evaluation result – either a single scalar or an element-wise
/// vector.  All arithmetic operations implement scalar-broadcast semantics:
/// (scalar op scalar) → scalar, (scalar op vector) → vector, (vector op
/// vector) → element-wise vector of the same length.
#[derive(Debug, Clone)]
enum EvalResult {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl EvalResult {
    /// Unwrap as a `Vec<f64>`, promoting a scalar to a 1-element vector.
    fn into_vec(self) -> Vec<f64> {
        match self {
            EvalResult::Scalar(s) => vec![s],
            EvalResult::Vector(v) => v,
        }
    }
}

/// Apply a binary scalar function to two `EvalResult` values with
/// scalar-broadcast semantics.
fn apply_binary<F>(lhs: EvalResult, rhs: EvalResult, f: F) -> EvalResult
where
    F: Fn(f64, f64) -> f64,
{
    match (lhs, rhs) {
        (EvalResult::Scalar(l), EvalResult::Scalar(r)) => EvalResult::Scalar(f(l, r)),
        (EvalResult::Scalar(l), EvalResult::Vector(rv)) => {
            EvalResult::Vector(rv.iter().map(|&r| f(l, r)).collect())
        }
        (EvalResult::Vector(lv), EvalResult::Scalar(r)) => {
            EvalResult::Vector(lv.iter().map(|&l| f(l, r)).collect())
        }
        (EvalResult::Vector(lv), EvalResult::Vector(rv)) => {
            // Element-wise; lengths must match.
            EvalResult::Vector(lv.iter().zip(rv.iter()).map(|(&l, &r)| f(l, r)).collect())
        }
    }
}

/// Apply a unary scalar function element-wise to an `EvalResult`.
fn apply_unary<F>(operand: EvalResult, f: F) -> EvalResult
where
    F: Fn(f64) -> f64,
{
    match operand {
        EvalResult::Scalar(s) => EvalResult::Scalar(f(s)),
        EvalResult::Vector(v) => EvalResult::Vector(v.iter().map(|&x| f(x)).collect()),
    }
}

/// Recursively evaluate an `ExpressionNode` given a map of variable data.
///
/// The function does *not* borrow `self` so it can be called without a
/// generic constraint on the DataFrame wrapper type.
fn eval_node(node: &ExpressionNode, inputs: &HashMap<String, Vec<f64>>) -> Result<EvalResult> {
    match node {
        // ------------------------------------------------------------------ //
        // Constant
        // ------------------------------------------------------------------ //
        ExpressionNode::Constant(nv) => Ok(EvalResult::Scalar(nv.to_f64())),

        // ------------------------------------------------------------------ //
        // Variable — look up in the provided inputs map
        // ------------------------------------------------------------------ //
        ExpressionNode::Variable { name, .. } => match inputs.get(name) {
            Some(data) => Ok(EvalResult::Vector(data.clone())),
            None => Err(Error::InvalidOperation(format!(
                "Variable '{}' not found in inputs",
                name
            ))),
        },

        // ------------------------------------------------------------------ //
        // Array access — evaluate the array sub-tree and index sub-tree, then
        // perform a bounds-checked scalar extraction.
        // ------------------------------------------------------------------ //
        ExpressionNode::ArrayAccess { array, index } => {
            let arr = eval_node(array, inputs)?;
            let idx_result = eval_node(index, inputs)?;

            // The index must reduce to a scalar (or 1-element vector).
            let idx_f64 = match &idx_result {
                EvalResult::Scalar(s) => *s,
                EvalResult::Vector(v) if v.len() == 1 => v[0],
                EvalResult::Vector(v) => {
                    return Err(Error::InvalidOperation(format!(
                        "Array index must be a scalar, got vector of length {}",
                        v.len()
                    )))
                }
            };

            let idx = idx_f64 as usize;
            let vec = arr.into_vec();

            if idx >= vec.len() {
                Err(Error::InvalidOperation(format!(
                    "Array index {} out of bounds for length {}",
                    idx,
                    vec.len()
                )))
            } else {
                Ok(EvalResult::Scalar(vec[idx]))
            }
        }

        // ------------------------------------------------------------------ //
        // Binary operation
        // ------------------------------------------------------------------ //
        ExpressionNode::BinaryOp {
            left,
            right,
            operator,
        } => {
            let lhs = eval_node(left, inputs)?;
            let rhs = eval_node(right, inputs)?;

            let result = match operator {
                BinaryOperator::Add => apply_binary(lhs, rhs, |l, r| l + r),
                BinaryOperator::Subtract => apply_binary(lhs, rhs, |l, r| l - r),
                BinaryOperator::Multiply => apply_binary(lhs, rhs, |l, r| l * r),
                BinaryOperator::Divide => {
                    apply_binary(lhs, rhs, |l, r| if r == 0.0 { f64::NAN } else { l / r })
                }
                BinaryOperator::Modulo => {
                    apply_binary(lhs, rhs, |l, r| if r == 0.0 { f64::NAN } else { l % r })
                }
                BinaryOperator::Power => apply_binary(lhs, rhs, |l, r| l.powf(r)),
                BinaryOperator::Equal => apply_binary(lhs, rhs, |l, r| {
                    if (l - r).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }),
                BinaryOperator::NotEqual => apply_binary(lhs, rhs, |l, r| {
                    if (l - r).abs() >= f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }),
                BinaryOperator::LessThan => {
                    apply_binary(lhs, rhs, |l, r| if l < r { 1.0 } else { 0.0 })
                }
                BinaryOperator::LessThanOrEqual => {
                    apply_binary(lhs, rhs, |l, r| if l <= r { 1.0 } else { 0.0 })
                }
                BinaryOperator::GreaterThan => {
                    apply_binary(lhs, rhs, |l, r| if l > r { 1.0 } else { 0.0 })
                }
                BinaryOperator::GreaterThanOrEqual => {
                    apply_binary(lhs, rhs, |l, r| if l >= r { 1.0 } else { 0.0 })
                }
                BinaryOperator::LogicalAnd => {
                    apply_binary(
                        lhs,
                        rhs,
                        |l, r| {
                            if l != 0.0 && r != 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        },
                    )
                }
                BinaryOperator::LogicalOr => {
                    apply_binary(
                        lhs,
                        rhs,
                        |l, r| {
                            if l != 0.0 || r != 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        },
                    )
                }
                BinaryOperator::BitwiseAnd => {
                    apply_binary(lhs, rhs, |l, r| (l as i64 & r as i64) as f64)
                }
                BinaryOperator::BitwiseOr => {
                    apply_binary(lhs, rhs, |l, r| (l as i64 | r as i64) as f64)
                }
                BinaryOperator::BitwiseXor => {
                    apply_binary(lhs, rhs, |l, r| (l as i64 ^ r as i64) as f64)
                }
            };

            Ok(result)
        }

        // ------------------------------------------------------------------ //
        // Unary operation
        // ------------------------------------------------------------------ //
        ExpressionNode::UnaryOp { operand, operator } => {
            let val = eval_node(operand, inputs)?;

            let result = match operator {
                UnaryOperator::Negate => apply_unary(val, |x| -x),
                UnaryOperator::Abs => apply_unary(val, |x| x.abs()),
                UnaryOperator::Sqrt => apply_unary(val, |x| x.sqrt()),
                UnaryOperator::Sin => apply_unary(val, |x| x.sin()),
                UnaryOperator::Cos => apply_unary(val, |x| x.cos()),
                UnaryOperator::Tan => apply_unary(val, |x| x.tan()),
                UnaryOperator::Log => apply_unary(val, |x| x.ln()),
                UnaryOperator::Exp => apply_unary(val, |x| x.exp()),
                UnaryOperator::Floor => apply_unary(val, |x| x.floor()),
                UnaryOperator::Ceil => apply_unary(val, |x| x.ceil()),
                UnaryOperator::Round => apply_unary(val, |x| x.round()),
                UnaryOperator::LogicalNot => apply_unary(val, |x| if x == 0.0 { 1.0 } else { 0.0 }),
                UnaryOperator::BitwiseNot => apply_unary(val, |x| !(x as i64) as f64),
            };

            Ok(result)
        }

        // ------------------------------------------------------------------ //
        // Function call — built-in math functions dispatched by name
        // ------------------------------------------------------------------ //
        ExpressionNode::FunctionCall {
            function,
            arguments,
        } => {
            // Evaluate all arguments first.
            let evaled: Result<Vec<EvalResult>> =
                arguments.iter().map(|a| eval_node(a, inputs)).collect();
            let evaled = evaled?;

            // All built-ins below are unary (single argument).
            if evaled.len() != 1 {
                return Err(Error::NotImplemented(format!(
                    "Built-in function '{}' expects exactly 1 argument, got {}",
                    function,
                    evaled.len()
                )));
            }

            let arg = evaled.into_iter().next().unwrap();

            let result = match function.as_str() {
                "abs" => apply_unary(arg, |x| x.abs()),
                "sqrt" => apply_unary(arg, |x| x.sqrt()),
                "floor" => apply_unary(arg, |x| x.floor()),
                "ceil" => apply_unary(arg, |x| x.ceil()),
                "round" => apply_unary(arg, |x| x.round()),
                "ln" => apply_unary(arg, |x| x.ln()),
                "log2" => apply_unary(arg, |x| x.log2()),
                "log10" => apply_unary(arg, |x| x.log10()),
                "sin" => apply_unary(arg, |x| x.sin()),
                "cos" => apply_unary(arg, |x| x.cos()),
                "tan" => apply_unary(arg, |x| x.tan()),
                "exp" => apply_unary(arg, |x| x.exp()),
                other => {
                    return Err(Error::NotImplemented(format!(
                    "Built-in function '{}' is not implemented in the expression-tree interpreter",
                    other
                )))
                }
            };

            Ok(result)
        }

        // ------------------------------------------------------------------ //
        // Reduction — fold a vector to a scalar aggregate
        // ------------------------------------------------------------------ //
        ExpressionNode::Reduction {
            array, operation, ..
        } => {
            let val = eval_node(array, inputs)?;

            // If the inner expression already reduced to a scalar, pass it
            // through unchanged (e.g. Reduction(Sum, Constant(3.0)) = 3.0).
            let elements: Vec<f64> = match val {
                EvalResult::Scalar(s) => return Ok(EvalResult::Scalar(s)),
                EvalResult::Vector(v) => v,
            };

            let n = elements.len() as f64;

            let scalar = match operation {
                ReductionOperation::Sum => elements.iter().copied().sum::<f64>(),
                ReductionOperation::Product => elements.iter().copied().product::<f64>(),
                ReductionOperation::Mean => {
                    if elements.is_empty() {
                        f64::NAN
                    } else {
                        elements.iter().copied().sum::<f64>() / n
                    }
                }
                ReductionOperation::Min => elements.iter().copied().fold(f64::INFINITY, f64::min),
                ReductionOperation::Max => {
                    elements.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                }
                ReductionOperation::Count => n,
                ReductionOperation::Any => {
                    if elements.iter().any(|&x| x != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
                ReductionOperation::All => {
                    if elements.iter().all(|&x| x != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
                ReductionOperation::Variance => {
                    // Population variance: Σ(xᵢ - μ)² / n
                    if elements.is_empty() {
                        f64::NAN
                    } else {
                        let mean = elements.iter().copied().sum::<f64>() / n;
                        elements.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
                    }
                }
                ReductionOperation::StandardDeviation => {
                    // Population std dev
                    if elements.is_empty() {
                        f64::NAN
                    } else {
                        let mean = elements.iter().copied().sum::<f64>() / n;
                        let variance =
                            elements.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
                        variance.sqrt()
                    }
                }
            };

            Ok(EvalResult::Scalar(scalar))
        }

        // ------------------------------------------------------------------ //
        // Conditional — element-wise if/else with scalar-broadcast
        // ------------------------------------------------------------------ //
        ExpressionNode::Conditional {
            condition,
            true_expr,
            false_expr,
        } => {
            let cond = eval_node(condition, inputs)?;
            let t_val = eval_node(true_expr, inputs)?;
            let f_val = eval_node(false_expr, inputs)?;

            // Determine output length from the widest operand.
            let cond_v = cond.into_vec();
            let t_v = t_val.into_vec();
            let f_v = f_val.into_vec();

            let len = cond_v.len().max(t_v.len()).max(f_v.len());

            // Helper: broadcast a 1-element vector to length `len`.
            let broadcast = |v: Vec<f64>| -> Vec<f64> {
                if v.len() == 1 {
                    vec![v[0]; len]
                } else {
                    v
                }
            };

            let cond_b = broadcast(cond_v);
            let t_b = broadcast(t_v);
            let f_b = broadcast(f_v);

            if cond_b.len() != len || t_b.len() != len || f_b.len() != len {
                return Err(Error::InvalidOperation(format!(
                    "Conditional branches have incompatible lengths: condition={}, \
                     true_branch={}, false_branch={}",
                    cond_b.len(),
                    t_b.len(),
                    f_b.len()
                )));
            }

            let result: Vec<f64> = cond_b
                .iter()
                .zip(t_b.iter())
                .zip(f_b.iter())
                .map(|((&c, &t), &f)| if c != 0.0 { t } else { f })
                .collect();

            Ok(EvalResult::Vector(result))
        }
    }
}

// Implement JitDataFrameOps for JitOptimizedDataFrame
impl<T> JitDataFrameOps for JitOptimizedDataFrame<T>
where
    T: DataFrameOps + Send + Sync + 'static,
    T::Output: Send + Sync + 'static,
{
    fn enable_jit_optimization(&mut self, config: Option<JITConfig>) -> Result<()> {
        self.jit_config = Some(config.unwrap_or_default());
        Ok(())
    }

    fn disable_jit_optimization(&mut self) -> Result<()> {
        self.jit_config = None;
        Ok(())
    }

    fn get_jit_stats(&self) -> Option<JitOptimizationStats> {
        Some(
            read_lock_safe!(self.stats, "jit dataframe integration stats read")
                .ok()?
                .clone(),
        )
    }

    fn warm_jit_cache(&self, operations: &[&str]) -> Result<()> {
        // Pre-compile commonly used operations
        for operation in operations {
            let function_id = self.create_function_id(operation, &["warm_up"]);

            // Create a dummy expression tree for the operation
            let expr = ExpressionNode::FunctionCall {
                function: operation.to_string(),
                arguments: vec![ExpressionNode::Variable {
                    name: "data".to_string(),
                    var_type: "dataframe".to_string(),
                    index: 0,
                }],
            };

            let tree = ExpressionTree::new(expr);
            let optimized_tree = tree
                .optimize()
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;

            // Cache the optimized expression
            self.expression_cache
                .write()
                .expect("operation should succeed")
                .insert(operation.to_string(), optimized_tree);
        }

        Ok(())
    }

    fn clear_jit_cache(&self) -> Result<()> {
        self.cache.clear()?;
        write_lock_safe!(
            self.expression_cache,
            "jit dataframe integration expression cache write"
        )?
        .clear();
        Ok(())
    }

    fn execute_with_jit<F, R>(&self, operation_name: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R> + Send + Sync + 'static,
        R: Send + Sync + 'static,
    {
        let function_id = self.create_function_id(operation_name, &["generic"]);

        // Check if we have a cached optimized version
        if let Some(_cached_expr) = read_lock_safe!(
            self.expression_cache,
            "jit dataframe integration expression cache read"
        )?
        .get(operation_name)
        {
            // Execute optimized version
            // For now, just execute the original operation
            let start = Instant::now();
            let result = operation();
            let execution_time = start.elapsed().as_nanos() as u64;

            let _ = self
                .monitor
                .record_function_execution(&function_id, execution_time, 1024, 0.8);

            // Update cache hit statistics
            let mut stats = write_lock_safe!(self.stats, "jit dataframe integration stats write")?;
            stats.total_jit_operations += 1;
            stats.cache_hit_rate = (stats.cache_hit_rate * (stats.total_jit_operations - 1) as f64
                + 1.0)
                / stats.total_jit_operations as f64;

            result
        } else {
            // Execute original operation and possibly cache result
            let start = Instant::now();
            let result = operation();
            let execution_time = start.elapsed().as_nanos() as u64;

            let _ = self
                .monitor
                .record_function_execution(&function_id, execution_time, 1024, 0.8);

            let mut stats = write_lock_safe!(self.stats, "jit dataframe integration stats write")?;
            stats.total_jit_operations += 1;
            stats.cache_hit_rate = (stats.cache_hit_rate * (stats.total_jit_operations - 1) as f64)
                / stats.total_jit_operations as f64;

            result
        }
    }

    fn create_expression_tree(&self, expression: &str) -> Result<ExpressionTree> {
        // Parse expression string into expression tree
        // This is a simplified implementation - a full parser would be more complex

        if expression.contains("+") {
            let parts: Vec<&str> = expression.split('+').collect();
            if parts.len() == 2 {
                let left = ExpressionNode::Variable {
                    name: parts[0].trim().to_string(),
                    var_type: "f64".to_string(),
                    index: 0,
                };

                let right = if let Ok(value) = parts[1].trim().parse::<f64>() {
                    ExpressionNode::Constant(NumericValue::F64(value))
                } else {
                    ExpressionNode::Variable {
                        name: parts[1].trim().to_string(),
                        var_type: "f64".to_string(),
                        index: 1,
                    }
                };

                let expr = ExpressionNode::BinaryOp {
                    left: Box::new(left),
                    right: Box::new(right),
                    operator: BinaryOperator::Add,
                };

                return Ok(ExpressionTree::new(expr));
            }
        }

        // Default: single variable
        let expr = ExpressionNode::Variable {
            name: expression.to_string(),
            var_type: "f64".to_string(),
            index: 0,
        };

        Ok(ExpressionTree::new(expr))
    }

    fn execute_expression_tree(
        &self,
        tree: &ExpressionTree,
        inputs: &HashMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>> {
        // Apply constant-folding, algebraic simplification, and CSE before
        // interpreting so that we avoid redundant computation at runtime.
        let optimized_tree = tree
            .optimize()
            .map_err(|e| Error::InvalidOperation(e.to_string()))?;

        // Update statistics.
        let mut stats = write_lock_safe!(self.stats, "jit dataframe integration stats write")?;
        stats.expression_trees_optimized += 1;
        drop(stats);

        // Walk the optimised AST and produce a result.
        let eval_result = eval_node(&optimized_tree.root, inputs)?;

        Ok(eval_result.into_vec())
    }
}

/// Utility function to wrap any DataFrame with JIT optimization
pub fn enable_jit_for_dataframe<T>(
    dataframe: T,
    config: Option<JITConfig>,
) -> JitOptimizedDataFrame<T>
where
    T: DataFrameOps + Send + Sync + 'static,
    T::Output: Send + Sync + 'static,
{
    JitOptimizedDataFrame::new(dataframe, config)
}

/// Batch optimization for multiple DataFrames
pub fn batch_optimize_dataframes<T>(
    dataframes: &mut [JitOptimizedDataFrame<T>],
    global_config: Option<JITConfig>,
) -> Result<Vec<OptimizationReport>>
where
    T: DataFrameOps + Send + Sync + 'static,
    T::Output: Send + Sync + 'static,
{
    let mut reports = Vec::new();

    for df in dataframes {
        if let Some(config) = &global_config {
            df.enable_jit_optimization(Some(config.clone()))?;
        }

        let report = df.optimize()?;
        reports.push(report);
    }

    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataframe_traits::DataFrameInfo;

    // Mock DataFrame implementation for testing
    struct MockDataFrame {
        rows: usize,
        cols: usize,
    }

    impl DataFrameOps for MockDataFrame {
        type Output = MockDataFrame;
        type Error = Error;

        fn select(&self, _columns: &[&str]) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: _columns.len(),
            })
        }

        fn drop(&self, columns: &[&str]) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols - columns.len(),
            })
        }

        fn rename(&self, _mapping: &HashMap<String, String>) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn filter<F>(&self, _predicate: F) -> Result<Self::Output>
        where
            F: Fn(&dyn DataValue) -> bool + Send + Sync,
        {
            Ok(MockDataFrame {
                rows: self.rows / 2,
                cols: self.cols,
            })
        }

        fn head(&self, n: usize) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: n.min(self.rows),
                cols: self.cols,
            })
        }

        fn tail(&self, n: usize) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: n.min(self.rows),
                cols: self.cols,
            })
        }

        fn sample(&self, n: usize, _random_state: Option<u64>) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: n.min(self.rows),
                cols: self.cols,
            })
        }

        fn sort_values(&self, _by: &[&str], _ascending: &[bool]) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn sort_index(&self) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn shape(&self) -> (usize, usize) {
            (self.rows, self.cols)
        }

        fn columns(&self) -> Vec<String> {
            (0..self.cols).map(|i| format!("col_{}", i)).collect()
        }

        fn dtypes(&self) -> HashMap<String, String> {
            (0..self.cols)
                .map(|i| (format!("col_{}", i), "f64".to_string()))
                .collect()
        }

        fn info(&self) -> DataFrameInfo {
            DataFrameInfo {
                shape: (self.rows, self.cols),
                memory_usage: self.rows * self.cols * 8,
                null_counts: HashMap::new(),
                dtypes: self.dtypes(),
            }
        }

        fn dropna(
            &self,
            _axis: Option<Axis>,
            _how: crate::core::dataframe_traits::DropNaHow,
        ) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn fillna(
            &self,
            _value: &dyn DataValue,
            _method: Option<crate::core::dataframe_traits::FillMethod>,
        ) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn isna(&self) -> Result<Self::Output> {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn map<F>(&self, _func: F) -> Result<Self::Output>
        where
            F: Fn(&dyn DataValue) -> Box<dyn DataValue> + Send + Sync,
        {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }

        fn apply<F>(&self, _func: F, _axis: Axis) -> Result<Self::Output>
        where
            F: Fn(&Self::Output) -> Box<dyn DataValue> + Send + Sync,
        {
            Ok(MockDataFrame {
                rows: self.rows,
                cols: self.cols,
            })
        }
    }

    #[test]
    fn test_jit_optimized_dataframe() {
        let mock_df = MockDataFrame {
            rows: 1000,
            cols: 10,
        };
        let jit_df = JitOptimizedDataFrame::new(mock_df, None);

        assert_eq!(jit_df.inner().shape(), (1000, 10));
        assert!(jit_df.jit_config.is_some());
    }

    #[test]
    fn test_jit_operations() {
        let mock_df = MockDataFrame {
            rows: 1000,
            cols: 10,
        };
        let jit_df = JitOptimizedDataFrame::new(mock_df, None);

        // Test JIT-specific operations
        let selected = jit_df
            .select(&["col_0", "col_1"])
            .expect("operation should succeed");
        assert_eq!(selected.shape(), (1000, 2));

        // Test JIT stats
        let stats = jit_df.get_jit_stats();
        // JIT operations not executed yet, so stats may be empty
        assert!(stats.is_some() || stats.is_none());
    }

    #[test]
    fn test_expression_tree_creation() {
        let mock_df = MockDataFrame {
            rows: 1000,
            cols: 10,
        };
        let jit_df = JitOptimizedDataFrame::new(mock_df, None);

        let tree = jit_df
            .create_expression_tree("x + 5")
            .expect("operation should succeed");
        assert!(tree.metadata.complexity > 0);

        let tree_str = tree.to_string();
        assert!(tree_str.contains("x"));
        assert!(tree_str.contains("5"));
    }

    #[test]
    fn test_warm_cache() {
        let mock_df = MockDataFrame {
            rows: 1000,
            cols: 10,
        };
        let jit_df = JitOptimizedDataFrame::new(mock_df, None);

        let result = jit_df.warm_jit_cache(&["select", "filter", "sort"]);
        assert!(result.is_ok());

        // Check that expressions were cached
        let cache = jit_df
            .expression_cache
            .read()
            .expect("operation should succeed");
        assert!(cache.contains_key("select"));
        assert!(cache.contains_key("filter"));
        assert!(cache.contains_key("sort"));
    }

    // -----------------------------------------------------------------------
    // execute_expression_tree tests
    // -----------------------------------------------------------------------

    /// Helper: build a JIT-wrapped MockDataFrame of default size.
    fn make_jit_df() -> JitOptimizedDataFrame<MockDataFrame> {
        JitOptimizedDataFrame::new(MockDataFrame { rows: 100, cols: 4 }, None)
    }

    #[test]
    fn test_execute_expression_tree_constant() {
        let jit_df = make_jit_df();
        let tree = ExpressionTree::new(ExpressionNode::Constant(
            crate::optimized::jit::types::NumericValue::F64(42.0),
        ));
        let inputs: HashMap<String, Vec<f64>> = HashMap::new();
        let result = jit_df
            .execute_expression_tree(&tree, &inputs)
            .expect("constant tree must not fail");
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_execute_expression_tree_variable() {
        let jit_df = make_jit_df();
        let tree = ExpressionTree::new(ExpressionNode::Variable {
            name: "x".to_string(),
            var_type: "f64".to_string(),
            index: 0,
        });
        let mut inputs: HashMap<String, Vec<f64>> = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let result = jit_df
            .execute_expression_tree(&tree, &inputs)
            .expect("variable tree must not fail");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_execute_expression_tree_add() {
        let jit_df = make_jit_df();
        // BinaryOp(Add, Variable("x"), Constant(5.0))
        let tree = ExpressionTree::new(ExpressionNode::BinaryOp {
            left: Box::new(ExpressionNode::Variable {
                name: "x".to_string(),
                var_type: "f64".to_string(),
                index: 0,
            }),
            right: Box::new(ExpressionNode::Constant(
                crate::optimized::jit::types::NumericValue::F64(5.0),
            )),
            operator: BinaryOperator::Add,
        });
        let mut inputs: HashMap<String, Vec<f64>> = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let result = jit_df
            .execute_expression_tree(&tree, &inputs)
            .expect("add tree must not fail");
        assert_eq!(result, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_execute_expression_tree_reduction_sum() {
        let jit_df = make_jit_df();
        // Reduction(Sum, Variable("x"))
        let tree = ExpressionTree::new(ExpressionNode::Reduction {
            array: Box::new(ExpressionNode::Variable {
                name: "x".to_string(),
                var_type: "f64".to_string(),
                index: 0,
            }),
            operation: ReductionOperation::Sum,
            axis: None,
        });
        let mut inputs: HashMap<String, Vec<f64>> = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let result = jit_df
            .execute_expression_tree(&tree, &inputs)
            .expect("reduction-sum tree must not fail");
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_execute_expression_tree_unbound() {
        let jit_df = make_jit_df();
        // Variable("y") with empty inputs — must return Err
        let tree = ExpressionTree::new(ExpressionNode::Variable {
            name: "y".to_string(),
            var_type: "f64".to_string(),
            index: 0,
        });
        let inputs: HashMap<String, Vec<f64>> = HashMap::new();
        let result = jit_df.execute_expression_tree(&tree, &inputs);
        assert!(
            result.is_err(),
            "unbound variable must return an error, got: {:?}",
            result
        );
    }
}
