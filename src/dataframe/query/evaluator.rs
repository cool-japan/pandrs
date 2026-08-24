//! Expression evaluation for DataFrame queries
//!
//! Two execution tiers are available:
//!
//! * a **vectorized** tier (the default) that evaluates a whole column per AST
//!   node - see [`super::vectorized`];
//! * a **row-by-row interpreter** that walks the expression tree once per row.
//!
//! Both tiers share the element-level semantics defined in [`super::ops`], so
//! they always agree on the result. Neither tier generates machine code: the
//! query engine has no JIT back end (`jit_core`'s `JitFunction` only wraps a
//! Rust closure unless a Cranelift context is attached, and the query engine
//! attaches none). The `jit_*` names below are kept for API compatibility and
//! documented for what they actually measure.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::ast::{BinaryOp, Expr, LiteralValue, UnaryOp};
use super::ops;
use super::vectorized::{load_column_values, ValueVec, VectorizedEvaluator};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::lock_safe;

/// Statistics for query expression preparation and execution.
///
/// # What the counters mean
///
/// The query engine never emits machine code, so `jit_executions` stays at zero
/// and every execution is counted as native. `vectorized_executions` and
/// `row_by_row_executions` split those native executions between the two real
/// tiers, and [`JitQueryStats::vectorized_speedup_ratio`] compares them.
///
/// "Compilation" refers to preparing an expression for execution (constant
/// folding and algebraic simplification), not to code generation.
#[derive(Debug, Clone, Default)]
pub struct JitQueryStats {
    /// Number of expression preparations (constant folding / simplification)
    pub compilations: u64,
    /// Number of executions that ran as machine code. The query engine has no
    /// code generator, so this is always zero.
    pub jit_executions: u64,
    /// Number of natively executed queries (vectorized plus row-by-row)
    pub native_executions: u64,
    /// Total expression preparation time in nanoseconds
    pub compilation_time_ns: u64,
    /// Total machine-code execution time in nanoseconds (always zero)
    pub jit_execution_time_ns: u64,
    /// Total native execution time in nanoseconds
    pub native_execution_time_ns: u64,
    /// Number of executions that used the column-at-a-time engine
    pub vectorized_executions: u64,
    /// Total column-at-a-time execution time in nanoseconds
    pub vectorized_execution_time_ns: u64,
    /// Number of executions that used the row-by-row interpreter
    pub row_by_row_executions: u64,
    /// Total row-by-row execution time in nanoseconds
    pub row_by_row_execution_time_ns: u64,
}

impl JitQueryStats {
    /// Record the preparation (constant folding) of one expression.
    pub fn record_compilation(&mut self, duration_ns: u64) {
        self.compilations += 1;
        self.compilation_time_ns += duration_ns;
    }

    /// Record an execution that ran as generated machine code.
    ///
    /// The query engine never calls this; it exists so that a future code
    /// generator can report itself honestly.
    pub fn record_jit_execution(&mut self, duration_ns: u64) {
        self.jit_executions += 1;
        self.jit_execution_time_ns += duration_ns;
    }

    /// Record a natively executed query.
    pub fn record_native_execution(&mut self, duration_ns: u64) {
        self.native_executions += 1;
        self.native_execution_time_ns += duration_ns;
    }

    /// Record a column-at-a-time execution (also counted as native).
    pub fn record_vectorized_execution(&mut self, duration_ns: u64) {
        self.vectorized_executions += 1;
        self.vectorized_execution_time_ns += duration_ns;
        self.record_native_execution(duration_ns);
    }

    /// Record a row-by-row interpreter execution (also counted as native).
    pub fn record_row_by_row_execution(&mut self, duration_ns: u64) {
        self.row_by_row_executions += 1;
        self.row_by_row_execution_time_ns += duration_ns;
        self.record_native_execution(duration_ns);
    }

    /// Average expression preparation time in nanoseconds.
    pub fn average_compilation_time_ns(&self) -> f64 {
        if self.compilations > 0 {
            self.compilation_time_ns as f64 / self.compilations as f64
        } else {
            0.0
        }
    }

    /// Ratio of average row-by-row time to average vectorized time.
    ///
    /// Returns `1.0` until both tiers have executed at least once.
    pub fn vectorized_speedup_ratio(&self) -> f64 {
        if self.vectorized_executions > 0 && self.row_by_row_executions > 0 {
            let avg_row =
                self.row_by_row_execution_time_ns as f64 / self.row_by_row_executions as f64;
            let avg_vec =
                self.vectorized_execution_time_ns as f64 / self.vectorized_executions as f64;
            if avg_vec > 0.0 {
                return avg_row / avg_vec;
            }
        }
        1.0
    }

    /// Ratio of average native time to average machine-code time.
    ///
    /// Always `1.0` for query expressions, because no machine code is generated.
    pub fn jit_speedup_ratio(&self) -> f64 {
        if self.jit_executions > 0 && self.native_executions > 0 {
            let avg_native = self.native_execution_time_ns as f64 / self.native_executions as f64;
            let avg_jit = self.jit_execution_time_ns as f64 / self.jit_executions as f64;
            if avg_jit > 0.0 {
                return avg_native / avg_jit;
            }
        }
        1.0
    }
}

/// A prepared expression, cached per query context.
#[derive(Clone)]
struct CompiledExpression {
    /// Expression signature used for cache lookup
    #[allow(dead_code)] // retained for cache diagnostics
    signature: String,
    /// Constant-folded form reused by later executions of the same expression
    prepared: Option<Expr>,
    /// Number of times this expression has been executed
    execution_count: u64,
    /// Last execution time
    #[allow(dead_code)] // retained for cache diagnostics / eviction policies
    last_execution: std::time::SystemTime,
}

/// Query execution context: variables, functions, tier selection and statistics.
pub struct QueryContext {
    /// Variable bindings for substitution.
    ///
    /// A variable is used when an identifier in the query does not name a
    /// column, and always when the identifier is written as `@name`.
    pub variables: HashMap<String, LiteralValue>,
    /// Available functions
    pub functions: HashMap<String, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    /// Prepared-expression cache, shared by clones of this context
    compiled_expressions: Arc<Mutex<HashMap<String, CompiledExpression>>>,
    /// Execution statistics
    jit_stats: Arc<Mutex<JitQueryStats>>,
    /// Number of executions after which an expression's prepared form is cached
    jit_threshold: u64,
    /// Select the vectorized tier (`true`) or the row-by-row interpreter
    /// (`false`). Both produce identical results.
    jit_enabled: bool,
}

impl std::fmt::Debug for QueryContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryContext")
            .field("variables", &self.variables)
            .field("functions", &format!("{} functions", self.functions.len()))
            .finish()
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        let mut context = Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            compiled_expressions: Arc::new(Mutex::new(HashMap::new())),
            jit_stats: Arc::new(Mutex::new(JitQueryStats::default())),
            jit_threshold: 5, // Cache the prepared form after 5 executions
            jit_enabled: true,
        };

        // Add built-in mathematical functions
        context.add_builtin_functions();
        context
    }
}

impl QueryContext {
    /// Create a new query context
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new query context with execution-tier settings.
    ///
    /// `jit_enabled` selects the vectorized tier; `jit_threshold` is the number
    /// of executions after which an expression's prepared (constant-folded)
    /// form is cached for reuse. Filtering results are identical either way.
    pub fn with_jit_settings(jit_enabled: bool, jit_threshold: u64) -> Self {
        let mut context = Self::default();
        context.jit_enabled = jit_enabled;
        context.jit_threshold = jit_threshold;
        context
    }

    /// Add a variable binding
    pub fn set_variable(&mut self, name: String, value: LiteralValue) {
        self.variables.insert(name, value);
    }

    /// Add a custom function
    pub fn add_function<F>(&mut self, name: String, func: F)
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.functions.insert(name, Box::new(func));
    }

    /// Get query execution statistics
    pub fn jit_stats(&self) -> Result<JitQueryStats> {
        Ok(lock_safe!(self.jit_stats, "query evaluator jit stats lock")?.clone())
    }

    /// Select the vectorized tier (`true`) or the row-by-row interpreter.
    pub fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = enabled;
    }

    /// Set the number of executions after which a prepared expression is cached
    pub fn set_jit_threshold(&mut self, threshold: u64) {
        self.jit_threshold = threshold;
    }

    /// Clear the prepared-expression cache
    pub fn clear_jit_cache(&mut self) -> Result<()> {
        let mut cache = lock_safe!(
            self.compiled_expressions,
            "query evaluator compiled expressions lock"
        )?;
        cache.clear();
        Ok(())
    }

    /// Number of distinct expressions tracked in the prepared-expression cache
    pub fn compiled_expressions_count(&self) -> Result<usize> {
        Ok(lock_safe!(
            self.compiled_expressions,
            "query evaluator compiled expressions lock"
        )?
        .len())
    }

    /// Add built-in mathematical functions
    fn add_builtin_functions(&mut self) {
        // Basic math functions
        self.add_function("abs".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].abs()
            }
        });

        self.add_function("sqrt".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].sqrt()
            }
        });

        self.add_function("log".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].ln()
            }
        });

        self.add_function("log10".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].log10()
            }
        });

        self.add_function("exp".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].exp()
            }
        });

        // Trigonometric functions
        self.add_function("sin".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].sin()
            }
        });

        self.add_function("cos".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].cos()
            }
        });

        self.add_function("tan".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args[0].tan()
            }
        });

        // Statistical functions
        self.add_function("min".to_string(), |args| {
            args.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        });

        self.add_function("max".to_string(), |args| {
            args.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        });

        self.add_function("sum".to_string(), |args| args.iter().sum());

        self.add_function("mean".to_string(), |args| {
            if args.is_empty() {
                0.0
            } else {
                args.iter().sum::<f64>() / args.len() as f64
            }
        });
    }
}

/// Row-by-row expression evaluator.
///
/// This is the reference implementation of the query semantics: it walks the
/// expression tree once per row. It is used directly when the vectorized tier is
/// disabled, and it defines the behaviour the vectorized tier is tested against.
pub struct Evaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    /// Cache for column data to avoid repeated loading and classification
    column_cache: std::cell::RefCell<HashMap<String, Rc<ValueVec>>>,
    /// Optimization flags
    enable_short_circuit: bool,
    enable_constant_folding: bool,
}

/// Query evaluator that uses the context's configured execution tier.
///
/// Despite the name, no machine code is generated; see the module docs.
pub struct JitEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
}

/// Column-at-a-time expression evaluator.
pub struct OptimizedEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self {
            dataframe,
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
            enable_short_circuit: true,
            enable_constant_folding: true,
        }
    }

    /// Create a new evaluator with optimization settings
    pub fn with_optimizations(
        dataframe: &'a DataFrame,
        context: &'a QueryContext,
        short_circuit: bool,
        constant_folding: bool,
    ) -> Self {
        Self {
            dataframe,
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
            enable_short_circuit: short_circuit,
            enable_constant_folding: constant_folding,
        }
    }

    /// Evaluate an expression row by row and return a boolean mask.
    pub fn evaluate_query(&self, expr: &Expr) -> Result<Vec<bool>> {
        let prepared = self.prepare(expr)?;
        self.evaluate_prepared_query(&prepared)
    }

    /// Evaluate an already-prepared expression row by row.
    fn evaluate_prepared_query(&self, expr: &Expr) -> Result<Vec<bool>> {
        let start = Instant::now();
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(expr, row_idx)?;
            result.push(ops::value_to_bool(&value)?);
        }

        if let Ok(mut stats) = lock_safe!(
            self.context.jit_stats,
            "query evaluator context jit stats lock"
        ) {
            stats.record_row_by_row_execution(start.elapsed().as_nanos() as u64);
        }

        Ok(result)
    }

    /// Evaluate a query using the tier configured on the context.
    ///
    /// The expression is prepared (constant folded) once per context and cached,
    /// then executed either column-at-a-time or row-by-row. Both tiers return
    /// the same mask.
    pub fn evaluate_query_with_jit(&self, expr: &Expr) -> Result<Vec<bool>> {
        let prepared = self.prepared_expression(expr)?;

        if self.context.jit_enabled {
            let start = Instant::now();
            let mask =
                VectorizedEvaluator::new(self.dataframe, self.context).evaluate_mask(&prepared)?;
            if let Ok(mut stats) = lock_safe!(
                self.context.jit_stats,
                "query evaluator context jit stats lock"
            ) {
                stats.record_vectorized_execution(start.elapsed().as_nanos() as u64);
            }
            Ok(mask)
        } else {
            self.evaluate_prepared_query(&prepared)
        }
    }

    /// Prepare an expression for execution (constant folding when enabled).
    fn prepare(&self, expr: &Expr) -> Result<Expr> {
        if self.enable_constant_folding {
            self.optimize_expression(expr)
        } else {
            Ok(expr.clone())
        }
    }

    /// Prepare an expression, reusing the context's cache when possible.
    fn prepared_expression(&self, expr: &Expr) -> Result<Expr> {
        let signature = self.expression_signature(expr);

        let cache_prepared = {
            let mut cache = lock_safe!(
                self.context.compiled_expressions,
                "query evaluator context compiled expressions lock"
            )?;
            match cache.get_mut(&signature) {
                Some(entry) => {
                    entry.execution_count += 1;
                    entry.last_execution = std::time::SystemTime::now();
                    if let Some(prepared) = &entry.prepared {
                        return Ok(prepared.clone());
                    }
                    entry.execution_count >= self.context.jit_threshold
                }
                None => {
                    cache.insert(
                        signature.clone(),
                        CompiledExpression {
                            signature: signature.clone(),
                            prepared: None,
                            execution_count: 1,
                            last_execution: std::time::SystemTime::now(),
                        },
                    );
                    1 >= self.context.jit_threshold
                }
            }
        };

        let start = Instant::now();
        let prepared = self.prepare(expr)?;
        if let Ok(mut stats) = lock_safe!(
            self.context.jit_stats,
            "query evaluator context jit stats lock"
        ) {
            stats.record_compilation(start.elapsed().as_nanos() as u64);
        }

        if cache_prepared {
            let mut cache = lock_safe!(
                self.context.compiled_expressions,
                "query evaluator context compiled expressions lock"
            )?;
            if let Some(entry) = cache.get_mut(&signature) {
                entry.prepared = Some(prepared.clone());
            }
        }

        Ok(prepared)
    }

    /// Generate a signature for an expression for caching
    fn expression_signature(&self, expr: &Expr) -> String {
        format!("{:?}", expr) // Simple signature based on Debug output
    }

    /// Optimize expression through constant folding and algebraic simplifications
    fn optimize_expression(&self, expr: &Expr) -> Result<Expr> {
        match expr {
            Expr::Binary { left, op, right } => {
                let optimized_left = self.optimize_expression(left)?;
                let optimized_right = self.optimize_expression(right)?;

                // Constant folding: if both operands are literals, evaluate at compile time
                if let (Expr::Literal(l), Expr::Literal(r)) = (&optimized_left, &optimized_right) {
                    let result = self.apply_binary_operation(l, op, r)?;
                    return Ok(Expr::Literal(result));
                }

                // Algebraic simplifications
                match (&optimized_left, op, &optimized_right) {
                    // AND optimizations: x && true = x, x && false = false
                    (expr, BinaryOp::And, Expr::Literal(LiteralValue::Boolean(true))) => {
                        Ok(expr.clone())
                    }
                    (Expr::Literal(LiteralValue::Boolean(true)), BinaryOp::And, expr) => {
                        Ok(expr.clone())
                    }

                    // OR optimizations: x || false = x
                    (expr, BinaryOp::Or, Expr::Literal(LiteralValue::Boolean(false))) => {
                        Ok(expr.clone())
                    }
                    (Expr::Literal(LiteralValue::Boolean(false)), BinaryOp::Or, expr) => {
                        Ok(expr.clone())
                    }

                    // Arithmetic optimizations: x + 0 = x, x * 1 = x
                    (expr, BinaryOp::Add, Expr::Literal(LiteralValue::Number(n))) if *n == 0.0 => {
                        Ok(expr.clone())
                    }
                    (Expr::Literal(LiteralValue::Number(n)), BinaryOp::Add, expr) if *n == 0.0 => {
                        Ok(expr.clone())
                    }
                    (expr, BinaryOp::Multiply, Expr::Literal(LiteralValue::Number(n)))
                        if *n == 1.0 =>
                    {
                        Ok(expr.clone())
                    }
                    (Expr::Literal(LiteralValue::Number(n)), BinaryOp::Multiply, expr)
                        if *n == 1.0 =>
                    {
                        Ok(expr.clone())
                    }

                    _ => Ok(Expr::Binary {
                        left: Box::new(optimized_left),
                        op: op.clone(),
                        right: Box::new(optimized_right),
                    }),
                }
            }

            Expr::Unary { op, operand } => {
                let optimized_operand = self.optimize_expression(operand)?;

                // Constant folding for unary operations
                if let Expr::Literal(val) = &optimized_operand {
                    let result = self.apply_unary_operation(op, val)?;
                    return Ok(Expr::Literal(result));
                }

                // Double negation elimination: !!x = x
                if let (
                    UnaryOp::Not,
                    Expr::Unary {
                        op: UnaryOp::Not,
                        operand,
                    },
                ) = (op, &optimized_operand)
                {
                    return Ok((**operand).clone());
                }

                Ok(Expr::Unary {
                    op: op.clone(),
                    operand: Box::new(optimized_operand),
                })
            }

            Expr::Function { name, args } => {
                let optimized_args: Result<Vec<Expr>> = args
                    .iter()
                    .map(|arg| self.optimize_expression(arg))
                    .collect();

                Ok(Expr::Function {
                    name: name.clone(),
                    args: optimized_args?,
                })
            }

            _ => Ok(expr.clone()),
        }
    }

    /// Evaluate an expression for a specific row
    pub fn evaluate_expression_for_row(&self, expr: &Expr, row_idx: usize) -> Result<LiteralValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),

            Expr::Variable(name) => match self.context.variables.get(name) {
                Some(value) => Ok(value.clone()),
                None => Err(Error::InvalidValue(format!(
                    "Undefined query variable '@{}'",
                    name
                ))),
            },

            Expr::Column(name) => {
                // Use cached column data if available
                {
                    let cache = self.column_cache.borrow();
                    if let Some(cached) = cache.get(name) {
                        return cached.cell_at(row_idx);
                    }
                }

                if !self.dataframe.contains_column(name) {
                    // An identifier that is not a column may be a context
                    // variable bound through `QueryContext::set_variable`.
                    if let Some(value) = self.context.variables.get(name) {
                        return Ok(value.clone());
                    }
                    return Err(Error::ColumnNotFound(name.clone()));
                }

                // Cache miss - load the column through the shared loader, so
                // this tier classifies columns exactly like the vectorized one.
                let loaded = Rc::new(load_column_values(self.dataframe, name)?);
                let value = loaded.cell_at(row_idx);
                self.column_cache
                    .borrow_mut()
                    .insert(name.clone(), Rc::clone(&loaded));
                value
            }

            Expr::Binary { left, op, right } => {
                // Implement short-circuiting for logical operations
                if self.enable_short_circuit {
                    match op {
                        BinaryOp::And => {
                            let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                            if let LiteralValue::Boolean(false) = left_val {
                                return Ok(LiteralValue::Boolean(false)); // Short-circuit: false && x = false
                            }
                            let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                            self.apply_binary_operation(&left_val, op, &right_val)
                        }
                        BinaryOp::Or => {
                            let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                            if let LiteralValue::Boolean(true) = left_val {
                                return Ok(LiteralValue::Boolean(true)); // Short-circuit: true || x = true
                            }
                            let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                            self.apply_binary_operation(&left_val, op, &right_val)
                        }
                        _ => {
                            let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                            let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                            self.apply_binary_operation(&left_val, op, &right_val)
                        }
                    }
                } else {
                    let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                    let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                    self.apply_binary_operation(&left_val, op, &right_val)
                }
            }

            Expr::Unary { op, operand } => {
                let operand_val = self.evaluate_expression_for_row(operand, row_idx)?;
                self.apply_unary_operation(op, &operand_val)
            }

            Expr::Function { name, args } => {
                let arg_values: Result<Vec<f64>> = args
                    .iter()
                    .map(|arg| {
                        let val = self.evaluate_expression_for_row(arg, row_idx)?;
                        match val {
                            LiteralValue::Number(n) => Ok(n),
                            LiteralValue::String(s) => ops::text_to_number(&s)
                                .ok_or_else(|| ops::non_numeric_text_error(&s)),
                            LiteralValue::Boolean(_) => Err(Error::InvalidValue(
                                "Function arguments must be numeric".to_string(),
                            )),
                        }
                    })
                    .collect();

                let arg_values = arg_values?;

                if let Some(func) = self.context.functions.get(name) {
                    let result = func(&arg_values);
                    Ok(LiteralValue::Number(result))
                } else {
                    Err(Error::InvalidValue(format!("Unknown function: {}", name)))
                }
            }
        }
    }

    /// Apply binary operation (shared with every other evaluation path)
    fn apply_binary_operation(
        &self,
        left: &LiteralValue,
        op: &BinaryOp,
        right: &LiteralValue,
    ) -> Result<LiteralValue> {
        ops::apply_binary(left, op, right)
    }

    /// Apply unary operation (shared with every other evaluation path)
    fn apply_unary_operation(&self, op: &UnaryOp, operand: &LiteralValue) -> Result<LiteralValue> {
        ops::apply_unary(op, operand)
    }
}

impl<'a> JitEvaluator<'a> {
    /// Create a new evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self { dataframe, context }
    }

    /// Evaluate a query with the context's configured execution tier.
    ///
    /// Every expression form is supported. An expression the vectorized tier
    /// cannot handle is an error, never a silently unfiltered result.
    pub fn evaluate_query_jit(&self, expr: &Expr) -> Result<Vec<bool>> {
        Evaluator::new(self.dataframe, self.context).evaluate_query_with_jit(expr)
    }
}

impl<'a> OptimizedEvaluator<'a> {
    /// Create a new optimized evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self { dataframe, context }
    }

    /// Evaluate a query column-at-a-time.
    ///
    /// Unlike [`Evaluator::evaluate_query_with_jit`] this always uses the
    /// vectorized tier, regardless of the context's tier setting.
    pub fn evaluate_query_vectorized(&self, expr: &Expr) -> Result<Vec<bool>> {
        let start = Instant::now();
        let mask = VectorizedEvaluator::new(self.dataframe, self.context).evaluate_mask(expr)?;

        if let Ok(mut stats) = lock_safe!(
            self.context.jit_stats,
            "query evaluator context jit stats lock"
        ) {
            stats.record_vectorized_execution(start.elapsed().as_nanos() as u64);
        }

        Ok(mask)
    }
}

// Manual Clone implementation for QueryContext since functions can't be cloned
impl Clone for QueryContext {
    fn clone(&self) -> Self {
        let mut new_context = Self {
            variables: self.variables.clone(),
            functions: HashMap::new(),
            compiled_expressions: Arc::clone(&self.compiled_expressions),
            jit_stats: Arc::clone(&self.jit_stats),
            jit_threshold: self.jit_threshold,
            jit_enabled: self.jit_enabled,
        };

        // Re-add built-in functions
        new_context.add_builtin_functions();

        new_context
    }
}
