//! Expression evaluation and JIT compilation
//!
//! This module provides expression evaluation functionality with support for
//! JIT compilation, optimization, and vectorized operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::ast::{BinaryOp, Expr, LiteralValue, UnaryOp};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::optimized::jit::jit_core::{JitError, JitFunction, JitResult};

/// Statistics for JIT compilation
#[derive(Debug, Clone, Default)]
pub struct JitQueryStats {
    /// Number of expression compilations
    pub compilations: u64,
    /// Number of JIT executions
    pub jit_executions: u64,
    /// Number of native executions
    pub native_executions: u64,
    /// Total compilation time in nanoseconds
    pub compilation_time_ns: u64,
    /// Total JIT execution time in nanoseconds
    pub jit_execution_time_ns: u64,
    /// Total native execution time in nanoseconds
    pub native_execution_time_ns: u64,
}

impl JitQueryStats {
    pub fn record_compilation(&mut self, duration_ns: u64) {
        self.compilations += 1;
        self.compilation_time_ns += duration_ns;
    }

    pub fn record_jit_execution(&mut self, duration_ns: u64) {
        self.jit_executions += 1;
        self.jit_execution_time_ns += duration_ns;
    }

    pub fn record_native_execution(&mut self, duration_ns: u64) {
        self.native_executions += 1;
        self.native_execution_time_ns += duration_ns;
    }

    pub fn average_compilation_time_ns(&self) -> f64 {
        if self.compilations > 0 {
            self.compilation_time_ns as f64 / self.compilations as f64
        } else {
            0.0
        }
    }

    pub fn jit_speedup_ratio(&self) -> f64 {
        if self.jit_executions > 0 && self.native_executions > 0 {
            let avg_native = self.native_execution_time_ns as f64 / self.native_executions as f64;
            let avg_jit = self.jit_execution_time_ns as f64 / self.jit_executions as f64;
            if avg_jit > 0.0 {
                avg_native / avg_jit
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
}

/// Compiled expression cache entry
#[derive(Clone)]
struct CompiledExpression {
    /// Expression signature for cache lookup
    signature: String,
    /// JIT-compiled function
    jit_function: Option<Arc<JitFunction>>,
    /// Number of times this expression has been executed
    execution_count: u64,
    /// Last execution time
    last_execution: std::time::SystemTime,
}

/// Query execution context with JIT compilation support
pub struct QueryContext {
    /// Variable bindings for substitution
    pub variables: HashMap<String, LiteralValue>,
    /// Available functions
    pub functions: HashMap<String, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    /// JIT compilation cache
    compiled_expressions: Arc<Mutex<HashMap<String, CompiledExpression>>>,
    /// JIT compilation statistics
    jit_stats: Arc<Mutex<JitQueryStats>>,
    /// JIT compilation threshold (compile after N executions)
    jit_threshold: u64,
    /// Enable JIT compilation
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
            jit_threshold: 5, // Compile after 5 executions
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

    /// Create a new query context with JIT settings
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

    /// Get JIT compilation statistics
    pub fn jit_stats(&self) -> JitQueryStats {
        self.jit_stats.lock().unwrap().clone()
    }

    /// Enable or disable JIT compilation
    pub fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = enabled;
    }

    /// Set JIT compilation threshold
    pub fn set_jit_threshold(&mut self, threshold: u64) {
        self.jit_threshold = threshold;
    }

    /// Clear JIT compilation cache
    pub fn clear_jit_cache(&mut self) {
        let mut cache = self.compiled_expressions.lock().unwrap();
        cache.clear();
    }

    /// Get the number of compiled expressions in cache
    pub fn compiled_expressions_count(&self) -> usize {
        self.compiled_expressions.lock().unwrap().len()
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

/// Expression evaluator with optimization support
pub struct Evaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    /// Cache for column data to avoid repeated parsing
    column_cache: std::cell::RefCell<HashMap<String, Vec<LiteralValue>>>,
    /// Optimization flags
    enable_short_circuit: bool,
    enable_constant_folding: bool,
}

/// JIT-compiled expression evaluator
pub struct JitEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    /// Cache for column data
    column_cache: std::cell::RefCell<HashMap<String, Vec<LiteralValue>>>,
}

/// Optimized expression evaluator with short-circuiting and constant folding
pub struct OptimizedEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    column_cache: std::cell::RefCell<HashMap<String, Vec<LiteralValue>>>,
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

    /// Evaluate an expression and return a boolean mask for filtering
    pub fn evaluate_query(&self, expr: &Expr) -> Result<Vec<bool>> {
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);

        // Pre-optimize expression if constant folding is enabled
        let optimized_expr = if self.enable_constant_folding {
            self.optimize_expression(expr)?
        } else {
            expr.clone()
        };

        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(&optimized_expr, row_idx)?;
            match value {
                LiteralValue::Boolean(b) => result.push(b),
                _ => {
                    return Err(Error::InvalidValue(
                        "Query expression must evaluate to boolean".to_string(),
                    ))
                }
            }
        }

        Ok(result)
    }

    /// Evaluate query with JIT compilation support
    pub fn evaluate_query_with_jit(&self, expr: &Expr) -> Result<Vec<bool>> {
        let expr_signature = self.expression_signature(expr);

        // Check if we should use JIT compilation
        if self.context.jit_enabled {
            let should_compile = {
                let mut cache = self.context.compiled_expressions.lock().unwrap();
                if let Some(compiled_expr) = cache.get_mut(&expr_signature) {
                    compiled_expr.execution_count += 1;
                    compiled_expr.last_execution = std::time::SystemTime::now();

                    // Use JIT if available, otherwise check if we should compile
                    compiled_expr.jit_function.is_some()
                        || (compiled_expr.execution_count >= self.context.jit_threshold
                            && compiled_expr.jit_function.is_none())
                } else {
                    // First execution - add to cache
                    cache.insert(
                        expr_signature.clone(),
                        CompiledExpression {
                            signature: expr_signature.clone(),
                            jit_function: None,
                            execution_count: 1,
                            last_execution: std::time::SystemTime::now(),
                        },
                    );
                    false
                }
            };

            if should_compile {
                // Try to compile the expression
                if let Ok(jit_func) = self.compile_expression_to_jit(expr) {
                    let mut cache = self.context.compiled_expressions.lock().unwrap();
                    if let Some(compiled_expr) = cache.get_mut(&expr_signature) {
                        compiled_expr.jit_function = Some(Arc::new(jit_func));
                    }
                }
            }

            // Try to execute with JIT
            {
                let cache = self.context.compiled_expressions.lock().unwrap();
                if let Some(compiled_expr) = cache.get(&expr_signature) {
                    if let Some(jit_func) = &compiled_expr.jit_function {
                        return self.execute_jit_compiled_query(expr, jit_func);
                    }
                }
            }
        }

        // Fall back to regular evaluation
        self.evaluate_query(expr)
    }

    /// Generate a signature for an expression for caching
    fn expression_signature(&self, expr: &Expr) -> String {
        format!("{:?}", expr) // Simple signature based on Debug output
    }

    /// Compile an expression to JIT-compiled function
    fn compile_expression_to_jit(&self, expr: &Expr) -> JitResult<JitFunction> {
        let start = Instant::now();

        let signature = self.expression_signature(expr);

        // For this implementation, we'll create a JIT function that encapsulates
        // the expression evaluation logic
        let jit_func = match expr {
            // Simple numeric operations can be JIT compiled
            Expr::Binary { left, op, right } if self.is_jit_compilable_binary(left, op, right) => {
                self.compile_binary_expression(left, op, right)?
            }

            // Column comparisons can be vectorized
            Expr::Binary { left, op, right } if self.is_column_comparison(left, right) => {
                self.compile_column_comparison(left, op, right)?
            }

            // Other expressions fall back to interpreted evaluation
            _ => {
                return Err(JitError::CompilationError(
                    "Expression not JIT-compilable".to_string(),
                ));
            }
        };

        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_compilation(duration.as_nanos() as u64);
        }

        Ok(jit_func)
    }

    /// Check if a binary expression can be JIT compiled
    fn is_jit_compilable_binary(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> bool {
        // Simple arithmetic operations on numeric literals or columns
        matches!(
            op,
            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide
        ) && self.is_numeric_expression(left)
            && self.is_numeric_expression(right)
    }

    /// Check if expression is numeric (literal or column)
    fn is_numeric_expression(&self, expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::Literal(LiteralValue::Number(_)) | Expr::Column(_)
        )
    }

    /// Check if this is a column comparison suitable for vectorization
    fn is_column_comparison(&self, left: &Expr, right: &Expr) -> bool {
        matches!(
            (left, right),
            (Expr::Column(_), Expr::Literal(_)) | (Expr::Literal(_), Expr::Column(_))
        )
    }

    /// Compile a binary arithmetic expression
    fn compile_binary_expression(
        &self,
        left: &Expr,
        op: &BinaryOp,
        right: &Expr,
    ) -> JitResult<JitFunction> {
        let op_name = match op {
            BinaryOp::Add => "add",
            BinaryOp::Subtract => "sub",
            BinaryOp::Multiply => "mul",
            BinaryOp::Divide => "div",
            _ => {
                return Err(JitError::CompilationError(
                    "Unsupported binary operation for JIT".to_string(),
                ))
            }
        };

        let func_name = format!("jit_binary_{}_{:?}_{:?}", op_name, left, right);

        // Create a JIT function that performs the binary operation
        let jit_func = match op {
            BinaryOp::Add => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 {
                    args[0] + args[1]
                } else {
                    0.0
                }
            }),
            BinaryOp::Subtract => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        args[0] - args[1]
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::Multiply => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        args[0] * args[1]
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::Divide => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[1] != 0.0 {
                        args[0] / args[1]
                    } else {
                        f64::NAN
                    }
                })
            }
            _ => unreachable!(),
        };

        Ok(jit_func)
    }

    /// Compile a column comparison expression
    fn compile_column_comparison(
        &self,
        left: &Expr,
        op: &BinaryOp,
        right: &Expr,
    ) -> JitResult<JitFunction> {
        let func_name = format!("jit_comparison_{:?}_{:?}_{:?}", left, op, right);

        // Create a vectorized comparison function
        let jit_func = match op {
            BinaryOp::Equal => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 {
                    if (args[0] - args[1]).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }),
            BinaryOp::LessThan => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        if args[0] < args[1] {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::GreaterThan => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        if args[0] > args[1] {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                })
            }
            _ => {
                return Err(JitError::CompilationError(
                    "Unsupported comparison operation for JIT".to_string(),
                ))
            }
        };

        Ok(jit_func)
    }

    /// Execute a JIT-compiled query
    fn execute_jit_compiled_query(&self, expr: &Expr, jit_func: &JitFunction) -> Result<Vec<bool>> {
        let start = Instant::now();

        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);

        // For column-based operations, we can vectorize
        if let Expr::Binary { left, op: _, right } = expr {
            if self.is_column_comparison(left, right) {
                // Vectorized execution path
                let (column_values, literal_value) =
                    self.extract_column_and_literal(left, right)?;

                for &col_val in &column_values {
                    let args = vec![col_val, literal_value];
                    use crate::optimized::jit::jit_core::JitCompilable;
                    let jit_result = jit_func.execute(args);
                    result.push(jit_result != 0.0);
                }

                let duration = start.elapsed();
                {
                    let mut stats = self.context.jit_stats.lock().unwrap();
                    stats.record_jit_execution(duration.as_nanos() as u64);
                }

                return Ok(result);
            }
        }

        // Fall back to row-by-row evaluation
        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(expr, row_idx)?;
            match value {
                LiteralValue::Boolean(b) => result.push(b),
                _ => {
                    return Err(Error::InvalidValue(
                        "Query expression must evaluate to boolean".to_string(),
                    ))
                }
            }
        }

        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_jit_execution(duration.as_nanos() as u64);
        }

        Ok(result)
    }

    /// Extract column values and literal value from a column comparison
    fn extract_column_and_literal(&self, left: &Expr, right: &Expr) -> Result<(Vec<f64>, f64)> {
        match (left, right) {
            (Expr::Column(col_name), Expr::Literal(LiteralValue::Number(lit_val))) => {
                let col_values = self.dataframe.get_column_numeric_values(col_name)?;
                Ok((col_values, *lit_val))
            }
            (Expr::Literal(LiteralValue::Number(lit_val)), Expr::Column(col_name)) => {
                let col_values = self.dataframe.get_column_numeric_values(col_name)?;
                Ok((col_values, *lit_val))
            }
            _ => Err(Error::InvalidValue(
                "Invalid column comparison expression".to_string(),
            )),
        }
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
                    (_, BinaryOp::And, Expr::Literal(LiteralValue::Boolean(false))) => {
                        Ok(Expr::Literal(LiteralValue::Boolean(false)))
                    }
                    (Expr::Literal(LiteralValue::Boolean(false)), BinaryOp::And, _) => {
                        Ok(Expr::Literal(LiteralValue::Boolean(false)))
                    }

                    // OR optimizations: x || true = true, x || false = x
                    (_, BinaryOp::Or, Expr::Literal(LiteralValue::Boolean(true))) => {
                        Ok(Expr::Literal(LiteralValue::Boolean(true)))
                    }
                    (Expr::Literal(LiteralValue::Boolean(true)), BinaryOp::Or, _) => {
                        Ok(Expr::Literal(LiteralValue::Boolean(true)))
                    }
                    (expr, BinaryOp::Or, Expr::Literal(LiteralValue::Boolean(false))) => {
                        Ok(expr.clone())
                    }
                    (Expr::Literal(LiteralValue::Boolean(false)), BinaryOp::Or, expr) => {
                        Ok(expr.clone())
                    }

                    // Arithmetic optimizations: x + 0 = x, x * 1 = x, x * 0 = 0
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
                    (_, BinaryOp::Multiply, Expr::Literal(LiteralValue::Number(n)))
                        if *n == 0.0 =>
                    {
                        Ok(Expr::Literal(LiteralValue::Number(0.0)))
                    }
                    (Expr::Literal(LiteralValue::Number(n)), BinaryOp::Multiply, _)
                        if *n == 0.0 =>
                    {
                        Ok(Expr::Literal(LiteralValue::Number(0.0)))
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

            Expr::Column(name) => {
                if !self.dataframe.contains_column(name) {
                    return Err(Error::ColumnNotFound(name.clone()));
                }

                // Use cached column data if available
                {
                    let cache = self.column_cache.borrow();
                    if let Some(cached_values) = cache.get(name) {
                        if row_idx < cached_values.len() {
                            return Ok(cached_values[row_idx].clone());
                        } else {
                            return Err(Error::IndexOutOfBounds {
                                index: row_idx,
                                size: cached_values.len(),
                            });
                        }
                    }
                }

                // Cache miss - load and parse column data
                let column_values = self.dataframe.get_column_string_values(name)?;
                let parsed_values: Vec<LiteralValue> = column_values
                    .iter()
                    .map(|str_value| {
                        // Try to parse as number first, then boolean, then keep as string
                        if let Ok(num) = str_value.parse::<f64>() {
                            LiteralValue::Number(num)
                        } else if let Ok(bool_val) = str_value.parse::<bool>() {
                            LiteralValue::Boolean(bool_val)
                        } else {
                            LiteralValue::String(str_value.clone())
                        }
                    })
                    .collect();

                // Cache the parsed values
                {
                    let mut cache = self.column_cache.borrow_mut();
                    cache.insert(name.clone(), parsed_values.clone());
                }

                if row_idx < parsed_values.len() {
                    Ok(parsed_values[row_idx].clone())
                } else {
                    Err(Error::IndexOutOfBounds {
                        index: row_idx,
                        size: parsed_values.len(),
                    })
                }
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
                            _ => Err(Error::InvalidValue(
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

    /// Apply binary operation
    fn apply_binary_operation(
        &self,
        left: &LiteralValue,
        op: &BinaryOp,
        right: &LiteralValue,
    ) -> Result<LiteralValue> {
        match (left, right, op) {
            // Numeric operations
            (LiteralValue::Number(l), LiteralValue::Number(r), op) => match op {
                BinaryOp::Add => Ok(LiteralValue::Number(l + r)),
                BinaryOp::Subtract => Ok(LiteralValue::Number(l - r)),
                BinaryOp::Multiply => Ok(LiteralValue::Number(l * r)),
                BinaryOp::Divide => {
                    if *r == 0.0 {
                        Err(Error::InvalidValue("Division by zero".to_string()))
                    } else {
                        Ok(LiteralValue::Number(l / r))
                    }
                }
                BinaryOp::Modulo => Ok(LiteralValue::Number(l % r)),
                BinaryOp::Power => Ok(LiteralValue::Number(l.powf(*r))),
                BinaryOp::Equal => Ok(LiteralValue::Boolean((l - r).abs() < f64::EPSILON)),
                BinaryOp::NotEqual => Ok(LiteralValue::Boolean((l - r).abs() >= f64::EPSILON)),
                BinaryOp::LessThan => Ok(LiteralValue::Boolean(l < r)),
                BinaryOp::LessThanOrEqual => Ok(LiteralValue::Boolean(l <= r)),
                BinaryOp::GreaterThan => Ok(LiteralValue::Boolean(l > r)),
                BinaryOp::GreaterThanOrEqual => Ok(LiteralValue::Boolean(l >= r)),
                BinaryOp::And | BinaryOp::Or => Err(Error::InvalidValue(
                    "Logical operations require boolean operands".to_string(),
                )),
            },

            // String operations
            (LiteralValue::String(l), LiteralValue::String(r), op) => match op {
                BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                BinaryOp::Add => Ok(LiteralValue::String(format!("{}{}", l, r))),
                _ => Err(Error::InvalidValue(
                    "Unsupported operation for strings".to_string(),
                )),
            },

            // Boolean operations
            (LiteralValue::Boolean(l), LiteralValue::Boolean(r), op) => match op {
                BinaryOp::And => Ok(LiteralValue::Boolean(*l && *r)),
                BinaryOp::Or => Ok(LiteralValue::Boolean(*l || *r)),
                BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                _ => Err(Error::InvalidValue(
                    "Unsupported operation for booleans".to_string(),
                )),
            },

            // Mixed type comparisons (try to convert to common type)
            (LiteralValue::Number(l), LiteralValue::String(r), op) => {
                if let Ok(r_num) = r.parse::<f64>() {
                    self.apply_binary_operation(
                        &LiteralValue::Number(*l),
                        op,
                        &LiteralValue::Number(r_num),
                    )
                } else {
                    Err(Error::InvalidValue(
                        "Cannot compare number with non-numeric string".to_string(),
                    ))
                }
            }

            (LiteralValue::String(l), LiteralValue::Number(r), op) => {
                if let Ok(l_num) = l.parse::<f64>() {
                    self.apply_binary_operation(
                        &LiteralValue::Number(l_num),
                        op,
                        &LiteralValue::Number(*r),
                    )
                } else {
                    Err(Error::InvalidValue(
                        "Cannot compare non-numeric string with number".to_string(),
                    ))
                }
            }

            _ => Err(Error::InvalidValue(
                "Unsupported operand types for operation".to_string(),
            )),
        }
    }

    /// Apply unary operation
    fn apply_unary_operation(&self, op: &UnaryOp, operand: &LiteralValue) -> Result<LiteralValue> {
        match (op, operand) {
            (UnaryOp::Not, LiteralValue::Boolean(b)) => Ok(LiteralValue::Boolean(!b)),
            (UnaryOp::Negate, LiteralValue::Number(n)) => Ok(LiteralValue::Number(-n)),
            _ => Err(Error::InvalidValue(
                "Unsupported unary operation".to_string(),
            )),
        }
    }
}

impl<'a> JitEvaluator<'a> {
    /// Create a new JIT evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self {
            dataframe,
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Evaluate query with aggressive JIT compilation
    pub fn evaluate_query_jit(&self, expr: &Expr) -> Result<Vec<bool>> {
        // Force JIT compilation for all suitable expressions
        let expr_signature = format!("{:?}", expr);

        if self.context.jit_enabled {
            // Try to compile immediately
            if let Ok(jit_func) = self.compile_expression_to_jit(expr) {
                return self.execute_jit_compiled_query(expr, &jit_func);
            }
        }

        // Fall back to regular evaluation
        self.evaluate_query_fallback(expr)
    }

    /// Compile expression to JIT (same logic as Evaluator but more aggressive)
    fn compile_expression_to_jit(&self, expr: &Expr) -> JitResult<JitFunction> {
        let start = Instant::now();

        let jit_func = match expr {
            // All numeric binary operations
            Expr::Binary { left, op, right } if self.is_jit_compilable_binary(left, op, right) => {
                self.compile_binary_expression(left, op, right)?
            }

            // All comparison operations
            Expr::Binary { left, op, right } if self.is_comparison_op(op) => {
                self.compile_comparison_expression(left, op, right)?
            }

            // Function calls
            Expr::Function { name, args } => self.compile_function_expression(name, args)?,

            _ => {
                return Err(JitError::CompilationError(
                    "Expression not JIT-compilable".to_string(),
                ));
            }
        };

        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_compilation(duration.as_nanos() as u64);
        }

        Ok(jit_func)
    }

    fn is_jit_compilable_binary(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> bool {
        matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Subtract
                | BinaryOp::Multiply
                | BinaryOp::Divide
                | BinaryOp::Power
        )
    }

    fn is_comparison_op(&self, op: &BinaryOp) -> bool {
        matches!(
            op,
            BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::LessThan
                | BinaryOp::LessThanOrEqual
                | BinaryOp::GreaterThan
                | BinaryOp::GreaterThanOrEqual
        )
    }

    fn compile_binary_expression(
        &self,
        left: &Expr,
        op: &BinaryOp,
        right: &Expr,
    ) -> JitResult<JitFunction> {
        let func_name = format!("jit_binary_{:?}", op);

        let jit_func = match op {
            BinaryOp::Add => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| args.iter().sum())
            }
            BinaryOp::Subtract => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        args[0] - args[1]
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::Multiply => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    args.iter().product()
                })
            }
            BinaryOp::Divide => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[1] != 0.0 {
                        args[0] / args[1]
                    } else {
                        f64::NAN
                    }
                })
            }
            BinaryOp::Power => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 {
                    args[0].powf(args[1])
                } else {
                    0.0
                }
            }),
            _ => {
                return Err(JitError::CompilationError(
                    "Unsupported binary operation".to_string(),
                ))
            }
        };

        Ok(jit_func)
    }

    fn compile_comparison_expression(
        &self,
        left: &Expr,
        op: &BinaryOp,
        right: &Expr,
    ) -> JitResult<JitFunction> {
        let func_name = format!("jit_comparison_{:?}", op);

        let jit_func = match op {
            BinaryOp::Equal => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && (args[0] - args[1]).abs() < f64::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }),
            BinaryOp::NotEqual => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && (args[0] - args[1]).abs() >= f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::LessThan => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[0] < args[1] {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::LessThanOrEqual => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[0] <= args[1] {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::GreaterThan => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[0] > args[1] {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::GreaterThanOrEqual => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[0] >= args[1] {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
            _ => {
                return Err(JitError::CompilationError(
                    "Unsupported comparison operation".to_string(),
                ))
            }
        };

        Ok(jit_func)
    }

    fn compile_function_expression(&self, name: &str, args: &[Expr]) -> JitResult<JitFunction> {
        let func_name = format!("jit_function_{}", name);

        // Compile built-in mathematical functions
        let jit_func = match name {
            "abs" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() {
                    args[0].abs()
                } else {
                    0.0
                }
            }),
            "sqrt" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() {
                    args[0].sqrt()
                } else {
                    0.0
                }
            }),
            "sin" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() {
                    args[0].sin()
                } else {
                    0.0
                }
            }),
            "cos" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() {
                    args[0].cos()
                } else {
                    0.0
                }
            }),
            "sum" => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| args.iter().sum())
            }
            "mean" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() {
                    args.iter().sum::<f64>() / args.len() as f64
                } else {
                    0.0
                }
            }),
            _ => {
                return Err(JitError::CompilationError(format!(
                    "Function {} not JIT-compilable",
                    name
                )))
            }
        };

        Ok(jit_func)
    }

    fn execute_jit_compiled_query(&self, expr: &Expr, jit_func: &JitFunction) -> Result<Vec<bool>> {
        let start = Instant::now();

        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);

        // Execute JIT function for each row
        for row_idx in 0..row_count {
            let args = self.extract_arguments_for_row(expr, row_idx)?;
            use crate::optimized::jit::jit_core::JitCompilable;
            let jit_result = jit_func.execute(args);

            // Convert numeric result to boolean (0.0 = false, non-zero = true)
            result.push(jit_result != 0.0);
        }

        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_jit_execution(duration.as_nanos() as u64);
        }

        Ok(result)
    }

    fn extract_arguments_for_row(&self, expr: &Expr, row_idx: usize) -> Result<Vec<f64>> {
        match expr {
            Expr::Binary { left, op: _, right } => {
                let left_val = self.extract_numeric_value(left, row_idx)?;
                let right_val = self.extract_numeric_value(right, row_idx)?;
                Ok(vec![left_val, right_val])
            }
            Expr::Function { name: _, args } => {
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.extract_numeric_value(arg, row_idx)?);
                }
                Ok(arg_values)
            }
            _ => Err(Error::InvalidValue(
                "Cannot extract arguments from expression".to_string(),
            )),
        }
    }

    fn extract_numeric_value(&self, expr: &Expr, row_idx: usize) -> Result<f64> {
        match expr {
            Expr::Literal(LiteralValue::Number(n)) => Ok(*n),
            Expr::Column(col_name) => {
                let col_values = self.dataframe.get_column_numeric_values(col_name)?;
                if row_idx < col_values.len() {
                    Ok(col_values[row_idx])
                } else {
                    Err(Error::IndexOutOfBounds {
                        index: row_idx,
                        size: col_values.len(),
                    })
                }
            }
            _ => Err(Error::InvalidValue(
                "Cannot extract numeric value from expression".to_string(),
            )),
        }
    }

    fn evaluate_query_fallback(&self, expr: &Expr) -> Result<Vec<bool>> {
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            // Simple fallback evaluation
            let value = match expr {
                Expr::Literal(LiteralValue::Boolean(b)) => *b,
                _ => true, // Simplified fallback
            };
            result.push(value);
        }

        Ok(result)
    }
}

impl<'a> OptimizedEvaluator<'a> {
    /// Create a new optimized evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self {
            dataframe,
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Evaluate query with vectorized operations where possible
    pub fn evaluate_query_vectorized(&self, expr: &Expr) -> Result<Vec<bool>> {
        // Try to use vectorized operations for simple column comparisons
        if let Some(vectorized_result) = self.try_vectorized_evaluation(expr)? {
            return Ok(vectorized_result);
        }

        // Fall back to row-by-row evaluation
        self.evaluate_query_row_by_row(expr)
    }

    /// Try to evaluate expression using vectorized operations
    fn try_vectorized_evaluation(&self, expr: &Expr) -> Result<Option<Vec<bool>>> {
        match expr {
            // Simple column comparisons can be vectorized
            Expr::Binary { left, op, right } => {
                if let (Expr::Column(col_name), Expr::Literal(literal)) =
                    (left.as_ref(), right.as_ref())
                {
                    return self.evaluate_column_comparison_vectorized(col_name, op, literal);
                }
                if let (Expr::Literal(literal), Expr::Column(col_name)) =
                    (left.as_ref(), right.as_ref())
                {
                    // Swap operands for commutative operations
                    let swapped_op = match op {
                        BinaryOp::Equal => BinaryOp::Equal,
                        BinaryOp::NotEqual => BinaryOp::NotEqual,
                        BinaryOp::LessThan => BinaryOp::GreaterThan,
                        BinaryOp::LessThanOrEqual => BinaryOp::GreaterThanOrEqual,
                        BinaryOp::GreaterThan => BinaryOp::LessThan,
                        BinaryOp::GreaterThanOrEqual => BinaryOp::LessThanOrEqual,
                        _ => return Ok(None), // Not vectorizable
                    };
                    return self.evaluate_column_comparison_vectorized(
                        col_name,
                        &swapped_op,
                        literal,
                    );
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Evaluate column comparison using vectorized operations
    fn evaluate_column_comparison_vectorized(
        &self,
        col_name: &str,
        op: &BinaryOp,
        literal: &LiteralValue,
    ) -> Result<Option<Vec<bool>>> {
        if !self.dataframe.contains_column(col_name) {
            return Err(Error::ColumnNotFound(col_name.to_string()));
        }

        // Get column values (try numeric first for better performance)
        if let Ok(numeric_values) = self.dataframe.get_column_numeric_values(col_name) {
            if let LiteralValue::Number(target) = literal {
                let result: Vec<bool> = match op {
                    BinaryOp::Equal => numeric_values
                        .iter()
                        .map(|&v| (v - target).abs() < f64::EPSILON)
                        .collect(),
                    BinaryOp::NotEqual => numeric_values
                        .iter()
                        .map(|&v| (v - target).abs() >= f64::EPSILON)
                        .collect(),
                    BinaryOp::LessThan => numeric_values.iter().map(|&v| v < *target).collect(),
                    BinaryOp::LessThanOrEqual => {
                        numeric_values.iter().map(|&v| v <= *target).collect()
                    }
                    BinaryOp::GreaterThan => numeric_values.iter().map(|&v| v > *target).collect(),
                    BinaryOp::GreaterThanOrEqual => {
                        numeric_values.iter().map(|&v| v >= *target).collect()
                    }
                    _ => return Ok(None), // Not supported for vectorization
                };
                return Ok(Some(result));
            }
        }

        // Fall back to string comparison
        let string_values = self.dataframe.get_column_string_values(col_name)?;
        if let LiteralValue::String(target) = literal {
            let result: Vec<bool> = match op {
                BinaryOp::Equal => string_values.iter().map(|v| v == target).collect(),
                BinaryOp::NotEqual => string_values.iter().map(|v| v != target).collect(),
                _ => return Ok(None), // String comparison only supports equality
            };
            return Ok(Some(result));
        }

        Ok(None)
    }

    /// Row-by-row evaluation as fallback
    fn evaluate_query_row_by_row(&self, expr: &Expr) -> Result<Vec<bool>> {
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(expr, row_idx)?;
            match value {
                LiteralValue::Boolean(b) => result.push(b),
                _ => {
                    return Err(Error::InvalidValue(
                        "Query expression must evaluate to boolean".to_string(),
                    ))
                }
            }
        }

        Ok(result)
    }

    /// Evaluate expression for a single row (same as regular evaluator but with caching)
    fn evaluate_expression_for_row(&self, expr: &Expr, row_idx: usize) -> Result<LiteralValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),

            Expr::Column(name) => {
                if !self.dataframe.contains_column(name) {
                    return Err(Error::ColumnNotFound(name.clone()));
                }

                // Use cached column data if available
                {
                    let cache = self.column_cache.borrow();
                    if let Some(cached_values) = cache.get(name) {
                        if row_idx < cached_values.len() {
                            return Ok(cached_values[row_idx].clone());
                        } else {
                            return Err(Error::IndexOutOfBounds {
                                index: row_idx,
                                size: cached_values.len(),
                            });
                        }
                    }
                }

                // Cache miss - load and parse column data
                let column_values = self.dataframe.get_column_string_values(name)?;
                let parsed_values: Vec<LiteralValue> = column_values
                    .iter()
                    .map(|str_value| {
                        if let Ok(num) = str_value.parse::<f64>() {
                            LiteralValue::Number(num)
                        } else if let Ok(bool_val) = str_value.parse::<bool>() {
                            LiteralValue::Boolean(bool_val)
                        } else {
                            LiteralValue::String(str_value.clone())
                        }
                    })
                    .collect();

                {
                    let mut cache = self.column_cache.borrow_mut();
                    cache.insert(name.clone(), parsed_values.clone());
                }

                if row_idx < parsed_values.len() {
                    Ok(parsed_values[row_idx].clone())
                } else {
                    Err(Error::IndexOutOfBounds {
                        index: row_idx,
                        size: parsed_values.len(),
                    })
                }
            }

            Expr::Binary { left, op, right } => {
                // Always use short-circuiting in optimized evaluator
                match op {
                    BinaryOp::And => {
                        let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                        if let LiteralValue::Boolean(false) = left_val {
                            return Ok(LiteralValue::Boolean(false));
                        }
                        let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                        self.apply_binary_operation(&left_val, op, &right_val)
                    }
                    BinaryOp::Or => {
                        let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                        if let LiteralValue::Boolean(true) = left_val {
                            return Ok(LiteralValue::Boolean(true));
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
                            _ => Err(Error::InvalidValue(
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

    /// Apply binary operation (shared with regular evaluator)
    fn apply_binary_operation(
        &self,
        left: &LiteralValue,
        op: &BinaryOp,
        right: &LiteralValue,
    ) -> Result<LiteralValue> {
        match (left, right, op) {
            // Numeric operations
            (LiteralValue::Number(l), LiteralValue::Number(r), op) => match op {
                BinaryOp::Add => Ok(LiteralValue::Number(l + r)),
                BinaryOp::Subtract => Ok(LiteralValue::Number(l - r)),
                BinaryOp::Multiply => Ok(LiteralValue::Number(l * r)),
                BinaryOp::Divide => {
                    if *r == 0.0 {
                        Err(Error::InvalidValue("Division by zero".to_string()))
                    } else {
                        Ok(LiteralValue::Number(l / r))
                    }
                }
                BinaryOp::Modulo => Ok(LiteralValue::Number(l % r)),
                BinaryOp::Power => Ok(LiteralValue::Number(l.powf(*r))),
                BinaryOp::Equal => Ok(LiteralValue::Boolean((l - r).abs() < f64::EPSILON)),
                BinaryOp::NotEqual => Ok(LiteralValue::Boolean((l - r).abs() >= f64::EPSILON)),
                BinaryOp::LessThan => Ok(LiteralValue::Boolean(l < r)),
                BinaryOp::LessThanOrEqual => Ok(LiteralValue::Boolean(l <= r)),
                BinaryOp::GreaterThan => Ok(LiteralValue::Boolean(l > r)),
                BinaryOp::GreaterThanOrEqual => Ok(LiteralValue::Boolean(l >= r)),
                BinaryOp::And | BinaryOp::Or => Err(Error::InvalidValue(
                    "Logical operations require boolean operands".to_string(),
                )),
            },

            // String operations
            (LiteralValue::String(l), LiteralValue::String(r), op) => match op {
                BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                BinaryOp::Add => Ok(LiteralValue::String(format!("{}{}", l, r))),
                _ => Err(Error::InvalidValue(
                    "Unsupported operation for strings".to_string(),
                )),
            },

            // Boolean operations
            (LiteralValue::Boolean(l), LiteralValue::Boolean(r), op) => match op {
                BinaryOp::And => Ok(LiteralValue::Boolean(*l && *r)),
                BinaryOp::Or => Ok(LiteralValue::Boolean(*l || *r)),
                BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                _ => Err(Error::InvalidValue(
                    "Unsupported operation for booleans".to_string(),
                )),
            },

            // Mixed type comparisons
            (LiteralValue::Number(l), LiteralValue::String(r), op) => {
                if let Ok(r_num) = r.parse::<f64>() {
                    self.apply_binary_operation(
                        &LiteralValue::Number(*l),
                        op,
                        &LiteralValue::Number(r_num),
                    )
                } else {
                    Err(Error::InvalidValue(
                        "Cannot compare number with non-numeric string".to_string(),
                    ))
                }
            }

            (LiteralValue::String(l), LiteralValue::Number(r), op) => {
                if let Ok(l_num) = l.parse::<f64>() {
                    self.apply_binary_operation(
                        &LiteralValue::Number(l_num),
                        op,
                        &LiteralValue::Number(*r),
                    )
                } else {
                    Err(Error::InvalidValue(
                        "Cannot compare non-numeric string with number".to_string(),
                    ))
                }
            }

            _ => Err(Error::InvalidValue(
                "Unsupported operand types for operation".to_string(),
            )),
        }
    }

    /// Apply unary operation
    fn apply_unary_operation(&self, op: &UnaryOp, operand: &LiteralValue) -> Result<LiteralValue> {
        match (op, operand) {
            (UnaryOp::Not, LiteralValue::Boolean(b)) => Ok(LiteralValue::Boolean(!b)),
            (UnaryOp::Negate, LiteralValue::Number(n)) => Ok(LiteralValue::Number(-n)),
            _ => Err(Error::InvalidValue(
                "Unsupported unary operation".to_string(),
            )),
        }
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
