//! Core JIT compilation infrastructure for pandrs
//! 
//! This module provides the fundamental JIT compilation capabilities,
//! supporting fast execution of numerical operations on arrays of data.
//! 
//! The JIT system supports multiple numeric types (f64, f32, i64, i32) and
//! provides a flexible, type-safe interface for creating JIT-compiled functions.

use std::sync::Arc;
use std::fmt;
use std::error::Error as StdError;

use super::types::{JitType, JitNumeric, TypedVector, NumericValue};

/// Error types for JIT compilation and execution
#[derive(Debug)]
pub enum JitError {
    /// Error during JIT compilation
    CompilationError(String),
    /// Error during JIT execution
    ExecutionError(String),
    /// Feature not available (JIT disabled)
    FeatureNotAvailable(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::CompilationError(s) => write!(f, "JIT compilation error: {}", s),
            JitError::ExecutionError(s) => write!(f, "JIT execution error: {}", s),
            JitError::FeatureNotAvailable(s) => write!(f, "JIT feature not available: {}", s),
        }
    }
}

impl StdError for JitError {}

/// Result type for JIT operations
pub type JitResult<T> = Result<T, JitError>;

/// Marker trait for functions that can be JIT-compiled
pub trait JitCompilable<Args, Result> {
    /// Execute the function with the given arguments
    fn execute(&self, args: Args) -> Result;
}

/// Marker trait for generic JIT-compiled numeric functions
pub trait GenericJitCompilable {
    /// Execute the function with typed vector input and return a numeric result
    fn execute_typed(&self, args: TypedVector) -> NumericValue;
    
    /// Get the supported input type
    fn input_type_name(&self) -> &'static str;
    
    /// Get the output type
    fn output_type_name(&self) -> &'static str;
}

/// Function signature for f64 array operations
pub type FloatArrayFn = dyn Fn(Vec<f64>) -> f64 + Send + Sync;

/// Function signature for f32 array operations
pub type Float32ArrayFn = dyn Fn(Vec<f32>) -> f32 + Send + Sync;

/// Function signature for i64 array operations
pub type Int64ArrayFn = dyn Fn(Vec<i64>) -> i64 + Send + Sync;

/// Function signature for i32 array operations
pub type Int32ArrayFn = dyn Fn(Vec<i32>) -> i32 + Send + Sync;

/// Represents a JIT-compiled function
#[derive(Clone)]
pub struct JitFunction<F = FloatArrayFn> {
    /// Function name for debugging and caching
    name: String,
    /// Native function to use when JIT is disabled
    native_fn: Arc<F>,
    /// Input type name
    input_type: &'static str,
    /// Output type name
    output_type: &'static str,
    /// JIT compilation context
    #[cfg(feature = "jit")]
    jit_context: Option<Arc<JitContext>>,
}

/// Runtime statistics for JIT-compiled functions
#[derive(Default, Debug, Clone)]
pub struct JitStats {
    /// Number of times the function was executed
    pub executions: u64,
    /// Total execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Number of times JIT compilation was used
    pub jit_used: u64,
    /// Number of times native fallback was used
    pub native_used: u64,
}

impl JitStats {
    /// Create a new empty stats object
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a JIT execution
    pub fn record_jit_execution(&mut self, duration_ns: u64) {
        self.executions += 1;
        self.execution_time_ns += duration_ns;
        self.jit_used += 1;
    }

    /// Record a native execution
    pub fn record_native_execution(&mut self, duration_ns: u64) {
        self.executions += 1;
        self.execution_time_ns += duration_ns;
        self.native_used += 1;
    }

    /// Get average execution time in nanoseconds
    pub fn average_execution_time_ns(&self) -> f64 {
        if self.executions > 0 {
            self.execution_time_ns as f64 / self.executions as f64
        } else {
            0.0
        }
    }
}

impl<F> JitFunction<F>
where
    F: Fn(Vec<f64>) -> f64 + Send + Sync + 'static,
{
    /// Create a new JIT function with a native implementation
    pub fn new(name: impl Into<String>, native_fn: F) -> Self {
        Self {
            name: name.into(),
            native_fn: Arc::new(native_fn),
            input_type: "f64",
            output_type: "f64",
            #[cfg(feature = "jit")]
            jit_context: None,
        }
    }

    /// Set a custom name for this JIT function
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    #[cfg(feature = "jit")]
    /// Compile and set the JIT function
    pub fn with_jit(mut self) -> JitResult<Self> {
        let name = self.name.clone();
        match JitContext::compile(&name) {
            Ok(ctx) => {
                self.jit_context = Some(Arc::new(ctx));
                Ok(self)
            }
            Err(e) => Err(e),
        }
    }

    /// Get the function name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the input type name
    pub fn input_type(&self) -> &'static str {
        self.input_type
    }
    
    /// Get the output type name
    pub fn output_type(&self) -> &'static str {
        self.output_type
    }
}

impl<F> JitCompilable<Vec<f64>, f64> for JitFunction<F>
where
    F: Fn(Vec<f64>) -> f64 + Send + Sync,
{
    fn execute(&self, args: Vec<f64>) -> f64 {
        // Record execution time for benchmarking
        let start = std::time::Instant::now();
        
        #[cfg(feature = "jit")]
        {
            if let Some(ctx) = &self.jit_context {
                // In a real implementation, this would call the JIT-compiled function
                // For now, we'll just use the native implementation
                let result = (self.native_fn)(args);
                let duration = start.elapsed().as_nanos() as u64;
                
                // In a real implementation, record stats
                // stats.record_jit_execution(duration);
                
                return result;
            }
        }
        
        // Fall back to native implementation
        let result = (self.native_fn)(args);
        let duration = start.elapsed().as_nanos() as u64;
        
        // In a real implementation, record stats
        // stats.record_native_execution(duration);
        
        result
    }
}

/// JIT compilation context
#[cfg(feature = "jit")]
pub struct JitContext {
    /// Function name for debugging and caching
    name: String,
    
    // In a real implementation, this would contain
    // the JIT module, compilation context, etc.
    // For now, it's a placeholder
}

#[cfg(feature = "jit")]
impl JitContext {
    /// Compile a function by name
    pub fn compile(name: &str) -> JitResult<Self> {
        // In a real implementation, this would use Cranelift to compile
        // the function. For now, it's just a stub.
        
        use cranelift_codegen::settings;
        
        let _flags_builder = settings::builder();
        // Configure JIT settings here
        
        // Create the JIT context
        Ok(Self {
            name: name.to_string(),
        })
    }
}

/// Decorator-like function to create JIT functions (similar to @numba.jit in Python)
pub fn jit<F>(name: impl Into<String>, f: F) -> JitFunction<F>
where
    F: Fn(Vec<f64>) -> f64 + Send + Sync + 'static,
{
    let func = JitFunction::new(name, f);
    
    #[cfg(feature = "jit")]
    {
        // In a real implementation, we'd compile right away
        // For now, just return the function
        return func;
    }
    
    #[cfg(not(feature = "jit"))]
    {
        func
    }
}