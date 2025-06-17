//! Core JIT compilation infrastructure for pandrs
//!
//! This module provides the fundamental JIT compilation capabilities,
//! supporting fast execution of numerical operations on arrays of data.
//!
//! The JIT system supports multiple numeric types (f64, f32, i64, i32) and
//! provides a flexible, type-safe interface for creating JIT-compiled functions.

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use super::types::{JitNumeric, JitType, NumericValue, TypedVector};

// JIT compilation imports
#[cfg(feature = "jit")]
use cranelift::prelude::*;
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::JITModule;
#[cfg(feature = "jit")]
use cranelift_module::Module;

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
pub struct JitFunction {
    /// Function name for debugging and caching
    name: String,
    /// Native function to use when JIT is disabled
    native_fn: Arc<FloatArrayFn>,
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

impl JitFunction {
    /// Create a new JIT function with a native implementation
    pub fn new<F>(name: impl Into<String>, native_fn: F) -> Self
    where
        F: Fn(Vec<f64>) -> f64 + Send + Sync + 'static,
    {
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

impl JitCompilable<Vec<f64>, f64> for JitFunction {
    fn execute(&self, args: Vec<f64>) -> f64 {
        // Record execution time for benchmarking
        let start = std::time::Instant::now();

        #[cfg(feature = "jit")]
        {
            if let Some(ctx) = &self.jit_context {
                // Use the JIT-compiled function for array operations
                match ctx.execute_array_sum(&args) {
                    Ok(result) => {
                        let duration = start.elapsed().as_nanos() as u64;
                        // In a real implementation, record JIT stats
                        // stats.record_jit_execution(duration);
                        return result;
                    }
                    Err(_) => {
                        // Fall back to native implementation on JIT error
                        // In production, you might want to log this error
                    }
                }
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
    /// The compiled function pointer
    compiled_fn: Option<*const u8>,
    /// JIT module for function management
    #[cfg(feature = "jit")]
    jit_module: Option<cranelift_jit::JITModule>,
}

#[cfg(feature = "jit")]
impl JitContext {
    /// Compile a function by name
    pub fn compile(name: &str) -> JitResult<Self> {
        use cranelift_jit::{JITBuilder, JITModule};
        use cranelift_module::{Linkage, Module};
        use target_lexicon::Triple;

        // Create JIT builder with current target
        let isa = cranelift_native::builder()
            .map_err(|e| {
                JitError::CompilationError(format!("Failed to create ISA builder: {}", e))
            })?
            .finish(settings::Flags::new(settings::builder()))
            .map_err(|e| JitError::CompilationError(format!("Failed to finish ISA: {}", e)))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Create JIT module
        let mut module = JITModule::new(builder);

        // Define function signature for array sum operation
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // array pointer
        sig.params.push(AbiParam::new(types::I64)); // array length
        sig.returns.push(AbiParam::new(types::F64)); // result

        // Create function declaration
        let func_id = module
            .declare_function(name, Linkage::Export, &sig)
            .map_err(|e| {
                JitError::CompilationError(format!("Function declaration failed: {}", e))
            })?;

        // Define function body
        let mut ctx = module.make_context();
        let mut builder_ctx = codegen::Context::new();
        builder_ctx.func.signature = sig.clone();

        // Build simple sum function
        {
            use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

            let mut func_ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut builder_ctx.func, &mut func_ctx);

            // Create entry block
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get function parameters
            let array_ptr = builder.block_params(entry_block)[0];
            let array_len = builder.block_params(entry_block)[1];

            // Initialize sum to 0.0
            let zero = builder.ins().f64const(0.0);
            let sum = Variable::new(0);
            builder.declare_var(sum, types::F64);
            builder.def_var(sum, zero);

            // Initialize loop counter
            let counter = Variable::new(1);
            builder.declare_var(counter, types::I64);
            let zero_i64 = builder.ins().iconst(types::I64, 0);
            builder.def_var(counter, zero_i64);

            // Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_end = builder.create_block();

            // Jump to loop header
            builder.ins().jump(loop_header, &[]);

            // Loop header: check condition
            builder.switch_to_block(loop_header);
            let current_counter = builder.use_var(counter);
            let condition = builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, current_counter, array_len);
            builder.ins().brif(condition, loop_body, &[], loop_end, &[]);

            // Loop body: add current element to sum
            builder.switch_to_block(loop_body);
            let current_counter = builder.use_var(counter);
            let element_offset = builder.ins().imul_imm(current_counter, 8); // 8 bytes per f64
            let element_ptr = builder.ins().iadd(array_ptr, element_offset);
            let element_value = builder
                .ins()
                .load(types::F64, MemFlags::new(), element_ptr, 0);

            let current_sum = builder.use_var(sum);
            let new_sum = builder.ins().fadd(current_sum, element_value);
            builder.def_var(sum, new_sum);

            // Increment counter
            let one = builder.ins().iconst(types::I64, 1);
            let next_counter = builder.ins().iadd(current_counter, one);
            builder.def_var(counter, next_counter);

            // Jump back to loop header
            builder.ins().jump(loop_header, &[]);

            // Loop end: return sum
            builder.switch_to_block(loop_end);
            let final_sum = builder.use_var(sum);
            builder.ins().return_(&[final_sum]);

            // Seal remaining blocks
            builder.seal_block(loop_header);
            builder.seal_block(loop_body);
            builder.seal_block(loop_end);

            builder.finalize();
        }

        // Define the function
        ctx.func = builder_ctx.func;
        module.define_function(func_id, &mut ctx).map_err(|e| {
            JitError::CompilationError(format!("Function definition failed: {}", e))
        })?;

        // Finalize function
        module.finalize_definitions().map_err(|e| {
            JitError::CompilationError(format!("Failed to finalize definitions: {}", e))
        })?;

        // Get function pointer
        let compiled_fn = module.get_finalized_function(func_id);

        Ok(Self {
            name: name.to_string(),
            compiled_fn: Some(compiled_fn),
            jit_module: Some(module),
        })
    }

    /// Execute the compiled function with array data
    pub fn execute_array_sum(&self, data: &[f64]) -> JitResult<f64> {
        if let Some(func_ptr) = self.compiled_fn {
            // Cast function pointer to correct signature
            let func: unsafe extern "C" fn(*const f64, i64) -> f64 =
                unsafe { std::mem::transmute(func_ptr) };

            // Call the JIT-compiled function
            let result = unsafe { func(data.as_ptr(), data.len() as i64) };

            Ok(result)
        } else {
            Err(JitError::ExecutionError(
                "Function not compiled".to_string(),
            ))
        }
    }
}

/// Decorator-like function to create JIT functions (similar to @numba.jit in Python)
pub fn jit<F>(name: impl Into<String>, f: F) -> JitFunction
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
