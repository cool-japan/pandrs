//! Generic JIT function implementations for various types
//!
//! This module provides specialized JIT functions for different numeric types,
//! allowing for type-parameterized operations while maintaining performance.

use std::sync::Arc;
use std::marker::PhantomData;

use super::jit_core::{JitCompilable, GenericJitCompilable, JitFunction, JitResult};
use super::types::{JitType, JitNumeric, TypedVector, NumericValue};

/// A type-parameterized JIT function
#[derive(Clone)]
pub struct GenericJitFunction<I, O>
where
    I: JitType,
    O: JitType,
{
    /// Inner JIT function implementation
    inner: Box<dyn GenericJitCompilable>,
    /// Phantom data for input type
    _input_type: PhantomData<I>,
    /// Phantom data for output type
    _output_type: PhantomData<O>,
}

impl<I, O> GenericJitFunction<I, O>
where
    I: JitType,
    O: JitType,
{
    /// Create a new generic JIT function
    pub fn new(inner: Box<dyn GenericJitCompilable>) -> Self {
        Self {
            inner,
            _input_type: PhantomData,
            _output_type: PhantomData,
        }
    }
    
    /// Execute the function with the given input
    pub fn execute(&self, input: Vec<I>) -> Option<O> {
        // Convert input to TypedVector
        let typed_input = TypedVector::new(input)?;
        
        // Execute the generic function
        let result = self.inner.execute_typed(typed_input);
        
        // Convert the result back to the output type
        O::from_numeric_value(&result)
    }
}

/// A JIT function with f64 input and output
#[derive(Clone)]
pub struct F64JitFunction {
    /// Inner JIT function
    inner: JitFunction<dyn Fn(Vec<f64>) -> f64 + Send + Sync>,
}

impl F64JitFunction {
    /// Create a new f64 JIT function
    pub fn new(name: impl Into<String>, f: impl Fn(Vec<f64>) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            inner: JitFunction::new(name, f),
        }
    }
    
    #[cfg(feature = "jit")]
    /// Add JIT compilation
    pub fn with_jit(self) -> JitResult<Self> {
        let inner = self.inner.with_jit()?;
        Ok(Self { inner })
    }
}

impl JitCompilable<Vec<f64>, f64> for F64JitFunction {
    fn execute(&self, args: Vec<f64>) -> f64 {
        self.inner.execute(args)
    }
}

impl GenericJitCompilable for F64JitFunction {
    fn execute_typed(&self, args: TypedVector) -> NumericValue {
        match args {
            TypedVector::F64(vec) => NumericValue::F64(self.inner.execute(vec)),
            // For other types, convert to f64 first
            _ => NumericValue::F64(self.inner.execute(args.to_f64_vec())),
        }
    }
    
    fn input_type_name(&self) -> &'static str {
        "f64"
    }
    
    fn output_type_name(&self) -> &'static str {
        "f64"
    }
}

/// A JIT function with f32 input and output
#[derive(Clone)]
pub struct F32JitFunction {
    /// Inner function
    f: Arc<dyn Fn(Vec<f32>) -> f32 + Send + Sync>,
    /// Function name
    name: String,
    /// JIT context
    #[cfg(feature = "jit")]
    jit_context: Option<Arc<super::jit_core::JitContext>>,
}

impl F32JitFunction {
    /// Create a new f32 JIT function
    pub fn new(name: impl Into<String>, f: impl Fn(Vec<f32>) -> f32 + Send + Sync + 'static) -> Self {
        Self {
            f: Arc::new(f),
            name: name.into(),
            #[cfg(feature = "jit")]
            jit_context: None,
        }
    }
    
    #[cfg(feature = "jit")]
    /// Add JIT compilation
    pub fn with_jit(mut self) -> JitResult<Self> {
        match super::jit_core::JitContext::compile(&self.name) {
            Ok(ctx) => {
                self.jit_context = Some(Arc::new(ctx));
                Ok(self)
            }
            Err(e) => Err(e),
        }
    }
}

impl JitCompilable<Vec<f32>, f32> for F32JitFunction {
    fn execute(&self, args: Vec<f32>) -> f32 {
        (self.f)(args)
    }
}

impl GenericJitCompilable for F32JitFunction {
    fn execute_typed(&self, args: TypedVector) -> NumericValue {
        match args {
            TypedVector::F32(vec) => NumericValue::F32(self.execute(vec)),
            TypedVector::F64(vec) => {
                // Convert f64 to f32
                let f32_vec = vec.into_iter().map(|x| x as f32).collect();
                NumericValue::F32(self.execute(f32_vec))
            }
            TypedVector::I64(vec) => {
                // Convert i64 to f32
                let f32_vec = vec.into_iter().map(|x| x as f32).collect();
                NumericValue::F32(self.execute(f32_vec))
            }
            TypedVector::I32(vec) => {
                // Convert i32 to f32
                let f32_vec = vec.into_iter().map(|x| x as f32).collect();
                NumericValue::F32(self.execute(f32_vec))
            }
        }
    }
    
    fn input_type_name(&self) -> &'static str {
        "f32"
    }
    
    fn output_type_name(&self) -> &'static str {
        "f32"
    }
}

/// A JIT function with i64 input and output
#[derive(Clone)]
pub struct I64JitFunction {
    /// Inner function
    f: Arc<dyn Fn(Vec<i64>) -> i64 + Send + Sync>,
    /// Function name
    name: String,
    /// JIT context
    #[cfg(feature = "jit")]
    jit_context: Option<Arc<super::jit_core::JitContext>>,
}

impl I64JitFunction {
    /// Create a new i64 JIT function
    pub fn new(name: impl Into<String>, f: impl Fn(Vec<i64>) -> i64 + Send + Sync + 'static) -> Self {
        Self {
            f: Arc::new(f),
            name: name.into(),
            #[cfg(feature = "jit")]
            jit_context: None,
        }
    }
    
    #[cfg(feature = "jit")]
    /// Add JIT compilation
    pub fn with_jit(mut self) -> JitResult<Self> {
        match super::jit_core::JitContext::compile(&self.name) {
            Ok(ctx) => {
                self.jit_context = Some(Arc::new(ctx));
                Ok(self)
            }
            Err(e) => Err(e),
        }
    }
}

impl JitCompilable<Vec<i64>, i64> for I64JitFunction {
    fn execute(&self, args: Vec<i64>) -> i64 {
        (self.f)(args)
    }
}

impl GenericJitCompilable for I64JitFunction {
    fn execute_typed(&self, args: TypedVector) -> NumericValue {
        match args {
            TypedVector::I64(vec) => NumericValue::I64(self.execute(vec)),
            TypedVector::I32(vec) => {
                // Convert i32 to i64
                let i64_vec = vec.into_iter().map(|x| x as i64).collect();
                NumericValue::I64(self.execute(i64_vec))
            }
            TypedVector::F64(vec) => {
                // Convert f64 to i64
                let i64_vec = vec.into_iter().map(|x| x as i64).collect();
                NumericValue::I64(self.execute(i64_vec))
            }
            TypedVector::F32(vec) => {
                // Convert f32 to i64
                let i64_vec = vec.into_iter().map(|x| x as i64).collect();
                NumericValue::I64(self.execute(i64_vec))
            }
        }
    }
    
    fn input_type_name(&self) -> &'static str {
        "i64"
    }
    
    fn output_type_name(&self) -> &'static str {
        "i64"
    }
}

/// A JIT function with i32 input and output
#[derive(Clone)]
pub struct I32JitFunction {
    /// Inner function
    f: Arc<dyn Fn(Vec<i32>) -> i32 + Send + Sync>,
    /// Function name
    name: String,
    /// JIT context
    #[cfg(feature = "jit")]
    jit_context: Option<Arc<super::jit_core::JitContext>>,
}

impl I32JitFunction {
    /// Create a new i32 JIT function
    pub fn new(name: impl Into<String>, f: impl Fn(Vec<i32>) -> i32 + Send + Sync + 'static) -> Self {
        Self {
            f: Arc::new(f),
            name: name.into(),
            #[cfg(feature = "jit")]
            jit_context: None,
        }
    }
    
    #[cfg(feature = "jit")]
    /// Add JIT compilation
    pub fn with_jit(mut self) -> JitResult<Self> {
        match super::jit_core::JitContext::compile(&self.name) {
            Ok(ctx) => {
                self.jit_context = Some(Arc::new(ctx));
                Ok(self)
            }
            Err(e) => Err(e),
        }
    }
}

impl JitCompilable<Vec<i32>, i32> for I32JitFunction {
    fn execute(&self, args: Vec<i32>) -> i32 {
        (self.f)(args)
    }
}

impl GenericJitCompilable for I32JitFunction {
    fn execute_typed(&self, args: TypedVector) -> NumericValue {
        match args {
            TypedVector::I32(vec) => NumericValue::I32(self.execute(vec)),
            TypedVector::I64(vec) => {
                // Convert i64 to i32
                let i32_vec = vec.into_iter().map(|x| x as i32).collect();
                NumericValue::I32(self.execute(i32_vec))
            }
            TypedVector::F64(vec) => {
                // Convert f64 to i32
                let i32_vec = vec.into_iter().map(|x| x as i32).collect();
                NumericValue::I32(self.execute(i32_vec))
            }
            TypedVector::F32(vec) => {
                // Convert f32 to i32
                let i32_vec = vec.into_iter().map(|x| x as i32).collect();
                NumericValue::I32(self.execute(i32_vec))
            }
        }
    }
    
    fn input_type_name(&self) -> &'static str {
        "i32"
    }
    
    fn output_type_name(&self) -> &'static str {
        "i32"
    }
}

// Helper functions to create type-specific JIT functions

/// Create a JIT function that operates on f64 values
pub fn jit_f64(name: impl Into<String>, f: impl Fn(Vec<f64>) -> f64 + Send + Sync + 'static) -> F64JitFunction {
    F64JitFunction::new(name, f)
}

/// Create a JIT function that operates on f32 values
pub fn jit_f32(name: impl Into<String>, f: impl Fn(Vec<f32>) -> f32 + Send + Sync + 'static) -> F32JitFunction {
    F32JitFunction::new(name, f)
}

/// Create a JIT function that operates on i64 values
pub fn jit_i64(name: impl Into<String>, f: impl Fn(Vec<i64>) -> i64 + Send + Sync + 'static) -> I64JitFunction {
    I64JitFunction::new(name, f)
}

/// Create a JIT function that operates on i32 values
pub fn jit_i32(name: impl Into<String>, f: impl Fn(Vec<i32>) -> i32 + Send + Sync + 'static) -> I32JitFunction {
    I32JitFunction::new(name, f)
}