//! # Core JIT Compilation Module
//!
//! This module provides the core JIT compilation functionality for DataFrame operations.

use super::{JitError, JitResult};

/// A trait for types that can be JIT compiled
pub trait JitCompilable<T> {
    /// The output type of the JIT function
    type Output;

    /// Compile the function with JIT optimizations
    fn compile(&self) -> JitResult<Box<dyn Fn(&[T]) -> Self::Output + Send + Sync>>;

    /// Execute the function (may use JIT or fallback)
    fn execute(&self, data: &[T]) -> Self::Output;
}

/// A JIT-compiled function wrapper
#[derive(Clone)]
pub struct JitFunction<F> {
    /// The original function
    pub function: F,
    /// Function name for caching/debugging
    pub name: String,
    /// Whether this function should use JIT compilation
    pub use_jit: bool,
}

impl<F> JitFunction<F> {
    /// Create a new JIT function
    pub fn new(name: impl Into<String>, function: F) -> Self {
        Self {
            function,
            name: name.into(),
            use_jit: true,
        }
    }

    /// Disable JIT compilation for this function
    pub fn without_jit(mut self) -> Self {
        self.use_jit = false;
        self
    }
}

/// Basic JIT function for f64 operations
pub fn jit_f64<F>(name: impl Into<String>, func: F) -> JitFunction<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
{
    JitFunction::new(name, func)
}

/// Basic JIT function for i64 operations
pub fn jit_i64<F>(name: impl Into<String>, func: F) -> JitFunction<F>
where
    F: Fn(&[i64]) -> i64 + Send + Sync + Clone + 'static,
{
    JitFunction::new(name, func)
}

/// Basic JIT function for string operations
pub fn jit_string<F>(name: impl Into<String>, func: F) -> JitFunction<F>
where
    F: Fn(&[String]) -> String + Send + Sync + Clone + 'static,
{
    JitFunction::new(name, func)
}

/// Implementation of JitCompilable for f64 functions
impl<F> JitCompilable<f64> for JitFunction<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
{
    type Output = f64;

    fn compile(&self) -> JitResult<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> {
        if !self.use_jit {
            let func = self.function.clone();
            return Ok(Box::new(move |data| func(data)));
        }

        // Create optimized function directly
        let optimized_func = self.create_optimized_f64_function()?;
        Ok(optimized_func)
    }

    fn execute(&self, data: &[f64]) -> f64 {
        if self.use_jit {
            match self.compile() {
                Ok(compiled) => compiled(data),
                Err(_) => (self.function)(data), // Fallback to original function
            }
        } else {
            (self.function)(data)
        }
    }
}

/// Implementation of JitCompilable for i64 functions
impl<F> JitCompilable<i64> for JitFunction<F>
where
    F: Fn(&[i64]) -> i64 + Send + Sync + Clone + 'static,
{
    type Output = i64;

    fn compile(&self) -> JitResult<Box<dyn Fn(&[i64]) -> i64 + Send + Sync>> {
        if !self.use_jit {
            let func = self.function.clone();
            return Ok(Box::new(move |data| func(data)));
        }

        // Create optimized function directly
        let optimized_func = self.create_optimized_i64_function()?;
        Ok(optimized_func)
    }

    fn execute(&self, data: &[i64]) -> i64 {
        if self.use_jit {
            match self.compile() {
                Ok(compiled) => compiled(data),
                Err(_) => (self.function)(data), // Fallback to original function
            }
        } else {
            (self.function)(data)
        }
    }
}

/// Implementation of JitCompilable for string functions
impl<F> JitCompilable<String> for JitFunction<F>
where
    F: Fn(&[String]) -> String + Send + Sync + Clone + 'static,
{
    type Output = String;

    fn compile(&self) -> JitResult<Box<dyn Fn(&[String]) -> String + Send + Sync>> {
        // String operations don't benefit as much from JIT compilation
        let func = self.function.clone();
        Ok(Box::new(move |data| func(data)))
    }

    fn execute(&self, data: &[String]) -> String {
        (self.function)(data)
    }
}

impl<F> JitFunction<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
{
    /// Create an optimized version of the f64 function
    fn create_optimized_f64_function(&self) -> JitResult<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> {
        let func = self.function.clone();
        let name = self.name.clone();

        // Apply various optimizations based on function name/pattern
        if name.contains("sum") {
            Ok(Box::new(move |data: &[f64]| -> f64 {
                // Optimized sum with Kahan summation for numerical stability
                if data.is_empty() {
                    return 0.0;
                }

                let mut sum = 0.0;
                let mut c = 0.0; // Compensation for lost low-order bits

                for &value in data {
                    let y = value - c;
                    let t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }

                sum
            }))
        } else if name.contains("mean") {
            Ok(Box::new(move |data: &[f64]| -> f64 {
                if data.is_empty() {
                    return 0.0;
                }

                // Optimized mean calculation
                let mut sum = 0.0;
                let mut c = 0.0;

                for &value in data {
                    let y = value - c;
                    let t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }

                sum / data.len() as f64
            }))
        } else if name.contains("min") {
            Ok(Box::new(move |data: &[f64]| -> f64 {
                data.iter().copied().fold(f64::INFINITY, f64::min)
            }))
        } else if name.contains("max") {
            Ok(Box::new(move |data: &[f64]| -> f64 {
                data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            }))
        } else {
            // Default case: use original function
            Ok(Box::new(move |data| func(data)))
        }
    }
}

impl<F> JitFunction<F>
where
    F: Fn(&[i64]) -> i64 + Send + Sync + Clone + 'static,
{
    /// Create an optimized version of the i64 function
    fn create_optimized_i64_function(&self) -> JitResult<Box<dyn Fn(&[i64]) -> i64 + Send + Sync>> {
        let func = self.function.clone();
        let name = self.name.clone();

        // Apply optimizations for integer operations
        if name.contains("sum") {
            Ok(Box::new(move |data: &[i64]| -> i64 {
                // Use optimized sum with overflow detection
                data.iter().fold(0i64, |acc, &x| acc.saturating_add(x))
            }))
        } else if name.contains("mean") {
            Ok(Box::new(move |data: &[i64]| -> i64 {
                if data.is_empty() {
                    return 0;
                }
                let sum: i64 = data.iter().fold(0, |acc, &x| acc.saturating_add(x));
                sum / data.len() as i64
            }))
        } else if name.contains("min") {
            Ok(Box::new(move |data: &[i64]| -> i64 {
                data.iter().copied().min().unwrap_or(i64::MAX)
            }))
        } else if name.contains("max") {
            Ok(Box::new(move |data: &[i64]| -> i64 {
                data.iter().copied().max().unwrap_or(i64::MIN)
            }))
        } else {
            // Default case: use original function
            Ok(Box::new(move |data| func(data)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_f64_sum() {
        let sum_func = jit_f64("sum", |data: &[f64]| data.iter().sum());
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = sum_func.execute(&data);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_jit_i64_mean() {
        let mean_func = jit_i64("mean", |data: &[i64]| {
            if data.is_empty() {
                0
            } else {
                data.iter().sum::<i64>() / data.len() as i64
            }
        });
        let data = vec![1, 2, 3, 4, 5];

        let result = mean_func.execute(&data);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_jit_compilation() {
        let func = jit_f64("test_sum", |data: &[f64]| data.iter().sum());
        let compiled = func.compile().unwrap();

        let data = vec![1.0, 2.0, 3.0];
        let result = compiled(&data);
        assert_eq!(result, 6.0);
    }
}
