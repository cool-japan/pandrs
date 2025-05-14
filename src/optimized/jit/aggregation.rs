//! Common JIT-compilable aggregation functions

use std::sync::Arc;
use super::{JitCompilable, JitFunction, JitContext};

/// Create a JIT-compiled sum function
pub fn sum() -> JitFunction<&[f64], f64> {
    let native_fn = Arc::new(|values: &[f64]| -> f64 {
        values.iter().sum()
    });
    
    #[cfg(feature = "jit")]
    {
        JitFunction::new(native_fn).with_jit(|| {
            // JIT compilation would go here in a real implementation
            JitContext::new().map(Arc::new)
        })
    }
    
    #[cfg(not(feature = "jit"))]
    {
        JitFunction::new(native_fn)
    }
}

/// Create a JIT-compiled mean function
pub fn mean() -> JitFunction<&[f64], f64> {
    let native_fn = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    });
    
    #[cfg(feature = "jit")]
    {
        JitFunction::new(native_fn).with_jit(|| {
            JitContext::new().map(Arc::new)
        })
    }
    
    #[cfg(not(feature = "jit"))]
    {
        JitFunction::new(native_fn)
    }
}

/// Create a JIT-compiled standard deviation function
pub fn std() -> JitFunction<&[f64], f64> {
    let native_fn = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (*v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    });
    
    #[cfg(feature = "jit")]
    {
        JitFunction::new(native_fn).with_jit(|| {
            JitContext::new().map(Arc::new)
        })
    }
    
    #[cfg(not(feature = "jit"))]
    {
        JitFunction::new(native_fn)
    }
}

/// Create a JIT-compiled variance function
pub fn var() -> JitFunction<&[f64], f64> {
    let native_fn = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter()
            .map(|v| (*v - mean).powi(2))
            .sum::<f64>() / values.len() as f64
    });
    
    #[cfg(feature = "jit")]
    {
        JitFunction::new(native_fn).with_jit(|| {
            JitContext::new().map(Arc::new)
        })
    }
    
    #[cfg(not(feature = "jit"))]
    {
        JitFunction::new(native_fn)
    }
}

/// Create a JIT-compiled min function
pub fn min() -> JitFunction<&[f64], f64> {
    let native_fn = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut min = f64::INFINITY;
        for &val in values {
            min = min.min(val);
        }
        min
    });
    
    #[cfg(feature = "jit")]
    {
        JitFunction::new(native_fn).with_jit(|| {
            JitContext::new().map(Arc::new)
        })
    }
    
    #[cfg(not(feature = "jit"))]
    {
        JitFunction::new(native_fn)
    }
}

/// Create a JIT-compiled max function
pub fn max() -> JitFunction<&[f64], f64> {
    let native_fn = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut max = f64::NEG_INFINITY;
        for &val in values {
            max = max.max(val);
        }
        max
    });
    
    #[cfg(feature = "jit")]
    {
        JitFunction::new(native_fn).with_jit(|| {
            JitContext::new().map(Arc::new)
        })
    }
    
    #[cfg(not(feature = "jit"))]
    {
        JitFunction::new(native_fn)
    }
}