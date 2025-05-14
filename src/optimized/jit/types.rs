//! Type system for JIT compilation
//!
//! This module provides abstractions for handling different data types
//! in JIT-compiled functions, allowing for type-parameterized operations.

use std::fmt;
use std::any::Any;
use std::convert::TryFrom;

/// A trait for numeric types that can be used in JIT-compiled functions
pub trait JitNumeric: Copy + Send + Sync + 'static + fmt::Debug {
    /// Convert to f64 for universal operations
    fn to_f64(&self) -> f64;
    
    /// Convert from f64 back to native type
    fn from_f64(value: f64) -> Self;
    
    /// Return the zero value of this type
    fn zero() -> Self;
    
    /// Return the one value of this type
    fn one() -> Self;
    
    /// Return the minimum value of this type
    fn min_value() -> Self;
    
    /// Return the maximum value of this type
    fn max_value() -> Self;
    
    /// Add two values of this type
    fn add(&self, other: &Self) -> Self;
    
    /// Subtract two values of this type
    fn sub(&self, other: &Self) -> Self;
    
    /// Multiply two values of this type
    fn mul(&self, other: &Self) -> Self;
    
    /// Divide two values of this type
    fn div(&self, other: &Self) -> Self;
    
    /// Compute the square root (may convert to f64 internally)
    fn sqrt(&self) -> Self;
    
    /// Raise to a power (may convert to f64 internally)
    fn pow(&self, exp: i32) -> Self;
    
    /// Check if this value is NaN
    fn is_nan(&self) -> bool;
    
    /// Get the type name for debugging
    fn type_name() -> &'static str;
}

/// Implementation of JitNumeric for f64
impl JitNumeric for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }
    
    fn from_f64(value: f64) -> Self {
        value
    }
    
    fn zero() -> Self {
        0.0
    }
    
    fn one() -> Self {
        1.0
    }
    
    fn min_value() -> Self {
        f64::MIN
    }
    
    fn max_value() -> Self {
        f64::MAX
    }
    
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    
    fn sqrt(&self) -> Self {
        self.sqrt()
    }
    
    fn pow(&self, exp: i32) -> Self {
        self.powi(exp)
    }
    
    fn is_nan(&self) -> bool {
        self.is_nan()
    }
    
    fn type_name() -> &'static str {
        "f64"
    }
}

/// Implementation of JitNumeric for f32
impl JitNumeric for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
    
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    
    fn zero() -> Self {
        0.0
    }
    
    fn one() -> Self {
        1.0
    }
    
    fn min_value() -> Self {
        f32::MIN
    }
    
    fn max_value() -> Self {
        f32::MAX
    }
    
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    
    fn sqrt(&self) -> Self {
        self.sqrt()
    }
    
    fn pow(&self, exp: i32) -> Self {
        self.powi(exp)
    }
    
    fn is_nan(&self) -> bool {
        self.is_nan()
    }
    
    fn type_name() -> &'static str {
        "f32"
    }
}

/// Implementation of JitNumeric for i64
impl JitNumeric for i64 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
    
    fn from_f64(value: f64) -> Self {
        value as i64
    }
    
    fn zero() -> Self {
        0
    }
    
    fn one() -> Self {
        1
    }
    
    fn min_value() -> Self {
        i64::MIN
    }
    
    fn max_value() -> Self {
        i64::MAX
    }
    
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    
    fn sqrt(&self) -> Self {
        (*self as f64).sqrt() as i64
    }
    
    fn pow(&self, exp: i32) -> Self {
        if exp >= 0 {
            self.pow(exp as u32)
        } else {
            // Negative exponents for integers are handled via f64 and rounded
            (*self as f64).powi(exp) as i64
        }
    }
    
    fn is_nan(&self) -> bool {
        false // Integers don't have NaN
    }
    
    fn type_name() -> &'static str {
        "i64"
    }
}

/// Implementation of JitNumeric for i32
impl JitNumeric for i32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
    
    fn from_f64(value: f64) -> Self {
        value as i32
    }
    
    fn zero() -> Self {
        0
    }
    
    fn one() -> Self {
        1
    }
    
    fn min_value() -> Self {
        i32::MIN
    }
    
    fn max_value() -> Self {
        i32::MAX
    }
    
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    
    fn sqrt(&self) -> Self {
        (*self as f64).sqrt() as i32
    }
    
    fn pow(&self, exp: i32) -> Self {
        if exp >= 0 {
            self.pow(exp as u32)
        } else {
            // Negative exponents for integers are handled via f64 and rounded
            (*self as f64).powi(exp) as i32
        }
    }
    
    fn is_nan(&self) -> bool {
        false // Integers don't have NaN
    }
    
    fn type_name() -> &'static str {
        "i32"
    }
}

/// Type-erased numeric value for interoperability between different types
#[derive(Debug, Clone)]
pub enum NumericValue {
    /// 64-bit floating point
    F64(f64),
    /// 32-bit floating point
    F32(f32),
    /// 64-bit signed integer
    I64(i64),
    /// 32-bit signed integer
    I32(i32),
}

impl NumericValue {
    /// Convert to f64 for universal operations
    pub fn to_f64(&self) -> f64 {
        match self {
            NumericValue::F64(val) => *val,
            NumericValue::F32(val) => *val as f64,
            NumericValue::I64(val) => *val as f64,
            NumericValue::I32(val) => *val as f64,
        }
    }
    
    /// Get the type name for debugging
    pub fn type_name(&self) -> &'static str {
        match self {
            NumericValue::F64(_) => "f64",
            NumericValue::F32(_) => "f32",
            NumericValue::I64(_) => "i64",
            NumericValue::I32(_) => "i32",
        }
    }
}

impl From<f64> for NumericValue {
    fn from(val: f64) -> Self {
        NumericValue::F64(val)
    }
}

impl From<f32> for NumericValue {
    fn from(val: f32) -> Self {
        NumericValue::F32(val)
    }
}

impl From<i64> for NumericValue {
    fn from(val: i64) -> Self {
        NumericValue::I64(val)
    }
}

impl From<i32> for NumericValue {
    fn from(val: i32) -> Self {
        NumericValue::I32(val)
    }
}

/// Marker trait for types that can be used in JIT-compiled functions
pub trait JitType: Send + Sync + 'static {
    /// Type returned by the operation
    type Output;
    
    /// Convert to a type-erased value
    fn to_numeric_value(&self) -> Option<NumericValue>;
    
    /// Try to create this type from a type-erased value
    fn from_numeric_value(value: &NumericValue) -> Option<Self> where Self: Sized;
    
    /// Get the type name for debugging
    fn type_name() -> &'static str;
}

impl JitType for f64 {
    type Output = f64;
    
    fn to_numeric_value(&self) -> Option<NumericValue> {
        Some(NumericValue::F64(*self))
    }
    
    fn from_numeric_value(value: &NumericValue) -> Option<Self> {
        match value {
            NumericValue::F64(val) => Some(*val),
            NumericValue::F32(val) => Some(*val as f64),
            NumericValue::I64(val) => Some(*val as f64),
            NumericValue::I32(val) => Some(*val as f64),
        }
    }
    
    fn type_name() -> &'static str {
        "f64"
    }
}

impl JitType for f32 {
    type Output = f32;
    
    fn to_numeric_value(&self) -> Option<NumericValue> {
        Some(NumericValue::F32(*self))
    }
    
    fn from_numeric_value(value: &NumericValue) -> Option<Self> {
        match value {
            NumericValue::F64(val) => Some(*val as f32),
            NumericValue::F32(val) => Some(*val),
            NumericValue::I64(val) => Some(*val as f32),
            NumericValue::I32(val) => Some(*val as f32),
        }
    }
    
    fn type_name() -> &'static str {
        "f32"
    }
}

impl JitType for i64 {
    type Output = i64;
    
    fn to_numeric_value(&self) -> Option<NumericValue> {
        Some(NumericValue::I64(*self))
    }
    
    fn from_numeric_value(value: &NumericValue) -> Option<Self> {
        match value {
            NumericValue::F64(val) => Some(*val as i64),
            NumericValue::F32(val) => Some(*val as i64),
            NumericValue::I64(val) => Some(*val),
            NumericValue::I32(val) => Some(*val as i64),
        }
    }
    
    fn type_name() -> &'static str {
        "i64"
    }
}

impl JitType for i32 {
    type Output = i32;
    
    fn to_numeric_value(&self) -> Option<NumericValue> {
        Some(NumericValue::I32(*self))
    }
    
    fn from_numeric_value(value: &NumericValue) -> Option<Self> {
        match value {
            NumericValue::F64(val) => Some(*val as i32),
            NumericValue::F32(val) => Some(*val as i32),
            NumericValue::I64(val) => Some(*val as i32),
            NumericValue::I32(val) => Some(*val),
        }
    }
    
    fn type_name() -> &'static str {
        "i32"
    }
}

/// Type-safe vector for JIT operations
#[derive(Debug, Clone)]
pub enum TypedVector {
    /// 64-bit floating point vector
    F64(Vec<f64>),
    /// 32-bit floating point vector
    F32(Vec<f32>),
    /// 64-bit signed integer vector
    I64(Vec<i64>),
    /// 32-bit signed integer vector
    I32(Vec<i32>),
}

impl TypedVector {
    /// Create a new vector from values
    pub fn new<T: JitType + Clone>(values: Vec<T>) -> Option<Self> {
        // This would require specialization in a full implementation
        // For now, just a simple implementation for numeric types
        if T::type_name() == "f64" {
            if let Some(vals) = to_f64_vec(&values) {
                return Some(TypedVector::F64(vals));
            }
        } else if T::type_name() == "f32" {
            if let Some(vals) = to_f32_vec(&values) {
                return Some(TypedVector::F32(vals));
            }
        } else if T::type_name() == "i64" {
            if let Some(vals) = to_i64_vec(&values) {
                return Some(TypedVector::I64(vals));
            }
        } else if T::type_name() == "i32" {
            if let Some(vals) = to_i32_vec(&values) {
                return Some(TypedVector::I32(vals));
            }
        }
        
        None
    }
    
    /// Get the length of the vector
    pub fn len(&self) -> usize {
        match self {
            TypedVector::F64(vec) => vec.len(),
            TypedVector::F32(vec) => vec.len(),
            TypedVector::I64(vec) => vec.len(),
            TypedVector::I32(vec) => vec.len(),
        }
    }
    
    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the type name
    pub fn type_name(&self) -> &'static str {
        match self {
            TypedVector::F64(_) => "f64",
            TypedVector::F32(_) => "f32",
            TypedVector::I64(_) => "i64",
            TypedVector::I32(_) => "i32",
        }
    }
    
    /// Convert to f64 vector for universal operations
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            TypedVector::F64(vec) => vec.clone(),
            TypedVector::F32(vec) => vec.iter().map(|&x| x as f64).collect(),
            TypedVector::I64(vec) => vec.iter().map(|&x| x as f64).collect(),
            TypedVector::I32(vec) => vec.iter().map(|&x| x as f64).collect(),
        }
    }
}

// Helper functions for type conversion
fn to_f64_vec<T: Any>(values: &[T]) -> Option<Vec<f64>> {
    if let Some(vals) = downcast_slice::<T, f64>(values) {
        return Some(vals.to_vec());
    }
    
    None
}

fn to_f32_vec<T: Any>(values: &[T]) -> Option<Vec<f32>> {
    if let Some(vals) = downcast_slice::<T, f32>(values) {
        return Some(vals.to_vec());
    }
    
    None
}

fn to_i64_vec<T: Any>(values: &[T]) -> Option<Vec<i64>> {
    if let Some(vals) = downcast_slice::<T, i64>(values) {
        return Some(vals.to_vec());
    }
    
    None
}

fn to_i32_vec<T: Any>(values: &[T]) -> Option<Vec<i32>> {
    if let Some(vals) = downcast_slice::<T, i32>(values) {
        return Some(vals.to_vec());
    }
    
    None
}

// Attempt to downcast a slice of one type to another
fn downcast_slice<T: Any, U: 'static>(slice: &[T]) -> Option<&[U]> {
    // This is a simplistic implementation that would be more robust in a real codebase
    // It works only for numeric types with direct type matching
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>() {
        let ptr = slice as *const [T] as *const [U];
        unsafe { Some(&*ptr) }
    } else {
        None
    }
}