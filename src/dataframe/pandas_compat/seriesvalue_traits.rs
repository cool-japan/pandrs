//! # SeriesValue - Trait Implementations
//!
//! This module contains trait implementations for `SeriesValue`.
//!
//! ## Implemented Traits
//!
//! - `From`
//! - `From`
//! - `From`
//! - `From`
//! - `From`
//! - `From`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SeriesValue;

impl From<i32> for SeriesValue {
    fn from(v: i32) -> Self {
        SeriesValue::Int(v as i64)
    }
}

impl From<i64> for SeriesValue {
    fn from(v: i64) -> Self {
        SeriesValue::Int(v)
    }
}

impl From<f64> for SeriesValue {
    fn from(v: f64) -> Self {
        SeriesValue::Float(v)
    }
}

impl From<String> for SeriesValue {
    fn from(v: String) -> Self {
        SeriesValue::String(v)
    }
}

impl From<&str> for SeriesValue {
    fn from(v: &str) -> Self {
        SeriesValue::String(v.to_string())
    }
}

impl From<bool> for SeriesValue {
    fn from(v: bool) -> Self {
        SeriesValue::Bool(v)
    }
}
