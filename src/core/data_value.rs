use std::fmt::{Debug, Display};
use std::sync::Arc;

use crate::core::error::{Error, Result};

/// Trait for data values that can be stored in a column
pub trait DataValue: Debug + Send + Sync {
    /// Returns the type name of the data value
    fn type_name(&self) -> &'static str;

    /// Converts the data value to a string representation
    fn to_string(&self) -> String;

    /// Creates a boxed clone of the data value
    fn clone_boxed(&self) -> Box<dyn DataValue>;

    /// Checks if the data value equals another data value
    fn equals(&self, other: &dyn DataValue) -> bool;

    /// Returns self as an Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

// Implementation for common types
impl DataValue for i64 {
    fn type_name(&self) -> &'static str {
        "i64"
    }

    fn to_string(&self) -> String {
        format!("{}", self)
    }

    fn clone_boxed(&self) -> Box<dyn DataValue> {
        Box::new(*self)
    }

    fn equals(&self, other: &dyn DataValue) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<i64>() {
            return self == other;
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl DataValue for f64 {
    fn type_name(&self) -> &'static str {
        "f64"
    }

    fn to_string(&self) -> String {
        format!("{}", self)
    }

    fn clone_boxed(&self) -> Box<dyn DataValue> {
        Box::new(*self)
    }

    fn equals(&self, other: &dyn DataValue) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<f64>() {
            return (self - other).abs() < f64::EPSILON;
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl DataValue for bool {
    fn type_name(&self) -> &'static str {
        "bool"
    }

    fn to_string(&self) -> String {
        format!("{}", self)
    }

    fn clone_boxed(&self) -> Box<dyn DataValue> {
        Box::new(*self)
    }

    fn equals(&self, other: &dyn DataValue) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<bool>() {
            return self == other;
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl DataValue for String {
    fn type_name(&self) -> &'static str {
        "String"
    }

    fn to_string(&self) -> String {
        self.clone()
    }

    fn clone_boxed(&self) -> Box<dyn DataValue> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn DataValue) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<String>() {
            return self == other;
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl DataValue for Arc<str> {
    fn type_name(&self) -> &'static str {
        "Arc<str>"
    }

    fn to_string(&self) -> String {
        format!("{}", self)
    }

    fn clone_boxed(&self) -> Box<dyn DataValue> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn DataValue) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Arc<str>>() {
            return self == other;
        }
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Add Any for downcasting
use std::any::Any;

/// Extension trait for DataValue that provides additional functionality
/// Note: The as_any method is now part of the DataValue trait
pub trait DataValueExt: DataValue {}

// Blanket implementation for all types implementing DataValue
impl<T: DataValue + 'static> DataValueExt for T {}

// DisplayExt trait for displaying values
pub trait DisplayExt: Display {
    /// Displays the value with a specified format
    fn display_with_format(&self, format: &str) -> String;
}

// Default implementation for all Display types
impl<T: Display> DisplayExt for T {
    fn display_with_format(&self, format: &str) -> String {
        match format {
            "" => format!("{}", self),
            _ => format!("{}", self), // In a real implementation, this would handle format strings
        }
    }
}
