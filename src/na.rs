use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

/// Type representing missing values (NA, Not Available)
///
/// In Rust, missing values are represented through the type system, so we define the NA type instead of using Option.
/// NA represents the absence of a value.
#[derive(Clone, Copy)]
pub enum NA<T> {
    /// Case when a value exists
    Value(T),
    /// Case when a value doesn't exist
    NA,
}

impl<T> NA<T> {
    /// Check if the value is missing
    pub fn is_na(&self) -> bool {
        match self {
            NA::Value(_) => false,
            NA::NA => true,
        }
    }

    /// Check if a value exists
    pub fn is_value(&self) -> bool {
        !self.is_na()
    }

    /// Get the value (if it exists)
    pub fn value(&self) -> Option<&T> {
        match self {
            NA::Value(v) => Some(v),
            NA::NA => None,
        }
    }

    /// Get the value (if it exists), or return a default value if it doesn't exist
    pub fn value_or<'a>(&'a self, default: &'a T) -> &'a T {
        match self {
            NA::Value(v) => v,
            NA::NA => default,
        }
    }

    /// Transform the value
    pub fn map<U, F>(&self, f: F) -> NA<U>
    where
        F: FnOnce(&T) -> U,
    {
        match self {
            NA::Value(v) => NA::Value(f(v)),
            NA::NA => NA::NA,
        }
    }
}

// From implementation: Automatic conversion from type T to NA<T>
impl<T> From<T> for NA<T> {
    fn from(value: T) -> Self {
        NA::Value(value)
    }
}

// From implementation: Automatic conversion from Option<T> to NA<T>
impl<T> From<Option<T>> for NA<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => NA::Value(v),
            None => NA::NA,
        }
    }
}

// Into implementation: Automatic conversion from NA<T> to Option<T>
impl<T> From<NA<T>> for Option<T> {
    fn from(na: NA<T>) -> Self {
        match na {
            NA::Value(v) => Some(v),
            NA::NA => None,
        }
    }
}

// Debug implementation
impl<T: Debug> Debug for NA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NA::Value(v) => write!(f, "{:?}", v),
            NA::NA => write!(f, "NA"),
        }
    }
}

// Display implementation
impl<T: Display> Display for NA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NA::Value(v) => write!(f, "{}", v),
            NA::NA => write!(f, "NA"),
        }
    }
}

// PartialEq implementation
impl<T: PartialEq> PartialEq for NA<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => a == b,
            (NA::NA, NA::NA) => true,
            _ => false,
        }
    }
}

// Eq implementation (when T: Eq)
impl<T: Eq> Eq for NA<T> {}

// PartialOrd implementation
impl<T: PartialOrd> PartialOrd for NA<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => a.partial_cmp(b),
            (NA::NA, NA::NA) => Some(Ordering::Equal),
            (NA::NA, _) => Some(Ordering::Less), // Define NA as always less than other values
            (_, NA::NA) => Some(Ordering::Greater),
        }
    }
}

// Ord implementation (when T: Ord)
impl<T: Ord> Ord for NA<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => a.cmp(b),
            (NA::NA, NA::NA) => Ordering::Equal,
            (NA::NA, _) => Ordering::Less,
            (_, NA::NA) => Ordering::Greater,
        }
    }
}

// Hash implementation
impl<T: Hash> Hash for NA<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            NA::Value(v) => {
                0.hash(state); // Tag value
                v.hash(state);
            }
            NA::NA => {
                1.hash(state); // Tag value
            }
        }
    }
}

// Implementation for numeric operations (Add)
impl<T: Add<Output = T>> Add for NA<T> {
    type Output = NA<T>;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => NA::Value(a + b),
            _ => NA::NA, // If either is NA, return NA
        }
    }
}

// Implementation for numeric operations (Sub)
impl<T: Sub<Output = T>> Sub for NA<T> {
    type Output = NA<T>;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => NA::Value(a - b),
            _ => NA::NA,
        }
    }
}

// Implementation for numeric operations (Mul)
impl<T: Mul<Output = T>> Mul for NA<T> {
    type Output = NA<T>;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => NA::Value(a * b),
            _ => NA::NA,
        }
    }
}

// Implementation for numeric operations (Div)
impl<T: Div<Output = T> + std::cmp::PartialEq + NumericCast> Div for NA<T> {
    type Output = NA<T>;

    fn div(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(_), NA::Value(b)) if b == T::from(0) => NA::NA, // Division by zero returns NA
            (NA::Value(a), NA::Value(b)) => NA::Value(a / b),
            _ => NA::NA,
        }
    }
}

// Helper trait for type conversion (used for T::from(0))
trait NumericCast {
    fn from(val: i32) -> Self;
}

// Implementation for basic types like i32, f64, etc.
macro_rules! impl_numeric_cast {
    ($($t:ty),*) => {
        $(
            impl NumericCast for $t {
                fn from(val: i32) -> Self {
                    val as $t
                }
            }
        )*
    };
}

// Implementation for numeric types
impl_numeric_cast!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
