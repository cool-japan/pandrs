use std::fmt::Debug;

use crate::core::error::Result;

// Re-export from legacy module for now
#[deprecated(
    since = "0.1.0",
    note = "Use new Series implementation in crate::series::base"
)]
pub use crate::series::Series as LegacySeries;

/// Series struct: 1-dimensional data structure
#[derive(Debug, Clone)]
pub struct Series<T>
where
    T: Debug + Clone,
{
    /// The values in the Series
    values: Vec<T>,
    /// The name of the Series
    name: Option<String>,
}

impl<T> Series<T>
where
    T: Debug + Clone,
{
    /// Create a new Series
    pub fn new(data: Vec<T>, name: Option<String>) -> Result<Self> {
        Ok(Self { values: data, name })
    }

    /// Get the length of the Series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the Series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get an element at a specific index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Get a reference to the values in the Series
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Convert Series to Vec
    pub fn to_vec(&self) -> Vec<T> {
        self.values.clone()
    }

    /// Get the name of the Series
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Set the name of the Series
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Create a new Series with the specified name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Convert to f64 values
    pub fn as_f64(&self) -> Result<Vec<f64>>
    where
        T: Into<f64> + Copy,
    {
        let mut result = Vec::with_capacity(self.values.len());
        for value in &self.values {
            result.push((*value).into());
        }
        Ok(result)
    }

    /// Convert to string series
    pub fn to_string_series(&self) -> Result<Series<String>>
    where
        T: std::fmt::Display,
    {
        let string_values: Vec<String> = self.values.iter().map(|v| v.to_string()).collect();
        Series::new(string_values, self.name.clone())
    }

    /// Shift values by `periods`, filling exposed slots with NA.
    ///
    /// Positive `periods` shifts values toward larger indices (inserts NA at the front);
    /// negative `periods` shifts toward smaller indices (inserts NA at the end).
    /// When `|periods|` is greater than or equal to the Series length, the result is
    /// all NA. Matches pandas' `Series.shift(periods)` semantics with the default
    /// `fill_value` of NA.
    pub fn shift(&self, periods: i64) -> Result<crate::series::NASeries<T>> {
        let len = self.values.len();
        let abs = periods.unsigned_abs().min(len as u64) as usize;
        let mut shifted: Vec<crate::na::NA<T>> = Vec::with_capacity(len);

        if periods >= 0 {
            for _ in 0..abs {
                shifted.push(crate::na::NA::NA);
            }
            for v in &self.values[..len - abs] {
                shifted.push(crate::na::NA::Value(v.clone()));
            }
        } else {
            for v in &self.values[abs..] {
                shifted.push(crate::na::NA::Value(v.clone()));
            }
            for _ in 0..abs {
                shifted.push(crate::na::NA::NA);
            }
        }

        crate::series::NASeries::new(shifted, self.name.clone())
    }
}

// Additional implementations for numeric types
//
// `i32`/`i64`/`f64` each get their own small `sum`/`mean`/`min`/`max` block
// below rather than one generic impl: the three types need genuinely
// different missing-value handling (`i32`/`i64` have no NA representation
// at all -- every stored element participates; `f64` uses `NaN` as this
// crate's missing-value sentinel for a plain, non-`NA`-wrapped series and
// so must skip it, matching pandas' `skipna=True` default), and mirroring
// the existing per-type style keeps that difference explicit at each call
// site instead of hidden behind a shared generic bound.
impl Series<i32> {
    /// Calculate the sum of the Series.
    ///
    /// Accumulates in `i64` so the result cannot silently wrap around for
    /// series whose true sum exceeds `i32::MAX`/`i32::MIN` (which a plain
    /// `i32` accumulation would either panic on in debug builds or wrap on
    /// in release builds). `i32` has no NA representation, so every stored
    /// element participates -- there is nothing to skip.
    pub fn sum(&self) -> i64 {
        self.values.iter().map(|&v| v as i64).sum()
    }

    /// Calculate the mean of the Series.
    ///
    /// `i32` has no NA representation, so the divisor is simply `len()`
    /// (every element is a real observation; there is no separate
    /// "non-NA count" to track for this type).
    pub fn mean(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(crate::core::error::Error::EmptySeries);
        }
        Ok(self.sum() as f64 / self.len() as f64)
    }

    /// Get the minimum value in the Series.
    pub fn min(&self) -> Result<i32> {
        self.values
            .iter()
            .min()
            .cloned()
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }

    /// Get the maximum value in the Series.
    pub fn max(&self) -> Result<i32> {
        self.values
            .iter()
            .max()
            .cloned()
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }
}

impl Series<i64> {
    /// Calculate the sum of the Series. `i64` has no NA representation, so
    /// every stored element participates. Unlike [`Series::<i32>::sum`],
    /// there is no wider standard integer type to promote to here; a sum
    /// that itself overflows `i64` follows Rust's normal integer-overflow
    /// semantics (panics in debug builds, wraps in release), same as any
    /// other bare `i64` addition.
    pub fn sum(&self) -> i64 {
        self.values.iter().sum()
    }

    /// Calculate the mean of the Series. `i64` has no NA representation,
    /// so the divisor is simply `len()`.
    pub fn mean(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(crate::core::error::Error::EmptySeries);
        }
        Ok(self.sum() as f64 / self.len() as f64)
    }

    /// Get the minimum value in the Series.
    pub fn min(&self) -> Result<i64> {
        self.values
            .iter()
            .min()
            .cloned()
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }

    /// Get the maximum value in the Series.
    pub fn max(&self) -> Result<i64> {
        self.values
            .iter()
            .max()
            .cloned()
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }
}

impl Series<f64> {
    /// Calculate the sum of the Series, skipping `NaN` (pandas
    /// `skipna=True` semantics -- `NaN` is this crate's missing-value
    /// sentinel for a plain, non-`NA`-wrapped float series, since
    /// `Series<T>` itself has no null bitmap). Returns `0.0` if the series
    /// is empty or every value is `NaN`, matching both
    /// [`Series::<i32>::sum`]'s empty-series-sums-to-0 convention and
    /// pandas' own `Series.sum()`, which also returns `0.0` for an empty or
    /// all-NaN series (sum of zero observations is the additive identity,
    /// not a fabricated stand-in for missing data).
    pub fn sum(&self) -> f64 {
        self.values.iter().copied().filter(|v| !v.is_nan()).sum()
    }

    /// Calculate the mean of the Series, skipping `NaN` (pandas
    /// `skipna=True`). The divisor is the count of non-`NaN` values, not
    /// `len()`. Returns `Err(EmptySeries)` when there are no non-`NaN`
    /// observations to average -- this covers both an empty series and a
    /// non-empty, all-`NaN` one; unlike `sum`'s additive-identity case,
    /// there is no honest numeric value to report for "the average of zero
    /// observations".
    pub fn mean(&self) -> Result<f64> {
        let (sum, count) = self
            .values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold((0.0_f64, 0usize), |(s, c), v| (s + v, c + 1));
        if count == 0 {
            return Err(crate::core::error::Error::EmptySeries);
        }
        Ok(sum / count as f64)
    }

    /// Get the minimum value in the Series, skipping `NaN`.
    /// `Err(EmptySeries)` when there are no non-`NaN` observations
    /// (empty series, or every value `NaN`).
    pub fn min(&self) -> Result<f64> {
        self.values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.min(v)))
            })
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }

    /// Get the maximum value in the Series, skipping `NaN`.
    /// `Err(EmptySeries)` when there are no non-`NaN` observations
    /// (empty series, or every value `NaN`).
    pub fn max(&self) -> Result<f64> {
        self.values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }
}

// String-specific Series implementation
impl Series<String> {
    /// Get string accessor for string operations
    pub fn str(&self) -> Result<crate::series::string_accessor::StringAccessor> {
        crate::series::string_accessor::StringAccessor::new(self.clone()).map_err(|e| {
            crate::core::error::Error::Type(format!("Failed to create string accessor: {:?}", e))
        })
    }
}

// DateTime-specific Series implementation
impl Series<chrono::NaiveDateTime> {
    /// Get datetime accessor for datetime operations
    pub fn dt(&self) -> Result<crate::series::datetime_accessor::DateTimeAccessor> {
        crate::series::datetime_accessor::DateTimeAccessor::new(self.clone()).map_err(|e| {
            crate::core::error::Error::Type(format!("Failed to create datetime accessor: {:?}", e))
        })
    }
}

// Timezone-aware DateTime Series implementation
impl Series<chrono::DateTime<chrono::Utc>> {
    /// Get timezone-aware datetime accessor for datetime operations
    pub fn dt_tz(&self) -> Result<crate::series::datetime_accessor::DateTimeAccessorTz> {
        crate::series::datetime_accessor::DateTimeAccessorTz::new(self.clone()).map_err(|e| {
            crate::core::error::Error::Type(format!("Failed to create datetime accessor: {:?}", e))
        })
    }
}

// --- Idiomatic Rust trait implementations ---

impl<T> std::ops::Index<usize> for Series<T>
where
    T: Debug + Clone,
{
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.values[i]
    }
}

impl<'a, T> IntoIterator for &'a Series<T>
where
    T: Debug + Clone,
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<T> FromIterator<T> for Series<T>
where
    T: Debug + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values: Vec<T> = iter.into_iter().collect();
        // `new` only fails on validation, which doesn't exist here; the
        // fallback to an empty Series is unreachable in practice.
        Series::new(values, None).unwrap_or_else(|_| Series {
            values: Vec::new(),
            name: None,
        })
    }
}

impl<T> Extend<T> for Series<T>
where
    T: Debug + Clone,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.values.extend(iter);
    }
}

impl<T> std::fmt::Display for Series<T>
where
    T: Debug + Clone + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self.name.as_deref().unwrap_or("unnamed");
        write!(f, "Series[{}](", name)?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_index_trait() {
        let s = Series::new(vec![10i64, 20, 30], Some("nums".to_string())).unwrap();
        assert_eq!(s[0], 10);
        assert_eq!(s[2], 30);
    }

    #[test]
    fn test_series_into_iterator() {
        let s = Series::new(vec![1.0f64, 2.0, 3.0], Some("vals".to_string())).unwrap();
        let sum: f64 = (&s).into_iter().copied().sum();
        assert!((sum - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_collect() {
        let s = Series::new(vec![1i64, 2, 3], Some("x".to_string())).unwrap();
        let doubled: Vec<i64> = (&s).into_iter().map(|&v| v * 2).collect();
        assert_eq!(doubled, vec![2, 4, 6]);
    }

    #[test]
    fn test_series_from_iterator() {
        let v = vec![4i64, 5, 6];
        let s: Series<i64> = v.into_iter().collect();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], 4);
    }

    #[test]
    fn test_series_extend() {
        let mut s = Series::new(vec![1i64, 2], None).unwrap();
        s.extend(vec![3i64, 4]);
        assert_eq!(s.len(), 4);
        assert_eq!(s[3], 4);
    }

    #[test]
    fn test_series_display() {
        let s = Series::new(vec![1i64, 2, 3], Some("nums".to_string())).unwrap();
        let disp = s.to_string();
        assert!(disp.contains("nums"), "Display missing name: {}", disp);
        assert!(disp.contains("1"), "Display missing value: {}", disp);
    }
}
