//! # SIMD-Enhanced Column Operations
//!
//! This module provides SIMD-accelerated operations for column types,
//! extending the basic column functionality with high-performance
//! vectorized operations for numeric data.

use crate::column::{ColumnTrait, Float64Column, Int64Column};
use crate::core::error::{Error, Result};
use crate::optimized::jit::simd_column_ops::{
    simd_abs_f64, simd_abs_i64, simd_add_f64, simd_add_i64, simd_add_scalar_f64,
    simd_add_scalar_i64, simd_compare_f64, simd_compare_i64, simd_divide_f64, simd_multiply_f64,
    simd_multiply_i64, simd_multiply_scalar_f64, simd_sqrt_f64, simd_subtract_f64,
    simd_subtract_i64, ComparisonOp,
};

/// Trait for SIMD-enhanced column operations on Float64 columns
pub trait SIMDFloat64Ops {
    /// Element-wise addition with another Float64Column using SIMD
    fn simd_add(&self, other: &Float64Column) -> Result<Float64Column>;

    /// Element-wise subtraction with another Float64Column using SIMD
    fn simd_subtract(&self, other: &Float64Column) -> Result<Float64Column>;

    /// Element-wise multiplication with another Float64Column using SIMD
    fn simd_multiply(&self, other: &Float64Column) -> Result<Float64Column>;

    /// Element-wise division with another Float64Column using SIMD
    fn simd_divide(&self, other: &Float64Column) -> Result<Float64Column>;

    /// Scalar addition using SIMD
    fn simd_add_scalar(&self, scalar: f64) -> Result<Float64Column>;

    /// Scalar multiplication using SIMD
    fn simd_multiply_scalar(&self, scalar: f64) -> Result<Float64Column>;

    /// Absolute value using SIMD
    fn simd_abs(&self) -> Result<Float64Column>;

    /// Square root using SIMD
    fn simd_sqrt(&self) -> Result<Float64Column>;

    /// Element-wise comparison using SIMD
    fn simd_compare(&self, other: &Float64Column, op: ComparisonOp) -> Result<Vec<bool>>;

    /// Scalar comparison using SIMD
    fn simd_compare_scalar(&self, scalar: f64, op: ComparisonOp) -> Result<Vec<bool>>;
}

/// Trait for SIMD-enhanced column operations on Int64 columns
pub trait SIMDInt64Ops {
    /// Element-wise addition with another Int64Column using SIMD
    fn simd_add(&self, other: &Int64Column) -> Result<Int64Column>;

    /// Element-wise subtraction with another Int64Column using SIMD
    fn simd_subtract(&self, other: &Int64Column) -> Result<Int64Column>;

    /// Element-wise multiplication with another Int64Column using SIMD
    fn simd_multiply(&self, other: &Int64Column) -> Result<Int64Column>;

    /// Scalar addition using SIMD
    fn simd_add_scalar(&self, scalar: i64) -> Result<Int64Column>;

    /// Absolute value using SIMD
    fn simd_abs(&self) -> Result<Int64Column>;

    /// Element-wise comparison using SIMD
    fn simd_compare(&self, other: &Int64Column, op: ComparisonOp) -> Result<Vec<bool>>;

    /// Scalar comparison using SIMD
    fn simd_compare_scalar(&self, scalar: i64, op: ComparisonOp) -> Result<Vec<bool>>;

    /// Convert to Float64Column for mixed operations
    fn to_float64_simd(&self) -> Result<Float64Column>;
}

impl SIMDFloat64Ops for Float64Column {
    fn simd_add(&self, other: &Float64Column) -> Result<Float64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0.0; self.len()];
        simd_add_f64(&self.data, &other.data, &mut result_data)?;

        // Combine null masks if they exist
        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_subtract(&self, other: &Float64Column) -> Result<Float64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0.0; self.len()];
        simd_subtract_f64(&self.data, &other.data, &mut result_data)?;

        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_multiply(&self, other: &Float64Column) -> Result<Float64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0.0; self.len()];
        simd_multiply_f64(&self.data, &other.data, &mut result_data)?;

        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_divide(&self, other: &Float64Column) -> Result<Float64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0.0; self.len()];
        simd_divide_f64(&self.data, &other.data, &mut result_data)?;

        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_add_scalar(&self, scalar: f64) -> Result<Float64Column> {
        let mut result_data = vec![0.0; self.len()];
        simd_add_scalar_f64(&self.data, scalar, &mut result_data)?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }

    fn simd_multiply_scalar(&self, scalar: f64) -> Result<Float64Column> {
        let mut result_data = vec![0.0; self.len()];
        simd_multiply_scalar_f64(&self.data, scalar, &mut result_data)?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }

    fn simd_abs(&self) -> Result<Float64Column> {
        let mut result_data = vec![0.0; self.len()];
        simd_abs_f64(&self.data, &mut result_data)?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }

    fn simd_sqrt(&self) -> Result<Float64Column> {
        let mut result_data = vec![0.0; self.len()];
        simd_sqrt_f64(&self.data, &mut result_data)?;

        Ok(Float64Column {
            data: result_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }

    fn simd_compare(&self, other: &Float64Column, op: ComparisonOp) -> Result<Vec<bool>> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result = vec![false; self.len()];
        simd_compare_f64(&self.data, &other.data, op, &mut result)?;

        // Handle null values - comparisons with nulls should be false
        apply_null_mask_to_comparison(&mut result, &self.null_mask, &other.null_mask);

        Ok(result)
    }

    fn simd_compare_scalar(&self, scalar: f64, op: ComparisonOp) -> Result<Vec<bool>> {
        let scalar_vec = vec![scalar; self.len()];
        let mut result = vec![false; self.len()];
        simd_compare_f64(&self.data, &scalar_vec, op, &mut result)?;

        // Handle null values
        apply_null_mask_to_comparison(&mut result, &self.null_mask, &None);

        Ok(result)
    }
}

impl SIMDInt64Ops for Int64Column {
    fn simd_add(&self, other: &Int64Column) -> Result<Int64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0i64; self.len()];
        simd_add_i64(&self.data, &other.data, &mut result_data)?;

        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Int64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_subtract(&self, other: &Int64Column) -> Result<Int64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0i64; self.len()];
        simd_subtract_i64(&self.data, &other.data, &mut result_data)?;

        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Int64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_multiply(&self, other: &Int64Column) -> Result<Int64Column> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result_data = vec![0i64; self.len()];
        simd_multiply_i64(&self.data, &other.data, &mut result_data)?;

        let null_mask = combine_null_masks(&self.null_mask, &other.null_mask, self.len())?;

        Ok(Int64Column {
            data: result_data.into(),
            null_mask,
            name: None,
        })
    }

    fn simd_add_scalar(&self, scalar: i64) -> Result<Int64Column> {
        let mut result_data = vec![0i64; self.len()];
        simd_add_scalar_i64(&self.data, scalar, &mut result_data)?;

        Ok(Int64Column {
            data: result_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }

    fn simd_abs(&self) -> Result<Int64Column> {
        let mut result_data = vec![0i64; self.len()];
        simd_abs_i64(&self.data, &mut result_data)?;

        Ok(Int64Column {
            data: result_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }

    fn simd_compare(&self, other: &Int64Column, op: ComparisonOp) -> Result<Vec<bool>> {
        if self.len() != other.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.len(),
                found: other.len(),
            });
        }

        let mut result = vec![false; self.len()];
        simd_compare_i64(&self.data, &other.data, op, &mut result)?;

        apply_null_mask_to_comparison(&mut result, &self.null_mask, &other.null_mask);

        Ok(result)
    }

    fn simd_compare_scalar(&self, scalar: i64, op: ComparisonOp) -> Result<Vec<bool>> {
        let scalar_vec = vec![scalar; self.len()];
        let mut result = vec![false; self.len()];
        simd_compare_i64(&self.data, &scalar_vec, op, &mut result)?;

        apply_null_mask_to_comparison(&mut result, &self.null_mask, &None);

        Ok(result)
    }

    fn to_float64_simd(&self) -> Result<Float64Column> {
        let float_data: Vec<f64> = self.data.iter().map(|&x| x as f64).collect();

        Ok(Float64Column {
            data: float_data.into(),
            null_mask: self.null_mask.clone(),
            name: None,
        })
    }
}

/// Combine null masks from two columns using logical OR
fn combine_null_masks(
    mask1: &Option<std::sync::Arc<[u8]>>,
    mask2: &Option<std::sync::Arc<[u8]>>,
    len: usize,
) -> Result<Option<std::sync::Arc<[u8]>>> {
    match (mask1, mask2) {
        (Some(m1), Some(m2)) => {
            let bytes_needed = (len + 7) / 8;
            let mut combined = vec![0u8; bytes_needed];

            for i in 0..bytes_needed.min(m1.len()).min(m2.len()) {
                combined[i] = m1[i] | m2[i];
            }

            Ok(Some(combined.into()))
        }
        (Some(m), None) | (None, Some(m)) => Ok(Some(m.clone())),
        (None, None) => Ok(None),
    }
}

/// Apply null masks to comparison results (nulls should produce false)
fn apply_null_mask_to_comparison(
    result: &mut [bool],
    mask1: &Option<std::sync::Arc<[u8]>>,
    mask2: &Option<std::sync::Arc<[u8]>>,
) {
    // If either value is null, the comparison result should be false
    for i in 0..result.len() {
        let is_null = is_null_at_index(mask1, i) || is_null_at_index(mask2, i);
        if is_null {
            result[i] = false;
        }
    }
}

/// Check if a value is null at the given index
fn is_null_at_index(mask: &Option<std::sync::Arc<[u8]>>, index: usize) -> bool {
    if let Some(m) = mask {
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        if byte_idx < m.len() {
            (m[byte_idx] & (1 << bit_idx)) != 0
        } else {
            false
        }
    } else {
        false
    }
}

/// High-level SIMD operations for mixed column arithmetic
pub struct SIMDColumnArithmetic;

impl SIMDColumnArithmetic {
    /// Add two numeric columns with automatic type promotion
    pub fn add_columns(
        left: &crate::column::Column,
        right: &crate::column::Column,
    ) -> Result<crate::column::Column> {
        use crate::column::Column;

        match (left, right) {
            (Column::Float64(l), Column::Float64(r)) => Ok(Column::Float64(l.simd_add(r)?)),
            (Column::Int64(l), Column::Int64(r)) => Ok(Column::Int64(l.simd_add(r)?)),
            (Column::Float64(l), Column::Int64(r)) => {
                let r_float = r.to_float64_simd()?;
                Ok(Column::Float64(l.simd_add(&r_float)?))
            }
            (Column::Int64(l), Column::Float64(r)) => {
                let l_float = l.to_float64_simd()?;
                Ok(Column::Float64(l_float.simd_add(r)?))
            }
            _ => Err(Error::InvalidOperation(
                "SIMD addition not supported for these column types".to_string(),
            )),
        }
    }

    /// Multiply two numeric columns with automatic type promotion
    pub fn multiply_columns(
        left: &crate::column::Column,
        right: &crate::column::Column,
    ) -> Result<crate::column::Column> {
        use crate::column::Column;

        match (left, right) {
            (Column::Float64(l), Column::Float64(r)) => Ok(Column::Float64(l.simd_multiply(r)?)),
            (Column::Int64(l), Column::Int64(r)) => Ok(Column::Int64(l.simd_multiply(r)?)),
            (Column::Float64(l), Column::Int64(r)) => {
                let r_float = r.to_float64_simd()?;
                Ok(Column::Float64(l.simd_multiply(&r_float)?))
            }
            (Column::Int64(l), Column::Float64(r)) => {
                let l_float = l.to_float64_simd()?;
                Ok(Column::Float64(l_float.simd_multiply(r)?))
            }
            _ => Err(Error::InvalidOperation(
                "SIMD multiplication not supported for these column types".to_string(),
            )),
        }
    }

    /// Apply scalar operation to a column
    pub fn multiply_scalar(
        column: &crate::column::Column,
        scalar: f64,
    ) -> Result<crate::column::Column> {
        use crate::column::Column;

        match column {
            Column::Float64(col) => Ok(Column::Float64(col.simd_multiply_scalar(scalar)?)),
            Column::Int64(col) => {
                let float_col = col.to_float64_simd()?;
                Ok(Column::Float64(float_col.simd_multiply_scalar(scalar)?))
            }
            _ => Err(Error::InvalidOperation(
                "SIMD scalar multiplication not supported for this column type".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{Float64Column, Int64Column};

    #[test]
    fn test_simd_float64_add() {
        let col1 = Float64Column::new(vec![1.0, 2.0, 3.0, 4.0]);
        let col2 = Float64Column::new(vec![1.0, 1.0, 1.0, 1.0]);

        let result = col1.simd_add(&col2).unwrap();

        assert_eq!(result.data(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_simd_float64_scalar_multiply() {
        let col = Float64Column::new(vec![1.0, 2.0, 3.0, 4.0]);

        let result = col.simd_multiply_scalar(2.0).unwrap();

        assert_eq!(result.data(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_float64_abs() {
        let col = Float64Column::new(vec![-1.0, 2.0, -3.0, 4.0]);

        let result = col.simd_abs().unwrap();

        assert_eq!(result.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_simd_float64_compare() {
        let col1 = Float64Column::new(vec![1.0, 2.0, 3.0, 4.0]);
        let col2 = Float64Column::new(vec![1.0, 1.0, 4.0, 4.0]);

        let result = col1.simd_compare(&col2, ComparisonOp::GreaterThan).unwrap();

        assert_eq!(result, vec![false, true, false, false]);
    }

    #[test]
    fn test_simd_int64_add() {
        let col1 = Int64Column::new(vec![1, 2, 3, 4]);
        let col2 = Int64Column::new(vec![1, 1, 1, 1]);

        let result = col1.simd_add(&col2).unwrap();

        assert_eq!(result.data(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_simd_int64_to_float64() {
        let col = Int64Column::new(vec![1, 2, 3, 4]);

        let result = col.to_float64_simd().unwrap();

        assert_eq!(result.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mixed_column_arithmetic() {
        use crate::column::Column;

        let col1 = Column::Float64(Float64Column::new(vec![1.0, 2.0, 3.0]));
        let col2 = Column::Int64(Int64Column::new(vec![1, 2, 3]));

        let result = SIMDColumnArithmetic::add_columns(&col1, &col2).unwrap();

        if let Column::Float64(float_result) = result {
            assert_eq!(float_result.data(), vec![2.0, 4.0, 6.0]);
        } else {
            panic!("Expected Float64 column result");
        }
    }

    #[test]
    fn test_length_mismatch_error() {
        let col1 = Float64Column::new(vec![1.0, 2.0]);
        let col2 = Float64Column::new(vec![1.0]);

        let result = col1.simd_add(&col2);
        assert!(result.is_err());
    }
}
