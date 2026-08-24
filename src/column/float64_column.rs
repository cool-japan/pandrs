use std::any::Any;
use std::sync::Arc;

use crate::core::column::{Column, ColumnTrait, ColumnType};
use crate::core::error::{Error, Result};

/// Structure representing a Float64 column
#[derive(Debug, Clone)]
pub struct Float64Column {
    pub(crate) data: Arc<[f64]>,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
}

impl Float64Column {
    /// Create a new Float64Column
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: None,
        }
    }

    /// Create a Float64Column with a name
    pub fn with_name(data: Vec<f64>, name: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: Some(name.into()),
        }
    }

    /// Create a Float64Column with NULL values
    pub fn with_nulls(data: Vec<f64>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };

        Self {
            data: data.into(),
            null_mask,
            name: None,
        }
    }

    /// Set the name
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get the name
    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get a reference to the underlying data (for testing and demonstration)
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get data at the specified index
    pub fn get(&self, index: usize) -> Result<Option<f64>> {
        if index >= self.data.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.data.len(),
            });
        }

        // Check for NULL value
        if let Some(ref mask) = self.null_mask {
            let byte_idx = index / 8;
            let bit_idx = index % 8;
            if byte_idx < mask.len() && (mask[byte_idx] & (1 << bit_idx)) != 0 {
                return Ok(None);
            }
        }

        Ok(Some(self.data[index]))
    }

    /// Calculate the sum of data, skipping NULLs and `NaN` (pandas
    /// `skipna=True` semantics -- matches [`crate::series::Series<f64>::sum`]).
    /// Returns `0.0` if there are no NULL/non-`NaN` values to sum (empty
    /// column, or every value NULL/`NaN`): the sum of zero observations is
    /// the additive identity, not a fabricated stand-in for missing data.
    ///
    /// The no-NULL fast path uses 4-way unrolled accumulators (F10): a
    /// plain `iter().sum()` with a `NaN` filter is a strictly serial
    /// dependency chain (each add depends on the previous), which prevents
    /// the compiler from overlapping the floating-point adds. Splitting
    /// into 4 independent lanes lets those adds execute with instruction-level
    /// parallelism and only combines the lanes once, at the end.
    pub fn sum(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        match &self.null_mask {
            None => sum_skipnan_unrolled(&self.data),
            Some(mask) => {
                let mut sum = 0.0;
                for i in 0..self.data.len() {
                    if is_null_at(mask, i) {
                        continue;
                    }
                    let v = self.data[i];
                    if !v.is_nan() {
                        sum += v;
                    }
                }
                sum
            }
        }
    }

    /// Calculate the mean (average) of data, skipping NULLs and `NaN`
    /// (pandas `skipna=True`). The divisor is the count of NULL-free,
    /// non-`NaN` values, not `len()`. Returns `None` when there are no such
    /// observations (empty column, or every value NULL/`NaN`) -- unlike
    /// `sum`'s additive-identity case, there is no honest numeric value for
    /// "the average of zero observations".
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        let (sum, count) = match &self.null_mask {
            None => sum_count_skipnan_unrolled(&self.data),
            Some(mask) => {
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..self.data.len() {
                    if is_null_at(mask, i) {
                        continue;
                    }
                    let v = self.data[i];
                    if !v.is_nan() {
                        sum += v;
                        count += 1;
                    }
                }
                (sum, count)
            }
        };

        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// Calculate the minimum value of data, skipping NULLs and `NaN`.
    ///
    /// Unlike the previous implementation, this does **not** exclude
    /// infinities: the filter is `!x.is_nan()`, not `x.is_finite()`, so
    /// `min([1.0, -inf, 2.0])` correctly reports `-inf` (pandas: `min` and
    /// `max` only ever skip missing values, never legitimate `+-inf`
    /// observations). Returns `None` when there are no NULL-free,
    /// non-`NaN` observations.
    pub fn min(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        match &self.null_mask {
            None => min_skipnan_unrolled(&self.data),
            Some(mask) => {
                let mut result = f64::INFINITY;
                let mut any_valid = false;
                for i in 0..self.data.len() {
                    if is_null_at(mask, i) {
                        continue;
                    }
                    let v = self.data[i];
                    any_valid |= !v.is_nan();
                    result = result.min(v);
                }
                if any_valid {
                    Some(result)
                } else {
                    None
                }
            }
        }
    }

    /// Calculate the maximum value of data, skipping NULLs and `NaN`.
    ///
    /// As with [`Self::min`], infinities are never excluded: `max([1.0,
    /// inf, 2.0])` correctly reports `inf`. Returns `None` when there are
    /// no NULL-free, non-`NaN` observations.
    pub fn max(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        match &self.null_mask {
            None => max_skipnan_unrolled(&self.data),
            Some(mask) => {
                let mut result = f64::NEG_INFINITY;
                let mut any_valid = false;
                for i in 0..self.data.len() {
                    if is_null_at(mask, i) {
                        continue;
                    }
                    let v = self.data[i];
                    any_valid |= !v.is_nan();
                    result = result.max(v);
                }
                if any_valid {
                    Some(result)
                } else {
                    None
                }
            }
        }
    }

    /// Create a new column by applying a mapping function
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let mapped_data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();

        Self {
            data: mapped_data.into(),
            null_mask: self.null_mask.clone(),
            name: self.name.clone(),
        }
    }

    /// Create a new column based on filtering conditions
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<f64>) -> bool,
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();

        for i in 0..self.data.len() {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or(f64::NAN));
                if has_nulls {
                    filtered_nulls.push(value.is_none());
                }
            }
        }

        if has_nulls {
            Self::with_nulls(filtered_data, filtered_nulls)
        } else {
            Self::new(filtered_data)
        }
    }
}

impl ColumnTrait for Float64Column {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Float64
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_column(&self) -> crate::core::column::Column {
        // Convert the legacy Column type to the core Column type
        let legacy_column = Column::Float64(self.clone());
        // This is a temporary workaround - in a complete solution,
        // we would implement proper conversion between column types
        crate::core::column::Column::from_any(Box::new(legacy_column))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Return whether `data[i]` is NULL under `mask`, using the same bit
/// layout as [`Float64Column::get`] (bit set == NULL).
#[inline]
fn is_null_at(mask: &[u8], i: usize) -> bool {
    let byte_idx = i / 8;
    let bit_idx = i % 8;
    byte_idx < mask.len() && (mask[byte_idx] & (1 << bit_idx)) != 0
}

/// Sum `data`, skipping `NaN` (pandas `skipna=True`), using 4-way unrolled
/// accumulators so the adds are independent of each other and can execute
/// with instruction-level parallelism instead of forming one long serial
/// dependency chain (F10).
#[inline]
fn sum_skipnan_unrolled(data: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for (lane, &v) in acc.iter_mut().zip(chunk) {
            *lane += if v.is_nan() { 0.0 } else { v };
        }
    }

    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for &v in remainder {
        if !v.is_nan() {
            total += v;
        }
    }
    total
}

/// Sum and count `data` together (skipping `NaN`), 4-way unrolled. Shared
/// by [`Float64Column::mean`]'s no-NULL fast path.
#[inline]
fn sum_count_skipnan_unrolled(data: &[f64]) -> (f64, usize) {
    let mut acc = [0.0f64; 4];
    let mut cnt = [0usize; 4];
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for lane in 0..4 {
            let v = chunk[lane];
            let valid = !v.is_nan();
            acc[lane] += if valid { v } else { 0.0 };
            cnt[lane] += valid as usize;
        }
    }

    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let mut count = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3]);
    for &v in remainder {
        let valid = !v.is_nan();
        total += if valid { v } else { 0.0 };
        count += valid as usize;
    }
    (total, count)
}

/// Minimum of `data`, skipping `NaN` but keeping `+-inf`, 4-way unrolled.
///
/// Lanes are seeded with `INFINITY` (the identity element for `min`) and
/// folded with `f64::min`, which -- per its documented behavior -- returns
/// the non-`NaN` operand whenever exactly one side is `NaN`. That means a
/// lane can never become `NaN` from this recurrence, so no per-element
/// branch is needed to skip `NaN`; validity is instead tracked separately
/// via a branchless OR-accumulate, purely to distinguish "every value was
/// NaN" (-> `None`) from "the real minimum is `INFINITY`" (-> `Some(inf)`,
/// which IS a legitimate result per this column's NaN-only skip contract).
#[inline]
fn min_skipnan_unrolled(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut lanes = [f64::INFINITY; 4];
    let mut any_valid = false;
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for (lane, &v) in lanes.iter_mut().zip(chunk) {
            any_valid |= !v.is_nan();
            *lane = lane.min(v);
        }
    }

    let mut result = lanes[0].min(lanes[1]).min(lanes[2]).min(lanes[3]);
    for &v in remainder {
        any_valid |= !v.is_nan();
        result = result.min(v);
    }

    if any_valid {
        Some(result)
    } else {
        None
    }
}

/// Maximum of `data`, skipping `NaN` but keeping `+-inf`, 4-way unrolled.
/// Mirror image of [`min_skipnan_unrolled`]; see its docs for the lane/NaN
/// reasoning.
#[inline]
fn max_skipnan_unrolled(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut lanes = [f64::NEG_INFINITY; 4];
    let mut any_valid = false;
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for (lane, &v) in lanes.iter_mut().zip(chunk) {
            any_valid |= !v.is_nan();
            *lane = lane.max(v);
        }
    }

    let mut result = lanes[0].max(lanes[1]).max(lanes[2]).max(lanes[3]);
    for &v in remainder {
        any_valid |= !v.is_nan();
        result = result.max(v);
    }

    if any_valid {
        Some(result)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exhaustively cover lengths that land NaN in every lane position of
    // both a full unrolled chunk and the scalar remainder, for every
    // aggregate -- this is exactly where a lane-fold bug would hide.
    const LENGTHS: [usize; 8] = [0, 1, 3, 4, 5, 7, 8, 9];

    #[test]
    fn sum_and_mean_skip_nan_at_every_lane_and_remainder_position() {
        for &len in &LENGTHS {
            for nan_pos in 0..len {
                let mut data = vec![1.0f64; len];
                data[nan_pos] = f64::NAN;
                let col = Float64Column::new(data);

                let expected_sum = (len - 1) as f64; // every value is 1.0 except one NaN
                assert_eq!(
                    col.sum(),
                    expected_sum,
                    "len={len} nan_pos={nan_pos}: sum must skip NaN"
                );

                if len > 1 {
                    assert_eq!(
                        col.mean(),
                        Some(1.0),
                        "len={len} nan_pos={nan_pos}: mean must skip NaN in both sum and count"
                    );
                } else {
                    assert_eq!(col.mean(), None, "all-NaN column has no mean");
                }
            }
        }
    }

    #[test]
    fn all_nan_sums_to_zero_and_has_no_mean_min_max() {
        for &len in &[1, 3, 4, 5, 8] {
            let col = Float64Column::new(vec![f64::NAN; len]);
            assert_eq!(col.sum(), 0.0);
            assert_eq!(col.mean(), None);
            assert_eq!(col.min(), None);
            assert_eq!(col.max(), None);
        }
    }

    #[test]
    fn min_max_skip_nan_but_keep_infinity_at_every_position() {
        for &len in &LENGTHS {
            if len == 0 {
                continue;
            }
            for nan_pos in 0..len {
                let mut data: Vec<f64> = (0..len).map(|i| i as f64).collect();
                data[nan_pos] = f64::NAN;
                let col = Float64Column::new(data);

                // `len == 1` means the single element IS the NaN at
                // `nan_pos` -- zero valid observations remain, so the
                // correct result is `None`, not the fold identity
                // (`fold` over an empty iterator returns its seed
                // unchanged, which would otherwise be indistinguishable
                // from a genuine result equal to that seed).
                if len == 1 {
                    assert_eq!(col.min(), None, "len=1 nan_pos={nan_pos}: no valid values");
                    assert_eq!(col.max(), None, "len=1 nan_pos={nan_pos}: no valid values");
                    continue;
                }

                let expected_min = (0..len)
                    .filter(|&i| i != nan_pos)
                    .map(|i| i as f64)
                    .fold(f64::INFINITY, f64::min);
                let expected_max = (0..len)
                    .filter(|&i| i != nan_pos)
                    .map(|i| i as f64)
                    .fold(f64::NEG_INFINITY, f64::max);

                assert_eq!(col.min(), Some(expected_min), "len={len} nan_pos={nan_pos}");
                assert_eq!(col.max(), Some(expected_max), "len={len} nan_pos={nan_pos}");
            }
        }
    }

    #[test]
    fn min_max_never_exclude_infinity() {
        let col = Float64Column::new(vec![1.0, f64::INFINITY, 2.0]);
        assert_eq!(col.max(), Some(f64::INFINITY), "max must not skip +inf");
        assert_eq!(col.min(), Some(1.0));

        let col = Float64Column::new(vec![1.0, f64::NEG_INFINITY, 2.0]);
        assert_eq!(col.min(), Some(f64::NEG_INFINITY), "min must not skip -inf");
        assert_eq!(col.max(), Some(2.0));

        // NaN alongside infinity: NaN is skipped, infinity is not.
        let col = Float64Column::new(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(col.max(), Some(f64::INFINITY));
        assert_eq!(col.min(), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn null_mask_path_also_skips_nan_and_keeps_infinity() {
        // NaN *and* NULL in the same column: NULLs come from the mask,
        // NaN from the value itself -- both must be skipped, and a
        // non-NULL infinity must still count.
        let data = vec![1.0, f64::NAN, f64::INFINITY, 5.0, 2.0];
        let nulls = vec![false, false, false, true, false]; // index 3 (5.0) is NULL
        let col = Float64Column::with_nulls(data, nulls);

        // Surviving, non-NULL, non-NaN values: 1.0, inf, 2.0
        assert_eq!(col.sum(), f64::INFINITY);
        assert_eq!(col.max(), Some(f64::INFINITY));
        assert_eq!(col.min(), Some(1.0));
        assert_eq!(col.mean(), Some(f64::INFINITY));
    }

    #[test]
    fn empty_column_aggregates() {
        let col = Float64Column::new(vec![]);
        assert_eq!(col.sum(), 0.0);
        assert_eq!(col.mean(), None);
        assert_eq!(col.min(), None);
        assert_eq!(col.max(), None);
    }
}
