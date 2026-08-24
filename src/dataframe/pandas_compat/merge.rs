//! Merge and join operations for DataFrames
//!
//! Provides pandas-compatible merge and join functionality.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::join::{join_with_suffixes, JoinType as CoreJoinType};

/// Join type for merge operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join - only matching rows from both DataFrames
    Inner,
    /// Left join - all rows from left, matching from right (with NaN for non-matches)
    Left,
    /// Right join - all rows from right, matching from left (with NaN for non-matches)
    Right,
    /// Outer join - all rows from both DataFrames (with NaN for non-matches)
    Outer,
}

impl From<JoinType> for CoreJoinType {
    fn from(how: JoinType) -> Self {
        match how {
            JoinType::Inner => CoreJoinType::Inner,
            JoinType::Left => CoreJoinType::Left,
            JoinType::Right => CoreJoinType::Right,
            JoinType::Outer => CoreJoinType::Outer,
        }
    }
}

/// Merge two DataFrames on a common column
///
/// This is a thin, pandas-flavoured wrapper over
/// [`crate::dataframe::join::join_with_suffixes`]: both entry points must agree
/// on key normalisation, NA handling and column layout, so there is exactly one
/// implementation of those rules.
///
/// # Semantics
///
/// * The key space is decided from **both** operands: two numeric key columns
///   are compared as `f64` (with `-0.0` and `0.0` collapsed onto one key), two
///   textual key columns are compared byte-for-byte, and a numeric/textual
///   mismatch is an error rather than a silent zero-match join or a panic.
/// * A missing (`NaN`) key never matches anything, not even another `NaN`;
///   the row itself is still emitted for the join types that keep unmatched
///   rows.
/// * Column dtypes are preserved. Where a join introduces missing values,
///   floats keep their type and use `NaN`, integers/booleans widen to `f64`
///   (as in pandas), and text/date columns fall back to a textual NA marker
///   because the column model has no nullable string type.
///
/// # Arguments
/// * `left` - Left DataFrame
/// * `right` - Right DataFrame
/// * `on` - Column name to join on (must exist in both DataFrames)
/// * `how` - Join type (inner, left, right, outer)
/// * `suffixes` - Tuple of suffixes to add to overlapping column names (left_suffix, right_suffix)
///
/// # Returns
/// Merged DataFrame whose columns are the left operand's columns in their
/// original order (the key column appears once, in its left-hand position)
/// followed by the right operand's non-key columns.
pub fn merge(
    left: &DataFrame,
    right: &DataFrame,
    on: &str,
    how: JoinType,
    suffixes: (&str, &str),
) -> Result<DataFrame> {
    // Validate that join column exists in both DataFrames. These checks are
    // kept here (rather than deferred to the join implementation) so that
    // `merge`'s documented error variant and wording stay stable.
    if !left.contains_column(on) {
        return Err(Error::InvalidValue(format!(
            "Join column '{}' not found in left DataFrame",
            on
        )));
    }
    if !right.contains_column(on) {
        return Err(Error::InvalidValue(format!(
            "Join column '{}' not found in right DataFrame",
            on
        )));
    }

    join_with_suffixes(left, right, on, how.into(), suffixes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::join::NA_STRING;
    use crate::series::Series;

    fn create_left_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "key".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                ],
                Some("key".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "value1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("value1".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df
    }

    fn create_right_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "key".to_string(),
            Series::new(
                vec![
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                    "E".to_string(),
                ],
                Some("key".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "value2".to_string(),
            Series::new(vec![20.0, 30.0, 40.0, 50.0], Some("value2".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df
    }

    #[test]
    fn test_merge_inner() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Inner, ("_x", "_y"))
            .expect("test should succeed");

        // Inner join should have 3 rows (B, C, D)
        assert_eq!(result.row_count(), 3);

        let keys = result
            .get_column_string_values("key")
            .expect("test should succeed");
        assert_eq!(keys, vec!["B", "C", "D"]);

        let val1 = result
            .get_column_numeric_values("value1")
            .expect("test should succeed");
        assert_eq!(val1, vec![2.0, 3.0, 4.0]);

        let val2 = result
            .get_column_numeric_values("value2")
            .expect("test should succeed");
        assert_eq!(val2, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_merge_left() {
        let left = create_left_df();
        let right = create_right_df();

        let result =
            merge(&left, &right, "key", JoinType::Left, ("_x", "_y")).expect("test should succeed");

        // Left join should have 4 rows (all from left: A, B, C, D)
        assert_eq!(result.row_count(), 4);

        let keys = result
            .get_column_string_values("key")
            .expect("test should succeed");
        assert_eq!(keys, vec!["A", "B", "C", "D"]);

        let val1 = result
            .get_column_numeric_values("value1")
            .expect("test should succeed");
        assert_eq!(val1, vec![1.0, 2.0, 3.0, 4.0]);

        let val2 = result
            .get_column_numeric_values("value2")
            .expect("test should succeed");
        assert!(val2[0].is_nan()); // A has no match
        assert_eq!(val2[1], 20.0);
        assert_eq!(val2[2], 30.0);
        assert_eq!(val2[3], 40.0);
    }

    #[test]
    fn test_merge_right() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Right, ("_x", "_y"))
            .expect("test should succeed");

        // Right join should have 4 rows (all from right: B, C, D, E)
        assert_eq!(result.row_count(), 4);

        let keys = result
            .get_column_string_values("key")
            .expect("test should succeed");
        assert_eq!(keys, vec!["B", "C", "D", "E"]);

        let val1 = result
            .get_column_numeric_values("value1")
            .expect("test should succeed");
        assert_eq!(val1[0], 2.0);
        assert_eq!(val1[1], 3.0);
        assert_eq!(val1[2], 4.0);
        assert!(val1[3].is_nan()); // E has no match

        let val2 = result
            .get_column_numeric_values("value2")
            .expect("test should succeed");
        assert_eq!(val2, vec![20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_merge_outer() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Outer, ("_x", "_y"))
            .expect("test should succeed");

        // Outer join should have 5 rows (A, B, C, D, E)
        assert_eq!(result.row_count(), 5);

        let keys = result
            .get_column_string_values("key")
            .expect("test should succeed");
        assert_eq!(keys, vec!["A", "B", "C", "D", "E"]);

        let val1 = result
            .get_column_numeric_values("value1")
            .expect("test should succeed");
        assert_eq!(val1[0], 1.0); // A
        assert_eq!(val1[1], 2.0); // B
        assert_eq!(val1[2], 3.0); // C
        assert_eq!(val1[3], 4.0); // D
        assert!(val1[4].is_nan()); // E (no match in left)

        let val2 = result
            .get_column_numeric_values("value2")
            .expect("test should succeed");
        assert!(val2[0].is_nan()); // A (no match in right)
        assert_eq!(val2[1], 20.0); // B
        assert_eq!(val2[2], 30.0); // C
        assert_eq!(val2[3], 40.0); // D
        assert_eq!(val2[4], 50.0); // E
    }

    #[test]
    fn test_merge_with_overlapping_columns() {
        let mut left = DataFrame::new();
        left.add_column(
            "key".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("key".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        left.add_column(
            "value".to_string(),
            Series::new(vec![1.0, 2.0], Some("value".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");

        let mut right = DataFrame::new();
        right
            .add_column(
                "key".to_string(),
                Series::new(
                    vec!["A".to_string(), "B".to_string()],
                    Some("key".to_string()),
                )
                .expect("test should succeed"),
            )
            .expect("test should succeed");
        right
            .add_column(
                "value".to_string(),
                Series::new(vec![10.0, 20.0], Some("value".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");

        let result = merge(&left, &right, "key", JoinType::Inner, ("_left", "_right"))
            .expect("test should succeed");

        // Should have renamed overlapping 'value' column
        assert!(result.contains_column("value_left"));
        assert!(result.contains_column("value_right"));

        let val_left = result
            .get_column_numeric_values("value_left")
            .expect("test should succeed");
        assert_eq!(val_left, vec![1.0, 2.0]);

        let val_right = result
            .get_column_numeric_values("value_right")
            .expect("test should succeed");
        assert_eq!(val_right, vec![10.0, 20.0]);
    }

    #[test]
    fn test_merge_numeric_key() {
        let mut left = DataFrame::new();
        left.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("id".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        left.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                ],
                Some("name".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");

        let mut right = DataFrame::new();
        right
            .add_column(
                "id".to_string(),
                Series::new(vec![2.0, 3.0, 4.0], Some("id".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");
        right
            .add_column(
                "score".to_string(),
                Series::new(vec![85.0, 90.0, 95.0], Some("score".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");

        let result =
            merge(&left, &right, "id", JoinType::Inner, ("_x", "_y")).expect("test should succeed");

        assert_eq!(result.row_count(), 2); // Only 2.0 and 3.0 match

        let names = result
            .get_column_string_values("name")
            .expect("test should succeed");
        assert_eq!(names, vec!["Bob", "Charlie"]);

        let scores = result
            .get_column_numeric_values("score")
            .expect("test should succeed");
        assert_eq!(scores, vec![85.0, 90.0]);
    }

    /// The left key column is numeric and the right one is textual. This used
    /// to hit `.expect("test should succeed")` in the production path and abort
    /// the process; it must now be a recoverable error.
    #[test]
    fn test_merge_numeric_left_string_right_key_is_error_not_panic() {
        let mut left = DataFrame::new();
        left.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0], Some("id".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        left.add_column(
            "name".to_string(),
            Series::new(
                vec!["Alice".to_string(), "Bob".to_string()],
                Some("name".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");

        let mut right = DataFrame::new();
        right
            .add_column(
                "id".to_string(),
                Series::new(
                    vec!["one".to_string(), "two".to_string()],
                    Some("id".to_string()),
                )
                .expect("test should succeed"),
            )
            .expect("test should succeed");
        right
            .add_column(
                "score".to_string(),
                Series::new(vec![1.0, 2.0], Some("score".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");

        let result = merge(&left, &right, "id", JoinType::Inner, ("_x", "_y"));
        assert!(result.is_err());
    }

    /// String columns coming from the right operand keep their exact text; the
    /// rows with no right-hand match read back as NA rather than as `""`.
    #[test]
    fn test_merge_left_preserves_strings_and_marks_missing() {
        let mut left = DataFrame::new();
        left.add_column(
            "key".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("key".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");

        let mut right = DataFrame::new();
        right
            .add_column(
                "key".to_string(),
                Series::new(vec!["B".to_string()], Some("key".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");
        right
            .add_column(
                "code".to_string(),
                Series::new(vec!["007".to_string()], Some("code".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");

        let result =
            merge(&left, &right, "key", JoinType::Left, ("_x", "_y")).expect("test should succeed");

        assert_eq!(
            result
                .get_column_string_values("code")
                .expect("test should succeed"),
            vec![NA_STRING, "007"]
        );
    }
}
