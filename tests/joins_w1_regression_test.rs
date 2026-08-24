//! Regression tests for the join / merge / concat fixes (wave 1).
//!
//! Every test here reproduces a specific defect that shipped in the previous
//! implementation:
//!
//! * `right_join` was implemented as `other.left_join(self)`, which swapped the
//!   output columns and their suffixes -- `v` and `v_right` came out exactly
//!   inverted with respect to pandas.
//! * the hash join materialised *every* column as `Series<String>`, destroying
//!   `i64`/`f64` dtypes.
//! * missing join keys were rendered as text and therefore all matched each
//!   other (pandas: `NaN` never matches).
//! * `merge` picked its key encoding per side, so a numeric-left/textual-right
//!   key hit `.expect("test should succeed")` in the production path.
//! * `concat` inferred each column's type from the first frame that had it,
//!   so `["007", "008"] + ["x", "y"]` became `[7, 8, NaN, NaN]` and the result
//!   depended on the operand order.
//! * `concat`'s `ignore_index` argument was ignored entirely.

use pandrs::dataframe::join::{join_with_suffixes, JoinExt, JoinType};
use pandrs::dataframe::pandas_compat::concat::{concat, ConcatAxis};
use pandrs::dataframe::pandas_compat::merge::{merge, JoinType as MergeHow};
use pandrs::index::Index;
use pandrs::{DataFrame, Series};

/// Marker a missing value reads back as in a column whose element type has no
/// in-band NA representation (kept in sync with `dataframe::join::NA_STRING`,
/// which is crate-private).
const NA: &str = "NaN";

fn df_i64(name: &str, values: Vec<i64>) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string())).unwrap(),
    )
    .unwrap();
    df
}

fn with_i64(mut df: DataFrame, name: &str, values: Vec<i64>) -> DataFrame {
    df.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string())).unwrap(),
    )
    .unwrap();
    df
}

fn with_f64(mut df: DataFrame, name: &str, values: Vec<f64>) -> DataFrame {
    df.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string())).unwrap(),
    )
    .unwrap();
    df
}

fn with_str(mut df: DataFrame, name: &str, values: Vec<&str>) -> DataFrame {
    df.add_column(
        name.to_string(),
        Series::new(
            values.into_iter().map(str::to_string).collect::<Vec<_>>(),
            Some(name.to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    df
}

fn df_str(name: &str, values: Vec<&str>) -> DataFrame {
    with_str(DataFrame::new(), name, values)
}

fn df_f64(name: &str, values: Vec<f64>) -> DataFrame {
    with_f64(DataFrame::new(), name, values)
}

// ---------------------------------------------------------------------------
// right join orientation
// ---------------------------------------------------------------------------

/// pandas:
/// ```text
/// >>> pd.merge(left, right, on="id", how="right", suffixes=("", "_right"))
///    id    v v_right
/// 0   2   l2      r2
/// 1   3  NaN      r3
/// ```
/// The old implementation delegated to `other.left_join(self)` and produced
/// `v = ["r2", "r3"]` / `v_right = ["l2", ""]` -- both the values and the
/// column layout were inverted.
#[test]
fn right_join_keeps_left_columns_first_and_does_not_invert_values() {
    let left = with_str(df_i64("id", vec![1, 2]), "v", vec!["l1", "l2"]);
    let right = with_str(df_i64("id", vec![2, 3]), "v", vec!["r2", "r3"]);

    let result = left.right_join(&right, "id").unwrap();

    assert_eq!(result.column_names(), &["id", "v", "v_right"]);
    assert_eq!(result.row_count(), 2);
    assert_eq!(
        result.get_column_string_values("id").unwrap(),
        vec!["2", "3"]
    );
    // The left operand's column keeps the LEFT values.
    assert_eq!(
        result.get_column_string_values("v").unwrap(),
        vec!["l2", NA]
    );
    // ... and the suffixed column keeps the RIGHT values.
    assert_eq!(
        result.get_column_string_values("v_right").unwrap(),
        vec!["r2", "r3"]
    );
}

/// A right join keeps every row of the right operand, in the right operand's
/// own order, and drops left-only rows.
#[test]
fn right_join_row_set_and_order_follow_the_right_operand() {
    let left = with_str(df_i64("id", vec![3, 1, 2]), "l", vec!["c", "a", "b"]);
    let right = with_str(df_i64("id", vec![2, 9, 1]), "r", vec!["x", "y", "z"]);

    let result = left.right_join(&right, "id").unwrap();

    assert_eq!(
        result.get_column_string_values("id").unwrap(),
        vec!["2", "9", "1"]
    );
    assert_eq!(
        result.get_column_string_values("l").unwrap(),
        vec!["b", NA, "a"]
    );
    assert_eq!(
        result.get_column_string_values("r").unwrap(),
        vec!["x", "y", "z"]
    );
}

// ---------------------------------------------------------------------------
// dtype preservation
// ---------------------------------------------------------------------------

#[test]
fn inner_join_preserves_i64_and_f64_dtypes() {
    let left = with_f64(df_i64("id", vec![1, 2, 3]), "amount", vec![1.5, 2.5, 3.5]);
    let right = with_str(df_i64("id", vec![2, 3]), "label", vec!["b", "c"]);

    let result = left.inner_join(&right, "id").unwrap();

    assert_eq!(result.row_count(), 2);
    assert_eq!(result.get_column::<i64>("id").unwrap().values(), &[2i64, 3]);
    assert_eq!(
        result.get_column::<f64>("amount").unwrap().values(),
        &[2.5f64, 3.5]
    );
    assert_eq!(
        result.get_column::<String>("label").unwrap().values(),
        &["b".to_string(), "c".to_string()]
    );
}

#[test]
fn left_join_with_all_rows_matched_preserves_i64() {
    let left = df_i64("id", vec![1, 2]);
    let right = with_i64(df_i64("id", vec![1, 2]), "qty", vec![10, 20]);

    let result = left.left_join(&right, "id").unwrap();

    assert_eq!(result.get_column::<i64>("id").unwrap().values(), &[1i64, 2]);
    // Nothing is missing, so the integer column keeps its exact dtype.
    assert_eq!(
        result.get_column::<i64>("qty").unwrap().values(),
        &[10i64, 20]
    );
}

#[test]
fn left_join_widens_integers_to_float_for_missing_rows() {
    let left = df_i64("id", vec![1, 2]);
    let right = with_i64(df_i64("id", vec![2]), "qty", vec![10]);

    let result = left.left_join(&right, "id").unwrap();

    // pandas performs the same int64 -> float64 upcast when a join introduces
    // missing values; the unmatched row must be NA, never 0.
    let qty = result.get_column::<f64>("qty").unwrap().values().to_vec();
    assert_eq!(qty.len(), 2);
    assert!(qty[0].is_nan(), "unmatched row must be NA, got {}", qty[0]);
    assert_eq!(qty[1], 10.0);

    // The key column has a value on every row, so it keeps its i64 dtype.
    assert_eq!(result.get_column::<i64>("id").unwrap().values(), &[1i64, 2]);
}

#[test]
fn outer_join_preserves_dtypes_and_marks_missing_values() {
    let left = with_f64(df_i64("id", vec![1, 2]), "lv", vec![1.0, 2.0]);
    let right = with_f64(df_i64("id", vec![2, 3]), "rv", vec![20.0, 30.0]);

    let result = left.outer_join(&right, "id").unwrap();

    assert_eq!(result.row_count(), 3);
    assert_eq!(
        result.get_column::<i64>("id").unwrap().values(),
        &[1i64, 2, 3]
    );
    let lv = result.get_column::<f64>("lv").unwrap().values().to_vec();
    assert_eq!(lv[0], 1.0);
    assert_eq!(lv[1], 2.0);
    assert!(lv[2].is_nan());
    let rv = result.get_column::<f64>("rv").unwrap().values().to_vec();
    assert!(rv[0].is_nan());
    assert_eq!(rv[1], 20.0);
    assert_eq!(rv[2], 30.0);
}

/// Text values must survive verbatim: re-inference used to turn `"007"` into
/// the float `7.0`.
#[test]
fn join_does_not_re_infer_numeric_looking_text() {
    let left = with_str(df_i64("id", vec![1]), "code", vec!["007"]);
    let right = with_str(df_i64("id", vec![1]), "other", vec!["008"]);

    let result = left.inner_join(&right, "id").unwrap();

    assert_eq!(
        result.get_column::<String>("code").unwrap().values(),
        &["007".to_string()]
    );
    assert_eq!(
        result.get_column_string_values("other").unwrap(),
        vec!["008"]
    );
}

// ---------------------------------------------------------------------------
// NA join keys
// ---------------------------------------------------------------------------

/// pandas: `NaN` keys never match, not even another `NaN`, so this inner join
/// has zero rows. The old implementation rendered the keys as the text `"NaN"`
/// and joined every missing key to every other one.
#[test]
fn nan_join_keys_produce_zero_matches() {
    let left = with_str(df_f64("id", vec![f64::NAN, 1.0]), "l", vec!["a", "b"]);
    let right = with_str(df_f64("id", vec![f64::NAN, 2.0]), "r", vec!["x", "y"]);

    let result = left.inner_join(&right, "id").unwrap();

    assert_eq!(result.row_count(), 0);
    // An empty result still has the full schema.
    assert_eq!(result.column_names(), &["id", "l", "r"]);
}

#[test]
fn left_join_keeps_rows_with_na_keys_but_matches_nothing() {
    let left = with_str(df_f64("id", vec![f64::NAN, 1.0]), "l", vec!["a", "b"]);
    let right = with_str(df_f64("id", vec![f64::NAN, 1.0]), "r", vec!["x", "y"]);

    let result = left.left_join(&right, "id").unwrap();

    // Both left rows survive; only the non-NA key finds a partner.
    assert_eq!(result.row_count(), 2);
    assert_eq!(
        result.get_column_string_values("l").unwrap(),
        vec!["a", "b"]
    );
    assert_eq!(result.get_column_string_values("r").unwrap(), vec![NA, "y"]);
}

#[test]
fn outer_join_keeps_na_keys_from_both_sides_unmatched() {
    let left = df_f64("id", vec![f64::NAN]);
    let right = df_f64("id", vec![f64::NAN]);

    let result = left.outer_join(&right, "id").unwrap();

    // pandas emits one row per unmatched NaN key on each side.
    assert_eq!(result.row_count(), 2);
}

/// `-0.0` and `0.0` are the same number and must land in the same key bucket
/// even though their bit patterns differ.
#[test]
fn negative_zero_and_positive_zero_keys_match() {
    let left = with_str(df_f64("id", vec![-0.0]), "l", vec!["a"]);
    let right = with_str(df_f64("id", vec![0.0]), "r", vec!["x"]);

    let result = left.inner_join(&right, "id").unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.get_column_string_values("l").unwrap(), vec!["a"]);
    assert_eq!(result.get_column_string_values("r").unwrap(), vec!["x"]);
}

/// An i64 key and an f64 key describe the same numbers and must join.
#[test]
fn integer_and_float_key_columns_share_one_key_space() {
    let left = with_str(df_i64("id", vec![1, 2]), "l", vec!["a", "b"]);
    let right = with_str(df_f64("id", vec![2.0, 3.0]), "r", vec!["x", "y"]);

    let result = left.inner_join(&right, "id").unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.get_column_string_values("l").unwrap(), vec!["b"]);
    assert_eq!(result.get_column_string_values("r").unwrap(), vec!["x"]);
}

// ---------------------------------------------------------------------------
// suffixes
// ---------------------------------------------------------------------------

#[test]
fn join_suffixes_are_configurable() {
    let left = with_str(df_i64("id", vec![1]), "v", vec!["l"]);
    let right = with_str(df_i64("id", vec![1]), "v", vec!["r"]);

    let result = join_with_suffixes(&left, &right, "id", JoinType::Inner, ("_x", "_y")).unwrap();

    assert_eq!(result.column_names(), &["id", "v_x", "v_y"]);
    assert_eq!(result.get_column_string_values("v_x").unwrap(), vec!["l"]);
    assert_eq!(result.get_column_string_values("v_y").unwrap(), vec!["r"]);

    // The default stays "_right" on the right-hand side only.
    let defaulted = left.inner_join(&right, "id").unwrap();
    assert_eq!(defaulted.column_names(), &["id", "v", "v_right"]);
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

/// This used to abort the process through `.expect("test should succeed")` in
/// the production merge path.
#[test]
fn merge_numeric_left_string_right_key_returns_error_instead_of_panicking() {
    let left = with_str(df_f64("id", vec![1.0, 2.0]), "name", vec!["a", "b"]);
    let right = with_f64(df_str("id", vec!["one", "two"]), "score", vec![1.0, 2.0]);

    let result = merge(&left, &right, "id", MergeHow::Inner, ("_x", "_y"));
    assert!(
        result.is_err(),
        "mixed key dtypes must be a recoverable error"
    );

    // ... in every direction and for every join type.
    for how in [
        MergeHow::Inner,
        MergeHow::Left,
        MergeHow::Right,
        MergeHow::Outer,
    ] {
        assert!(merge(&left, &right, "id", how, ("_x", "_y")).is_err());
        assert!(merge(&right, &left, "id", how, ("_x", "_y")).is_err());
    }
}

/// Two textual key columns holding numeric-looking text must be compared as
/// text (the old implementation encoded one side as float bits and the other
/// as text, so nothing ever matched).
#[test]
fn merge_matches_textual_keys_verbatim() {
    let left = with_str(df_str("key", vec!["007", "08"]), "l", vec!["a", "b"]);
    let right = with_str(df_str("key", vec!["007", "8"]), "r", vec!["x", "y"]);

    let result = merge(&left, &right, "key", MergeHow::Inner, ("_x", "_y")).unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.get_column_string_values("key").unwrap(), vec!["007"]);
    assert_eq!(result.get_column_string_values("l").unwrap(), vec!["a"]);
    assert_eq!(result.get_column_string_values("r").unwrap(), vec!["x"]);
}

/// The suffix rename path used to rebuild the whole frame through numeric /
/// string re-inference, which destroyed dtypes on the way.
#[test]
fn merge_suffix_rename_preserves_dtypes() {
    let left = with_str(df_i64("id", vec![1]), "v", vec!["007"]);
    let right = with_i64(df_i64("id", vec![1]), "v", vec![42]);

    let result = merge(&left, &right, "id", MergeHow::Inner, ("_l", "_r")).unwrap();

    assert_eq!(result.column_names(), &["id", "v_l", "v_r"]);
    assert_eq!(
        result.get_column::<String>("v_l").unwrap().values(),
        &["007".to_string()]
    );
    assert_eq!(result.get_column::<i64>("v_r").unwrap().values(), &[42i64]);
    assert_eq!(result.get_column::<i64>("id").unwrap().values(), &[1i64]);
}

// ---------------------------------------------------------------------------
// concat
// ---------------------------------------------------------------------------

/// `["007", "008"] + ["x", "y"]` used to become `[7, 8, NaN, NaN]` because the
/// column type was inferred from the first frame only (and `"007"` parses as a
/// number). Reversing the operands produced a different result.
#[test]
fn concat_rows_is_order_independent_and_preserves_text() {
    let numeric_looking = df_str("code", vec!["007", "008"]);
    let plain_text = df_str("code", vec!["x", "y"]);

    let forward = concat(&[&numeric_looking, &plain_text], ConcatAxis::Rows, true).unwrap();
    assert_eq!(
        forward.get_column_string_values("code").unwrap(),
        vec!["007", "008", "x", "y"]
    );

    let reversed = concat(&[&plain_text, &numeric_looking], ConcatAxis::Rows, true).unwrap();
    assert_eq!(
        reversed.get_column_string_values("code").unwrap(),
        vec!["x", "y", "007", "008"]
    );
}

/// The same order-independence when the frames genuinely disagree about the
/// column's dtype: the union falls back to text and every frame's own values
/// survive.
#[test]
fn concat_rows_with_conflicting_dtypes_falls_back_to_text_either_way() {
    let numbers = df_i64("v", vec![1, 2]);
    let text = df_str("v", vec!["a", "b"]);

    let forward = concat(&[&numbers, &text], ConcatAxis::Rows, true).unwrap();
    assert_eq!(
        forward.get_column_string_values("v").unwrap(),
        vec!["1", "2", "a", "b"]
    );

    let reversed = concat(&[&text, &numbers], ConcatAxis::Rows, true).unwrap();
    assert_eq!(
        reversed.get_column_string_values("v").unwrap(),
        vec!["a", "b", "1", "2"]
    );
}

#[test]
fn concat_rows_unions_integer_and_float_columns_as_float() {
    let ints = df_i64("v", vec![1, 2]);
    let floats = df_f64("v", vec![3.5]);

    let result = concat(&[&ints, &floats], ConcatAxis::Rows, true).unwrap();

    assert_eq!(
        result.get_column::<f64>("v").unwrap().values(),
        &[1.0f64, 2.0, 3.5]
    );
}

#[test]
fn concat_rows_preserves_integer_dtype_when_all_frames_agree() {
    let a = df_i64("v", vec![1, 2]);
    let b = df_i64("v", vec![3]);

    let result = concat(&[&a, &b], ConcatAxis::Rows, true).unwrap();

    assert_eq!(
        result.get_column::<i64>("v").unwrap().values(),
        &[1i64, 2, 3]
    );
}

/// Rows contributed by a frame that lacks a text column are missing, not `""`.
#[test]
fn concat_rows_fills_absent_text_columns_with_na_not_empty_string() {
    let left = df_str("a", vec!["p", "q"]);
    let right = df_str("b", vec!["r"]);

    let result = concat(&[&left, &right], ConcatAxis::Rows, true).unwrap();

    assert_eq!(result.row_count(), 3);
    assert_eq!(
        result.get_column_string_values("a").unwrap(),
        vec!["p", "q", NA]
    );
    assert_eq!(
        result.get_column_string_values("b").unwrap(),
        vec![NA, NA, "r"]
    );
}

/// `ignore_index` used to be an `_`-prefixed dead parameter even though the
/// rustdoc documented it.
#[test]
fn concat_rows_ignore_index_renumbers_rows() {
    let mut a = df_f64("v", vec![1.0, 2.0]);
    a.set_index(Index::new(vec!["a1".to_string(), "a2".to_string()]).unwrap())
        .unwrap();
    let mut b = df_f64("v", vec![3.0]);
    b.set_index(Index::new(vec!["b1".to_string()]).unwrap())
        .unwrap();

    let renumbered = concat(&[&a, &b], ConcatAxis::Rows, true).unwrap();
    assert_eq!(renumbered.row_count(), 3);
    // No labels are carried over: the result uses the implicit positional index.
    assert_eq!(renumbered.get_index().string_values(), Some(Vec::new()));

    let kept = concat(&[&a, &b], ConcatAxis::Rows, false).unwrap();
    assert_eq!(
        kept.get_index().string_values(),
        Some(vec!["a1".to_string(), "a2".to_string(), "b1".to_string()])
    );
}

/// Frames without labels of their own have nothing to preserve, so
/// `ignore_index = false` must keep working for them.
#[test]
fn concat_rows_without_explicit_indexes_still_works_with_ignore_index_false() {
    let a = df_f64("v", vec![1.0, 2.0]);
    let b = df_f64("v", vec![3.0]);

    let result = concat(&[&a, &b], ConcatAxis::Rows, false).unwrap();
    assert_eq!(result.row_count(), 3);
    assert_eq!(result.get_index().string_values(), Some(Vec::new()));

    // Real labels that collide are reported rather than silently renumbered.
    let mut labelled = df_f64("v", vec![9.0]);
    labelled
        .set_index(Index::new(vec!["0".to_string()]).unwrap())
        .unwrap();
    assert!(concat(&[&labelled, &a], ConcatAxis::Rows, false).is_err());
}

/// A single frame is still renumbered when `ignore_index` is set.
#[test]
fn concat_single_frame_honours_ignore_index() {
    let mut a = df_f64("v", vec![1.0, 2.0]);
    a.set_index(Index::new(vec!["a1".to_string(), "a2".to_string()]).unwrap())
        .unwrap();

    let kept = concat(&[&a], ConcatAxis::Rows, false).unwrap();
    assert_eq!(
        kept.get_index().string_values(),
        Some(vec!["a1".to_string(), "a2".to_string()])
    );

    let renumbered = concat(&[&a], ConcatAxis::Rows, true).unwrap();
    assert_eq!(renumbered.get_index().string_values(), Some(Vec::new()));
    assert_eq!(
        renumbered.get_column::<f64>("v").unwrap().values(),
        &[1.0f64, 2.0]
    );
}

/// On `axis = Columns` the concatenation axis is the column labels.
#[test]
fn concat_columns_ignore_index_replaces_column_labels() {
    let a = df_f64("a", vec![1.0, 2.0]);
    let b = df_f64("b", vec![10.0, 20.0]);

    let positional = concat(&[&a, &b], ConcatAxis::Columns, true).unwrap();
    assert_eq!(positional.column_names(), &["0", "1"]);

    let named = concat(&[&a, &b], ConcatAxis::Columns, false).unwrap();
    assert_eq!(named.column_names(), &["a", "b"]);
}

/// Column-wise concatenation used to re-infer every column's dtype.
#[test]
fn concat_columns_preserves_dtypes() {
    let a = df_str("code", vec!["007"]);
    let b = df_i64("n", vec![5]);

    let result = concat(&[&a, &b], ConcatAxis::Columns, false).unwrap();

    assert_eq!(
        result.get_column::<String>("code").unwrap().values(),
        &["007".to_string()]
    );
    assert_eq!(result.get_column::<i64>("n").unwrap().values(), &[5i64]);
}
