//! Wave 3b regression test for `pandrs::plugins`.
//!
//! This is a follow-up to `plugins_versioning_schema_w3_regression_test.rs`
//! covering one defect found during this pass's verification that wasn't
//! already covered there: `FilterTransformPlugin::transform`'s zero-match
//! path (`build_empty_like`) used to collapse every column to `f64`/`String`
//! regardless of its real element type -- the same blanket-recasting defect
//! `copy_column_typed` exists to avoid on the non-empty path (see
//! `src/plugins/builtins/transform.rs`). A focused white-box test for the
//! same fix also lives alongside the code
//! (`test_filter_zero_matches_preserves_concrete_column_types` in
//! `src/plugins/builtins/transform.rs`); this file covers the same ground
//! through the crate's public surface instead.
//!
//! All five scenarios `plugins_versioning_schema_w3_regression_test.rs` was
//! asked to confirm (migration preserves i64/bool; NotNull catches numeric
//! NA; `add_migration` then `migrate()` finds path; versioning uuid
//! uniqueness; plugin unregister) are already present and passing in that
//! file, so nothing is duplicated here.

use pandrs::plugins::{FilterTransformPlugin, TransformPlugin};
use pandrs::{DataFrame, Series};
use std::collections::HashMap;

#[test]
fn regression_filter_zero_matches_preserves_concrete_column_types() {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("series"),
    )
    .expect("add id");
    df.add_column(
        "active".to_string(),
        Series::new(vec![true, false, true], Some("active".to_string())).expect("series"),
    )
    .expect("add active");
    df.add_column(
        "score".to_string(),
        Series::new(vec![1.5f64, 2.5, 3.5], Some("score".to_string())).expect("series"),
    )
    .expect("add score");

    let mut opts = HashMap::new();
    opts.insert("column".to_string(), "id".to_string());
    opts.insert("operator".to_string(), "eq".to_string());
    // No row has id == 999: the filter matches zero rows, which routes
    // through the DataFrame-building path that used to collapse every
    // column's type.
    opts.insert("value".to_string(), "999".to_string());

    let result = FilterTransformPlugin::new()
        .transform(df, &opts)
        .expect("a filter matching zero rows must still succeed");
    assert_eq!(result.row_count(), 0);

    // Each column must come back as its real concrete type -- "no rows
    // matched" must stay distinguishable from "the schema changed".
    assert!(
        result.get_column::<i64>("id").is_ok(),
        "id must stay i64, not become f64"
    );
    assert!(
        result.get_column::<bool>("active").is_ok(),
        "active must stay bool, not become String"
    );
    assert!(
        result.get_column::<f64>("score").is_ok(),
        "score must stay f64"
    );
}
