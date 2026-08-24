//! Regression tests for the Wave 3b API-surface hygiene pass.
//!
//! These are largely compile-level assertions that the public-API cleanups
//! stay in place:
//!
//! * Item 2 — the `pandrs::error` module and `PandRSError` are a *stable,
//!   non-deprecated* alias of `pandrs::core::error`. The crate-level
//!   `#![deny(deprecated)]` below turns any reintroduced `#[deprecated]` on
//!   that path (or on any prelude export used here) into a hard compile error.
//! * Item 4 — the prelude exports the I/O free functions (`read_csv`,
//!   `write_json`, `JsonOrient`), the reshaping option structs
//!   (`MeltOptions`/`StackOptions`/`UnstackOptions`), and `JoinType`, all
//!   nameable from the prelude glob without ambiguity.
//! * Item 5 — the pivot inherent `groupby` was renamed to `groupby_pivot`, so
//!   `df.groupby(&[..])` in method syntax now resolves to the multi-key
//!   `GroupByExt::groupby` trait method instead of being shadowed.
//! * Item 8 — `stats::StatisticalAnalyzer::columns_ttest` exists under its new
//!   (non-`test_`-prefixed) name.
#![deny(deprecated)]

use pandrs::prelude::*;
use pandrs::stats::StatisticalAnalyzer;

#[test]
fn error_module_is_stable_non_deprecated_alias() {
    // Referencing the `pandrs::error` module path directly. Under
    // `#![deny(deprecated)]` this fails to compile if the alias is ever marked
    // `#[deprecated]` again.
    let err = pandrs::error::PandRSError::Column("stable-alias".to_string());
    assert!(matches!(err, pandrs::error::PandRSError::Column(_)));
}

#[test]
fn prelude_io_free_functions_round_trip() {
    use std::fs;

    let dir = std::env::temp_dir();
    let csv_path = dir.join("pandrs_w3b_api_surface.csv");
    let json_path = dir.join("pandrs_w3b_api_surface.json");

    fs::write(&csv_path, "name,age\nAlice,30\nBob,25\n").expect("write csv fixture");

    // `read_csv` comes from the prelude (item 4).
    let df = read_csv(&csv_path, true).expect("read_csv from prelude");
    assert_eq!(df.row_count(), 2);

    // `write_json` + `JsonOrient` come from the prelude (item 4).
    write_json(&df, &json_path, JsonOrient::Records).expect("write_json from prelude");
    let written = fs::read_to_string(&json_path).expect("read back json");
    assert!(!written.is_empty());

    let _ = fs::remove_file(&csv_path);
    let _ = fs::remove_file(&json_path);
}

#[test]
fn groupby_method_syntax_resolves_to_trait_not_pivot_shadow() {
    let mut df = DataFrame::new();
    df.add_column(
        "category".to_string(),
        Series::new(
            vec!["a".to_string(), "a".to_string(), "b".to_string()],
            Some("category".to_string()),
        )
        .expect("category series"),
    )
    .expect("add category");
    df.add_column(
        "value".to_string(),
        Series::new(vec![1i64, 2, 3], Some("value".to_string())).expect("value series"),
    )
    .expect("add value");

    // Method syntax with a slice argument resolves to `GroupByExt::groupby`
    // (brought in by the prelude). This compiles ONLY because the pivot
    // inherent `groupby(&str)` was renamed to `groupby_pivot`; otherwise the
    // inherent method shadows the trait and `&["category"]` fails to type-check
    // against `&str`.
    let grouped = df.groupby(&["category"]).expect("trait groupby");
    let _ = grouped;

    // The pivot-style single-column grouping is still reachable under its new,
    // non-shadowing name.
    let pivot_group = df.groupby_pivot("category").expect("pivot groupby");
    let _ = pivot_group.sum(&["value"]).expect("pivot sum");
}

#[test]
fn prelude_and_renamed_symbols_are_nameable() {
    // Item 4: reshaping option structs + `JoinType` are nameable from the
    // prelude glob (type positions, so no constructor knowledge is required).
    let _: Option<MeltOptions> = None;
    let _: Option<StackOptions> = None;
    let _: Option<UnstackOptions> = None;
    let _: Option<JoinType> = None;

    // Item 8: the renamed t-test method resolves under its new name.
    let _ttest_fn = StatisticalAnalyzer::columns_ttest;
}
