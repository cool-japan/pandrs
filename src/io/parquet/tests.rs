//! Unit tests for schema evolution and predicate pushdown internals.

use super::evolution::{
    apply_predicate_filters, apply_schema_evolution, PredicateFilter, SchemaEvolution,
};
use crate::dataframe::DataFrame;
use crate::series::Series;

fn sample_df() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    df.add_column(
        "age".to_string(),
        Series::new(vec![10i64, 20, 30], None).unwrap(),
    )
    .unwrap();
    df
}

#[test]
fn test_apply_predicate_equals() {
    let result = apply_predicate_filters(
        sample_df(),
        &[PredicateFilter::Equals("name".to_string(), "b".to_string())],
    )
    .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.get_column_string_values("name").unwrap(), vec!["b"]);
}

#[test]
fn test_apply_predicate_range() {
    let result = apply_predicate_filters(
        sample_df(),
        &[PredicateFilter::Range(
            "age".to_string(),
            "15".to_string(),
            "100".to_string(),
        )],
    )
    .unwrap();
    assert_eq!(result.row_count(), 2);
    assert_eq!(
        result.get_column_string_values("age").unwrap(),
        vec!["20", "30"]
    );
}

#[test]
fn test_apply_predicate_custom_is_not_implemented() {
    let result = apply_predicate_filters(
        sample_df(),
        &[PredicateFilter::Custom("age > 1".to_string())],
    );
    assert!(result.is_err());
}

#[test]
fn test_schema_evolution_rename_preserves_data() {
    let mut df = sample_df();
    let mut evolution = SchemaEvolution::default();
    evolution
        .column_mappings
        .insert("age".to_string(), "years".to_string());

    apply_schema_evolution(&mut df, &evolution).unwrap();

    assert!(df.contains_column("years"));
    assert!(!df.contains_column("age"));
    assert_eq!(
        df.get_column_string_values("years").unwrap(),
        vec!["10", "20", "30"]
    );
}
