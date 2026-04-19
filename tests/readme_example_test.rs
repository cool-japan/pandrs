//! Regression tests for README.md code snippets.
//!
//! These tests ensure that the Rust code blocks shown in `README.md`
//! compile and run against the real public API. They exist specifically to
//! catch drift between the README examples and the library surface.
//!
//! Tracking: https://github.com/cool-japan/pandrs/issues/13

use pandrs::dataframe::{AggFunc, GroupByExt, NamedAgg};
use pandrs::{DataFrame, Series};

/// Regression test for the README "Quick Start" snippet.
///
/// This mirrors the first Rust code block shown in `README.md`. Keep this
/// test in sync with that snippet so the README remains runnable.
#[test]
fn test_issue_13_readme_quick_start() -> Result<(), Box<dyn std::error::Error>> {
    // Create a DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()],
            Some("name".to_string()),
        )?,
    )?;
    df.add_column(
        "age".to_string(),
        Series::new(vec![30i64, 25, 35], Some("age".to_string()))?,
    )?;
    df.add_column(
        "department".to_string(),
        Series::new(
            vec![
                "Engineering".to_string(),
                "Engineering".to_string(),
                "Sales".to_string(),
            ],
            Some("department".to_string()),
        )?,
    )?;
    df.add_column(
        "salary".to_string(),
        Series::new(vec![75_000i64, 65_000, 85_000], Some("salary".to_string()))?,
    )?;

    // Compute the mean of a numeric column.
    let mean_salary = df.mean("salary")?;
    assert!((mean_salary - 75_000.0).abs() < 1e-9);

    // GroupBy + named aggregations. Use the explicit trait path because
    // `DataFrame` also has an inherent `groupby(&str)` method (from the
    // pivot module) that shadows `GroupByExt::groupby`.
    let grouped = GroupByExt::groupby(&df, &["department"])?.agg(vec![
        NamedAgg::new(
            "salary".to_string(),
            AggFunc::Mean,
            "salary_mean".to_string(),
        ),
        NamedAgg::new("salary".to_string(), AggFunc::Sum, "salary_sum".to_string()),
        NamedAgg::new("age".to_string(), AggFunc::Max, "age_max".to_string()),
    ])?;
    assert_eq!(grouped.row_count(), 2);
    assert!(grouped.contains_column("salary_mean"));
    assert!(grouped.contains_column("salary_sum"));
    assert!(grouped.contains_column("age_max"));

    Ok(())
}

/// Regression test for the README "Basic Data Analysis" snippet.
///
/// This mirrors the Rust code block titled "Basic Data Analysis" in
/// `README.md`. Because `DataFrame::from_csv` is currently a stub, the test
/// builds the data inline to stay focused on the filter + groupby + agg API
/// shape that the README advertises.
#[test]
fn test_issue_13_readme_basic_data_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = DataFrame::new();
    df.add_column(
        "city".to_string(),
        Series::new(
            vec![
                "Tallinn".to_string(),
                "Tallinn".to_string(),
                "Tartu".to_string(),
            ],
            Some("city".to_string()),
        )?,
    )?;
    df.add_column(
        "occupation".to_string(),
        Series::new(
            vec![
                "Engineer".to_string(),
                "Engineer".to_string(),
                "Analyst".to_string(),
            ],
            Some("occupation".to_string()),
        )?,
    )?;
    df.add_column(
        "age".to_string(),
        Series::new(vec![21i64, 34, 40], Some("age".to_string()))?,
    )?;
    df.add_column(
        "income".to_string(),
        Series::new(vec![55_000i64, 72_000, 81_000], Some("income".to_string()))?,
    )?;

    // Group + named aggregations (the concrete API the README claims).
    let result = GroupByExt::groupby(&df, &["city", "occupation"])?.agg(vec![
        NamedAgg::new(
            "income".to_string(),
            AggFunc::Mean,
            "income_mean".to_string(),
        ),
        NamedAgg::new(
            "income".to_string(),
            AggFunc::Median,
            "income_median".to_string(),
        ),
        NamedAgg::new("income".to_string(), AggFunc::Std, "income_std".to_string()),
        NamedAgg::new("age".to_string(), AggFunc::Mean, "age_mean".to_string()),
    ])?;

    assert!(result.contains_column("income_mean"));
    assert!(result.contains_column("income_median"));
    assert!(result.contains_column("income_std"));
    assert!(result.contains_column("age_mean"));
    assert_eq!(result.row_count(), 2);

    Ok(())
}

/// Regression test for `pandrs::prelude` (issue #13, complaint #1).
///
/// Ensures that `use pandrs::prelude::*` compiles and exposes the types that
/// users would expect: `DataFrame`, `Series`, `GroupByExt`, `AggFunc`,
/// `NamedAgg`, and `OptimizedDataFrame`.
#[test]
fn test_issue_13_prelude_exists_and_exports_core_types() -> Result<(), Box<dyn std::error::Error>> {
    use pandrs::prelude::*;

    // DataFrame and Series are accessible via the prelude.
    let mut df = DataFrame::new();
    df.add_column(
        "score".to_string(),
        Series::new(vec![10i64, 20, 30], Some("score".to_string()))?,
    )?;
    assert_eq!(df.row_count(), 3);

    // GroupByExt and AggFunc are accessible via the prelude.
    let grouped = GroupByExt::groupby(&df, &["score"])?.agg(vec![NamedAgg::new(
        "score".to_string(),
        AggFunc::Sum,
        "score_sum".to_string(),
    )])?;
    assert!(grouped.contains_column("score_sum"));

    // OptimizedDataFrame is also accessible.
    let mut opt = OptimizedDataFrame::new();
    opt.add_int_column("values", vec![1, 2, 3])?;
    assert_eq!(opt.row_count(), 3);

    Ok(())
}
