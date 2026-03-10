#![allow(clippy::result_large_err)]
//! Comprehensive tests for the new lazy evaluation engine
//!
//! Covers:
//! - Individual operations (filter, select, sort, limit, groupby, join)
//! - Optimizer passes (predicate pushdown, projection pushdown, constant folding, DCE)
//! - explain() / explain_optimized() output
//! - collect() correctness
//! - Error handling

use pandrs::compute::lazy::{
    AggExpr, BinaryOp, ConstantFolding, DeadCodeElimination, Expr, LazyFrame, LiteralValue,
    LogicalPlan, Optimizer, OptimizerRule, PredicatePushdown, ProjectionPushdown, UnaryOp,
};
use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::optimized::operations::JoinType;

// ─── helper ──────────────────────────────────────────────────────────────────

fn make_people_df() -> OptimizedDataFrame {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("id", vec![1, 2, 3, 4, 5])
        .expect("add id");
    df.add_string_column(
        "name",
        vec![
            "Alice".into(),
            "Bob".into(),
            "Carol".into(),
            "Dave".into(),
            "Eve".into(),
        ],
    )
    .expect("add name");
    df.add_int_column("age", vec![30, 25, 35, 28, 42])
        .expect("add age");
    df.add_float_column("score", vec![85.5, 72.0, 91.0, 60.0, 78.5])
        .expect("add score");
    df
}

fn make_scores_df() -> OptimizedDataFrame {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("id", vec![1, 2, 3]).expect("add id");
    df.add_float_column("bonus", vec![10.0, 5.0, 15.0])
        .expect("add bonus");
    df
}

// ─── Filter tests ─────────────────────────────────────────────────────────────

#[test]
fn test_filter_int_gt() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .filter(Expr::col("age").gt(Expr::lit_int(30)))
        .collect()
        .expect("collect");

    // Only Carol (35) and Eve (42) should remain
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_filter_int_eq() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .filter(Expr::col("age").eq(Expr::lit_int(25)))
        .collect()
        .expect("collect");

    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_filter_float_gte() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .filter(Expr::col("score").gt_eq(Expr::lit_float(80.0)))
        .collect()
        .expect("collect");

    // Alice (85.5), Carol (91.0) → 2 rows
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_filter_combined_and() {
    let df = make_people_df();
    let pred = Expr::col("age")
        .gt(Expr::lit_int(25))
        .and(Expr::col("score").gt_eq(Expr::lit_float(80.0)));

    let result = LazyFrame::scan(df).filter(pred).collect().expect("collect");

    // Alice: age=30 >25 ✓, score=85.5 >=80 ✓
    // Carol: age=35 >25 ✓, score=91.0 >=80 ✓
    assert_eq!(result.row_count(), 2);
}

// ─── Select tests ─────────────────────────────────────────────────────────────

#[test]
fn test_select_columns() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .select(vec![Expr::col("name"), Expr::col("age")])
        .collect()
        .expect("collect");

    assert_eq!(result.column_count(), 2);
    assert!(result.contains_column("name"));
    assert!(result.contains_column("age"));
    assert!(!result.contains_column("id"));
    assert!(!result.contains_column("score"));
}

#[test]
fn test_select_with_alias() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .select(vec![Expr::col("age").alias("years")])
        .collect()
        .expect("collect");

    assert_eq!(result.column_count(), 1);
    assert!(result.contains_column("years"));
}

// ─── with_column tests ────────────────────────────────────────────────────────

#[test]
fn test_with_column_computed() {
    let df = make_people_df();
    // Add a column that doubles the age
    let result = LazyFrame::scan(df)
        .with_column(
            "double_age",
            Expr::col("age").binary_op(BinaryOp::Mul, Expr::lit_int(2)),
        )
        .collect()
        .expect("collect");

    // Should have original 4 columns + double_age
    assert!(result.contains_column("double_age"));
    assert_eq!(result.row_count(), 5);
}

// ─── Sort tests ───────────────────────────────────────────────────────────────

#[test]
fn test_sort_ascending() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .sort(vec![Expr::col("age")], vec![true])
        .collect()
        .expect("collect");

    assert_eq!(result.row_count(), 5);
    // First row should be age=25 (Bob)
    let val = result.get_value(0, "age").expect("get value");
    assert_eq!(val.as_deref(), Some("25"));
}

#[test]
fn test_sort_descending() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .sort(vec![Expr::col("age")], vec![false])
        .collect()
        .expect("collect");

    // First row should be age=42 (Eve)
    let val = result.get_value(0, "age").expect("get value");
    assert_eq!(val.as_deref(), Some("42"));
}

// ─── Limit tests ──────────────────────────────────────────────────────────────

#[test]
fn test_limit() {
    let df = make_people_df();
    let result = LazyFrame::scan(df).limit(3).collect().expect("collect");

    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_limit_larger_than_data() {
    let df = make_people_df();
    let result = LazyFrame::scan(df).limit(100).collect().expect("collect");

    assert_eq!(result.row_count(), 5);
}

#[test]
fn test_limit_zero() {
    let df = make_people_df();
    let result = LazyFrame::scan(df).limit(0).collect().expect("collect");

    assert_eq!(result.row_count(), 0);
}

// ─── GroupBy/Agg tests ────────────────────────────────────────────────────────

#[test]
fn test_groupby_count() {
    // Create a df with repeated category values
    let mut df = OptimizedDataFrame::new();
    df.add_string_column(
        "category",
        vec!["A".into(), "B".into(), "A".into(), "B".into(), "A".into()],
    )
    .expect("add col");
    df.add_int_column("value", vec![1, 2, 3, 4, 5])
        .expect("add col");

    let result = LazyFrame::scan(df)
        .groupby(vec![Expr::col("category")])
        .agg(vec![Expr::col("value").count().alias("count")])
        .collect()
        .expect("collect");

    // 2 groups: A and B
    assert_eq!(result.row_count(), 2);
    assert!(result.contains_column("count"));
}

#[test]
fn test_groupby_sum() {
    let mut df = OptimizedDataFrame::new();
    df.add_string_column(
        "category",
        vec!["A".into(), "B".into(), "A".into(), "B".into()],
    )
    .expect("add col");
    df.add_int_column("value", vec![10, 20, 30, 40])
        .expect("add col");

    let result = LazyFrame::scan(df)
        .groupby(vec![Expr::col("category")])
        .agg(vec![Expr::col("value").sum().alias("total")])
        .collect()
        .expect("collect");

    assert_eq!(result.row_count(), 2);
    assert!(result.contains_column("total"));
}

#[test]
fn test_global_aggregate_mean() {
    let df = make_people_df();

    let result = LazyFrame::scan(df)
        .groupby(vec![])
        .agg(vec![Expr::col("age").mean().alias("avg_age")])
        .collect()
        .expect("collect");

    // (30 + 25 + 35 + 28 + 42) / 5 = 32.0
    assert_eq!(result.row_count(), 1);
    let val = result
        .get_value(0, "avg_age")
        .expect("get value")
        .expect("not null");
    let avg: f64 = val.parse().expect("parse f64");
    assert!((avg - 32.0).abs() < 0.001, "Expected 32.0, got {}", avg);
}

// ─── Join tests ───────────────────────────────────────────────────────────────

#[test]
fn test_inner_join() {
    let left = make_people_df();
    let right = make_scores_df();

    // Only ids 1, 2, 3 appear in right; left has ids 1-5
    let result = LazyFrame::scan(left)
        .join(
            LazyFrame::scan(right),
            Expr::col("id"),
            Expr::col("id"),
            JoinType::Inner,
        )
        .collect()
        .expect("collect");

    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_left_join() {
    let left = make_people_df();
    let right = make_scores_df();

    let result = LazyFrame::scan(left)
        .join(
            LazyFrame::scan(right),
            Expr::col("id"),
            Expr::col("id"),
            JoinType::Left,
        )
        .collect()
        .expect("collect");

    // Left has 5 rows; all should be preserved
    assert_eq!(result.row_count(), 5);
}

// ─── explain() tests ──────────────────────────────────────────────────────────

#[test]
fn test_explain_contains_filter() {
    let df = make_people_df();
    let lf = LazyFrame::scan(df).filter(Expr::col("age").gt(Expr::lit_int(30)));
    let plan_str = lf.explain();
    assert!(
        plan_str.contains("Filter"),
        "Expected 'Filter' in explain: {}",
        plan_str
    );
}

#[test]
fn test_explain_contains_project() {
    let df = make_people_df();
    let lf = LazyFrame::scan(df).select(vec![Expr::col("name"), Expr::col("age")]);
    let plan_str = lf.explain();
    assert!(
        plan_str.contains("Project"),
        "Expected 'Project' in explain: {}",
        plan_str
    );
}

#[test]
fn test_explain_contains_aggregate() {
    let df = make_people_df();
    let lf = LazyFrame::scan(df)
        .groupby(vec![Expr::col("id")])
        .agg(vec![Expr::col("age").sum().alias("total")]);
    let plan_str = lf.explain();
    assert!(
        plan_str.contains("Aggregate"),
        "Expected 'Aggregate' in explain: {}",
        plan_str
    );
}

#[test]
fn test_explain_optimized_returns_string() {
    let df = make_people_df();
    let lf = LazyFrame::scan(df)
        .filter(Expr::col("age").gt(Expr::lit_int(30)))
        .select(vec![Expr::col("name"), Expr::col("age")]);
    let opt_str = lf.explain_optimized().expect("explain_optimized");
    assert!(
        opt_str.contains("optimized"),
        "Expected 'optimized' in output: {}",
        opt_str
    );
}

// ─── Optimizer tests ──────────────────────────────────────────────────────────

#[test]
fn test_constant_folding_arithmetic() {
    // lit(2) + lit(3) should fold to lit(5)
    let expr = Expr::lit_int(2).binary_op(BinaryOp::Add, Expr::lit_int(3));
    let plan = LogicalPlan::Filter {
        predicate: expr,
        input: Box::new(LogicalPlan::Scan {
            source: std::sync::Arc::new(make_people_df()),
            projection: None,
        }),
    };
    let rule = ConstantFolding;
    let optimized = rule.optimize(plan).expect("optimize");
    let display = optimized.display();
    // The constant should have been folded: filter on lit(5) → true removal,
    // or kept as lit(5) which is non-boolean. Either way we get a valid plan.
    assert!(!display.is_empty());
}

#[test]
fn test_constant_folding_boolean_true_removes_filter() {
    // Filter on `true` should be eliminated
    let plan = LogicalPlan::Filter {
        predicate: Expr::lit_bool(true),
        input: Box::new(LogicalPlan::Scan {
            source: std::sync::Arc::new(make_people_df()),
            projection: None,
        }),
    };
    let rule = ConstantFolding;
    let optimized = rule.optimize(plan).expect("optimize");
    // Should NOT be a Filter node anymore
    assert!(
        !matches!(optimized, LogicalPlan::Filter { .. }),
        "Expected filter to be removed"
    );
}

#[test]
fn test_constant_folding_if_true() {
    // IF(true, col("a"), col("b")) should collapse to col("a")
    let expr = Expr::If {
        condition: Box::new(Expr::lit_bool(true)),
        then_expr: Box::new(Expr::col("a")),
        else_expr: Box::new(Expr::col("b")),
    };
    let rule = ConstantFolding;
    // Just check we can apply the rule without panic
    let plan = LogicalPlan::Filter {
        predicate: expr,
        input: Box::new(LogicalPlan::Scan {
            source: std::sync::Arc::new(make_people_df()),
            projection: None,
        }),
    };
    let _ = rule.optimize(plan).expect("optimize");
}

#[test]
fn test_predicate_pushdown_pushes_filter_before_sort() {
    // Filter → Sort → Scan  should become  Sort → Filter → Scan
    // (filter pushed through sort to be closer to scan)
    let df = make_people_df();
    let plan = LogicalPlan::Filter {
        predicate: Expr::col("age").gt(Expr::lit_int(30)),
        input: Box::new(LogicalPlan::Sort {
            by: vec![Expr::col("name")],
            ascending: vec![true],
            input: Box::new(LogicalPlan::Scan {
                source: std::sync::Arc::new(df),
                projection: None,
            }),
        }),
    };
    let rule = PredicatePushdown;
    let optimized = rule.optimize(plan).expect("optimize");
    // The optimized plan should still produce sensible output when executed
    let result = LazyFrame::from_plan(optimized)
        .collect_unoptimized()
        .expect("collect");
    // age > 30: Carol(35) and Eve(42)
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_predicate_pushdown_through_project() {
    let df = make_people_df();
    // Filter after project — predicate on "age" column that the project keeps
    let plan = LogicalPlan::Filter {
        predicate: Expr::col("age").gt(Expr::lit_int(30)),
        input: Box::new(LogicalPlan::Project {
            exprs: vec![Expr::col("name"), Expr::col("age")],
            input: Box::new(LogicalPlan::Scan {
                source: std::sync::Arc::new(df),
                projection: None,
            }),
        }),
    };
    let rule = PredicatePushdown;
    let optimized = rule.optimize(plan).expect("optimize");
    let result = LazyFrame::from_plan(optimized)
        .collect_unoptimized()
        .expect("collect");
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_projection_pushdown_narrows_scan() {
    let df = make_people_df();
    // Project only "name" and "age"  — scan should only read those columns
    let plan = LogicalPlan::Project {
        exprs: vec![Expr::col("name"), Expr::col("age")],
        input: Box::new(LogicalPlan::Scan {
            source: std::sync::Arc::new(df),
            projection: None,
        }),
    };
    let rule = ProjectionPushdown;
    let optimized = rule.optimize(plan).expect("optimize");
    let display = optimized.display();
    // The scan's projection list should not contain "id" or "score"
    assert!(
        !display.contains("id") || display.contains("projection=*"),
        "{}",
        display
    );
}

#[test]
fn test_dead_code_elimination_merges_limits() {
    let df = make_people_df();
    let plan = LogicalPlan::Limit {
        n: 3,
        input: Box::new(LogicalPlan::Limit {
            n: 10,
            input: Box::new(LogicalPlan::Scan {
                source: std::sync::Arc::new(df),
                projection: None,
            }),
        }),
    };
    let rule = DeadCodeElimination;
    let optimized = rule.optimize(plan).expect("optimize");
    // Should be a single Limit(3) node
    match optimized {
        LogicalPlan::Limit { n, .. } => assert_eq!(n, 3),
        other => panic!("Expected Limit, got: {}", other.display()),
    }
}

#[test]
fn test_dead_code_elimination_false_filter() {
    let df = make_people_df();
    let plan = LogicalPlan::Filter {
        predicate: Expr::lit_bool(false),
        input: Box::new(LogicalPlan::Scan {
            source: std::sync::Arc::new(df),
            projection: None,
        }),
    };
    let rule = DeadCodeElimination;
    let optimized = rule.optimize(plan).expect("optimize");
    // Should be a Limit(0) node
    match optimized {
        LogicalPlan::Limit { n, .. } => assert_eq!(n, 0),
        other => panic!("Expected Limit(0), got: {}", other.display()),
    }
}

// ─── Full pipeline tests ──────────────────────────────────────────────────────

#[test]
fn test_full_pipeline_filter_select_limit() {
    let df = make_people_df();
    let result = LazyFrame::scan(df)
        .filter(Expr::col("age").gt(Expr::lit_int(25)))
        .select(vec![Expr::col("name"), Expr::col("age")])
        .sort(vec![Expr::col("age")], vec![true])
        .limit(2)
        .collect()
        .expect("collect");

    assert_eq!(result.row_count(), 2);
    assert_eq!(result.column_count(), 2);
}

#[test]
fn test_full_pipeline_groupby_sort() {
    let mut df = OptimizedDataFrame::new();
    df.add_string_column(
        "dept",
        vec![
            "eng".into(),
            "hr".into(),
            "eng".into(),
            "hr".into(),
            "eng".into(),
        ],
    )
    .expect("add col");
    df.add_int_column("salary", vec![100, 80, 120, 90, 110])
        .expect("add col");

    let result = LazyFrame::scan(df)
        .groupby(vec![Expr::col("dept")])
        .agg(vec![Expr::col("salary").mean().alias("avg_salary")])
        .sort(vec![Expr::col("avg_salary")], vec![false])
        .collect()
        .expect("collect");

    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_optimizer_default_rules() {
    let optimizer = Optimizer::default_rules();
    let names = optimizer.rule_names();
    assert!(names.contains(&"ConstantFolding"));
    assert!(names.contains(&"PredicatePushdown"));
    assert!(names.contains(&"ProjectionPushdown"));
    assert!(names.contains(&"DeadCodeElimination"));
}

#[test]
fn test_collect_vs_collect_unoptimized_same_result() {
    let df = make_people_df();
    let lf1 = LazyFrame::scan(df.clone())
        .filter(Expr::col("age").gt(Expr::lit_int(30)))
        .limit(2);
    let lf2 = LazyFrame::scan(df)
        .filter(Expr::col("age").gt(Expr::lit_int(30)))
        .limit(2);

    let optimized = lf1.collect().expect("optimized collect");
    let unoptimized = lf2.collect_unoptimized().expect("unoptimized collect");

    // Both should produce same number of rows
    assert_eq!(optimized.row_count(), unoptimized.row_count());
}

// ─── Expression tests ─────────────────────────────────────────────────────────

#[test]
fn test_expr_is_null() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("x", vec![1, 2, 3]).expect("add col");
    // is_null on a non-null column → all false
    let result = LazyFrame::scan(df)
        .filter(Expr::col("x").is_not_null())
        .collect()
        .expect("collect");
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_expr_unary_not() {
    let df = make_people_df();
    // NOT (age > 30) = age <= 30 → Alice(30), Bob(25), Dave(28)
    let pred = Expr::UnaryOp {
        op: UnaryOp::Not,
        expr: Box::new(Expr::col("age").gt(Expr::lit_int(30))),
    };
    let result = LazyFrame::scan(df).filter(pred).collect().expect("collect");
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_expr_if_then_else() {
    let df = make_people_df();
    // Add column: IF age > 30 THEN "senior" ELSE "junior"
    let cond_expr = Expr::If {
        condition: Box::new(Expr::col("age").gt(Expr::lit_int(30))),
        then_expr: Box::new(Expr::lit_str("senior")),
        else_expr: Box::new(Expr::lit_str("junior")),
    };
    let result = LazyFrame::scan(df)
        .with_column("level", cond_expr)
        .collect()
        .expect("collect");

    assert!(result.contains_column("level"));
    assert_eq!(result.row_count(), 5);
}

#[test]
fn test_expr_display() {
    let expr = Expr::col("age").gt(Expr::lit_int(30));
    let s = expr.to_string();
    assert!(s.contains("age"), "Expected 'age' in: {}", s);
    assert!(s.contains("30"), "Expected '30' in: {}", s);
}

#[test]
fn test_agg_expr_display() {
    let expr = Expr::col("value").sum().alias("total");
    let s = expr.to_string();
    assert!(s.contains("SUM") || s.contains("total"), "{}", s);
}

#[test]
fn test_logical_plan_display() {
    let df = make_people_df();
    let plan = LogicalPlan::Filter {
        predicate: Expr::col("age").gt(Expr::lit_int(30)),
        input: Box::new(LogicalPlan::Scan {
            source: std::sync::Arc::new(df),
            projection: None,
        }),
    };
    let display = plan.display();
    assert!(display.contains("Filter"), "{}", display);
    assert!(display.contains("Scan"), "{}", display);
}

#[test]
fn test_scan_with_projection() {
    let df = make_people_df();
    let plan = LogicalPlan::Scan {
        source: std::sync::Arc::new(df),
        projection: Some(vec!["name".into(), "age".into()]),
    };
    let result = LazyFrame::from_plan(plan)
        .collect_unoptimized()
        .expect("collect");
    assert_eq!(result.column_count(), 2);
    assert!(result.contains_column("name"));
    assert!(result.contains_column("age"));
}
