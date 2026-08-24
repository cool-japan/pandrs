//! Regression tests for the DataFrame query engine (`df.query()` / `df.eval()`).
//!
//! Historically `df.query("a > 1 && b < 2")` returned **every** row: the JIT
//! path refused to compile `And`/`Or`/`Unary` nodes and the fallback produced an
//! all-`true` mask. These tests assert on the exact surviving rows so that a
//! regression to "no filter applied" cannot pass.

use pandrs::dataframe::query::{
    Evaluator, JitEvaluator, Lexer, LiteralValue, OptimizedEvaluator, Parser, QueryContext,
    QueryEngine, QueryExt, Token,
};
use pandrs::{DataFrame, Series};

/// Test-local result type: the library error is large, and helper functions in
/// tests only ever need its message.
type TestResult<T> = std::result::Result<T, String>;

/// Add a typed column to `df`.
fn column<T: 'static + std::fmt::Debug + Clone + Send + Sync>(
    df: &mut DataFrame,
    name: &str,
    values: Vec<T>,
) -> TestResult<()> {
    let series = Series::new(values, Some(name.to_string())).map_err(|e| e.to_string())?;
    df.add_column(name.to_string(), series)
        .map_err(|e| e.to_string())
}

/// Build the shared fixture:
///
/// | id | age | salary  | dept | active |
/// |----|-----|---------|------|--------|
/// | 0  | 25  | 50000.0 | IT   | true   |
/// | 1  | 35  | 65000.0 | HR   | false  |
/// | 2  | 45  | 85000.0 | IT   | true   |
/// | 3  | 30  | 60000.0 | Fin  | false  |
/// | 4  | 55  | 95000.0 | HR   | true   |
fn fixture() -> TestResult<DataFrame> {
    let mut df = DataFrame::new();
    column(&mut df, "id", vec![0i64, 1, 2, 3, 4])?;
    column(&mut df, "age", vec![25i64, 35, 45, 30, 55])?;
    column(
        &mut df,
        "salary",
        vec![50000.0f64, 65000.0, 85000.0, 60000.0, 95000.0],
    )?;
    column(
        &mut df,
        "dept",
        vec![
            "IT".to_string(),
            "HR".to_string(),
            "IT".to_string(),
            "Fin".to_string(),
            "HR".to_string(),
        ],
    )?;
    column(&mut df, "active", vec![true, false, true, false, true])?;
    Ok(df)
}

/// Ids surviving a query, in order.
fn ids(df: &DataFrame) -> TestResult<Vec<i64>> {
    Ok(df
        .get_column_string_values("id")
        .map_err(|e| e.to_string())?
        .iter()
        .map(|s| s.trim().parse::<i64>().unwrap_or(-1))
        .collect())
}

fn query_ids(df: &DataFrame, expr: &str) -> TestResult<Vec<i64>> {
    let filtered = df.query(expr).map_err(|e| e.to_string())?;
    ids(&filtered)
}

// ---------------------------------------------------------------------------
// 1. Compound boolean expressions must actually filter
// ---------------------------------------------------------------------------

#[test]
fn compound_and_does_not_return_every_row() {
    let df = fixture().expect("fixture");
    // age > 30 -> ids 1,2,4 ; salary < 90000 -> ids 0,1,2,3 ; AND -> 1,2
    let got = query_ids(&df, "age > 30 && salary < 90000").expect("query");
    assert_eq!(got, vec![1, 2], "&& must filter, not return every row");
    assert_ne!(got.len(), df.row_count(), "all-true mask regression");
}

#[test]
fn compound_or_filters_correctly() {
    let df = fixture().expect("fixture");
    // age < 28 -> 0 ; salary > 90000 -> 4
    let got = query_ids(&df, "age < 28 || salary > 90000").expect("query");
    assert_eq!(got, vec![0, 4]);
}

#[test]
fn pandas_single_ampersand_and_pipe_are_logical_operators() {
    let df = fixture().expect("fixture");
    assert_eq!(
        query_ids(&df, "age > 30 & salary < 90000").expect("single &"),
        vec![1, 2]
    );
    assert_eq!(
        query_ids(&df, "age < 28 | salary > 90000").expect("single |"),
        vec![0, 4]
    );
}

#[test]
fn logical_not_forms_all_work() {
    let df = fixture().expect("fixture");
    // dept == 'IT' -> 0,2 ; negated -> 1,3,4
    assert_eq!(
        query_ids(&df, "!(dept == 'IT')").expect("! form"),
        vec![1, 3, 4]
    );
    assert_eq!(
        query_ids(&df, "not (dept == 'IT')").expect("not form"),
        vec![1, 3, 4]
    );
    assert_eq!(
        query_ids(&df, "~(dept == 'IT')").expect("~ form"),
        vec![1, 3, 4]
    );
}

#[test]
fn nested_parentheses_and_mixed_operators() {
    let df = fixture().expect("fixture");
    // (age > 30 && salary > 80000) -> 2,4 ; (dept == 'Fin') -> 3
    let got = query_ids(&df, "(age > 30 && salary > 80000) || (dept == 'Fin')").expect("query");
    assert_eq!(got, vec![2, 3, 4]);

    // Deeper nesting with a negation inside.
    let got = query_ids(&df, "((age >= 30) & ~(dept == 'HR')) & (salary <= 85000)").expect("query");
    assert_eq!(got, vec![2, 3]);
}

#[test]
fn arithmetic_inside_comparison_combined_with_logic() {
    let df = fixture().expect("fixture");
    // salary / age > 1800 -> 0:2000, 1:1857.1, 2:1888.9, 3:2000, 4:1727.3 -> 0,1,2,3
    // and age < 40 -> 0,1,3
    let got = query_ids(&df, "salary / age > 1800 && age < 40").expect("query");
    assert_eq!(got, vec![0, 1, 3]);
}

// ---------------------------------------------------------------------------
// 2. Literals and string comparisons
// ---------------------------------------------------------------------------

#[test]
fn string_equality_and_inequality() {
    let df = fixture().expect("fixture");
    assert_eq!(query_ids(&df, "dept == 'IT'").expect("=="), vec![0, 2]);
    assert_eq!(
        query_ids(&df, "dept != 'IT'").expect("!="),
        vec![1, 3, 4],
        "string != must not error or return all rows"
    );
    assert_eq!(
        query_ids(&df, "dept == \"HR\" || dept == \"Fin\"").expect("double quotes"),
        vec![1, 3, 4]
    );
}

#[test]
fn boolean_literals_lowercase_and_capitalized() {
    let df = fixture().expect("fixture");
    assert_eq!(
        query_ids(&df, "active == true").expect("true"),
        vec![0, 2, 4]
    );
    assert_eq!(
        query_ids(&df, "active == false").expect("false"),
        vec![1, 3]
    );
    assert_eq!(
        query_ids(&df, "active == True").expect("True"),
        vec![0, 2, 4],
        "pandas-style capitalized True must lex as a boolean literal"
    );
    assert_eq!(
        query_ids(&df, "active == False").expect("False"),
        vec![1, 3]
    );
}

#[test]
fn bare_boolean_column_is_a_valid_mask() {
    let df = fixture().expect("fixture");
    assert_eq!(query_ids(&df, "active").expect("bare bool"), vec![0, 2, 4]);
    assert_eq!(query_ids(&df, "~active").expect("negated bool"), vec![1, 3]);
    assert_eq!(
        query_ids(&df, "active & (age > 30)").expect("bool column in compound"),
        vec![2, 4]
    );
}

#[test]
fn literal_only_queries_select_all_or_nothing() {
    let df = fixture().expect("fixture");
    assert_eq!(query_ids(&df, "true").expect("true"), vec![0, 1, 2, 3, 4]);
    assert!(query_ids(&df, "false").expect("false").is_empty());
}

// ---------------------------------------------------------------------------
// 3. dtype preservation
// ---------------------------------------------------------------------------

#[test]
fn query_preserves_column_dtypes() {
    let df = fixture().expect("fixture");
    let filtered = df.query("age > 30").expect("query");
    assert_eq!(filtered.row_count(), 3);
    assert_eq!(filtered.column_names(), df.column_names());

    // Every column keeps its concrete Series type instead of being rebuilt as
    // Series<String>.
    let age = filtered
        .get_column::<i64>("age")
        .expect("age must stay Series<i64>");
    assert_eq!(age.values(), &[35i64, 45, 55]);

    let salary = filtered
        .get_column::<f64>("salary")
        .expect("salary must stay Series<f64>");
    assert_eq!(salary.values(), &[65000.0f64, 85000.0, 95000.0]);

    let dept = filtered
        .get_column::<String>("dept")
        .expect("dept must stay Series<String>");
    assert_eq!(
        dept.values(),
        &["HR".to_string(), "IT".to_string(), "HR".to_string()]
    );

    let active = filtered
        .get_column::<bool>("active")
        .expect("active must stay Series<bool>");
    assert_eq!(active.values(), &[false, true, true]);
}

#[test]
fn query_preserves_narrow_numeric_dtypes() {
    let mut df = DataFrame::new();
    df.add_column(
        "small".to_string(),
        Series::new(vec![1i32, 2, 3], Some("small".to_string())).expect("series"),
    )
    .expect("add");
    df.add_column(
        "tiny".to_string(),
        Series::new(vec![1.5f32, 2.5, 3.5], Some("tiny".to_string())).expect("series"),
    )
    .expect("add");
    df.add_column(
        "count".to_string(),
        Series::new(vec![10u32, 20, 30], Some("count".to_string())).expect("series"),
    )
    .expect("add");

    let filtered = df.query("small > 1").expect("query");
    assert_eq!(filtered.row_count(), 2);
    assert_eq!(
        filtered
            .get_column::<i32>("small")
            .expect("i32 preserved")
            .values(),
        &[2i32, 3]
    );
    assert_eq!(
        filtered
            .get_column::<f32>("tiny")
            .expect("f32 preserved")
            .values(),
        &[2.5f32, 3.5]
    );
    assert_eq!(
        filtered
            .get_column::<u32>("count")
            .expect("u32 preserved")
            .values(),
        &[20u32, 30]
    );
}

#[test]
fn query_preserves_datetime_columns() {
    use chrono::{NaiveDate, NaiveDateTime};

    let stamp = |day: u32| -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, day)
            .expect("date")
            .and_hms_opt(12, 0, 0)
            .expect("time")
    };

    let mut df = DataFrame::new();
    df.add_column(
        "n".to_string(),
        Series::new(vec![1i64, 2, 3], Some("n".to_string())).expect("series"),
    )
    .expect("add");
    df.add_column(
        "when".to_string(),
        Series::new(vec![stamp(1), stamp(2), stamp(3)], Some("when".to_string())).expect("series"),
    )
    .expect("add");

    let filtered = df.query("n > 1").expect("query");
    assert_eq!(filtered.row_count(), 2);
    assert_eq!(
        filtered
            .get_column::<NaiveDateTime>("when")
            .expect("datetime column must stay Series<NaiveDateTime>")
            .values(),
        &[stamp(2), stamp(3)]
    );
}

#[test]
fn query_reports_unsupported_element_types() {
    // A column whose element type cannot be represented at all must produce a
    // clear error rather than placeholder values in the filtered frame.
    #[derive(Debug, Clone, PartialEq)]
    struct Custom(u8);

    let mut df = DataFrame::new();
    df.add_column(
        "n".to_string(),
        Series::new(vec![1i64, 2, 3], Some("n".to_string())).expect("series"),
    )
    .expect("add");
    df.add_column(
        "custom".to_string(),
        Series::new(
            vec![Custom(1), Custom(2), Custom(3)],
            Some("custom".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    let result = df.query("n > 1");
    match result {
        Ok(filtered) => panic!(
            "expected an error for an unrepresentable column type, got {} rows",
            filtered.row_count()
        ),
        Err(err) => {
            let message = err.to_string();
            assert!(
                message.contains("custom"),
                "error must name the offending column: {message}"
            );
        }
    }
}

#[test]
fn query_preserves_row_index_labels() {
    let mut df = DataFrame::new();
    df.add_column(
        "n".to_string(),
        Series::new(vec![1i64, 2, 3, 4], Some("n".to_string())).expect("series"),
    )
    .expect("add");
    let index = pandrs::index::Index::new(vec![
        "w".to_string(),
        "x".to_string(),
        "y".to_string(),
        "z".to_string(),
    ])
    .expect("index");
    df.set_index(index).expect("set_index");

    let filtered = df.query("n > 2").expect("query");
    assert_eq!(filtered.row_count(), 2);
    let labels = filtered
        .get_index()
        .string_values()
        .expect("filtered frame keeps a simple index");
    assert_eq!(labels, vec!["y".to_string(), "z".to_string()]);
}

// ---------------------------------------------------------------------------
// 4. String-typed (CSV-like) frames
// ---------------------------------------------------------------------------

#[test]
fn numeric_comparisons_work_on_string_columns() {
    // `read_csv` produces all-String columns; numeric predicates must still work.
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(
            vec!["0".to_string(), "1".to_string(), "2".to_string()],
            Some("id".to_string()),
        )
        .expect("series"),
    )
    .expect("add");
    df.add_column(
        "value".to_string(),
        Series::new(
            vec!["10".to_string(), "20".to_string(), "30".to_string()],
            Some("value".to_string()),
        )
        .expect("series"),
    )
    .expect("add");
    df.add_column(
        "flag".to_string(),
        Series::new(
            vec!["true".to_string(), "false".to_string(), "true".to_string()],
            Some("flag".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    assert_eq!(query_ids(&df, "value > 15").expect("numeric"), vec![1, 2]);
    assert_eq!(
        query_ids(&df, "value > 15 && flag == true").expect("mixed"),
        vec![2]
    );
}

/// Build a CSV-shaped frame (every column is `Series<String>`).
fn text_frame(values: Vec<&str>) -> DataFrame {
    let mut df = DataFrame::new();
    let ids: Vec<String> = (0..values.len()).map(|i| i.to_string()).collect();
    df.add_column(
        "id".to_string(),
        Series::new(ids, Some("id".to_string())).expect("series"),
    )
    .expect("add");
    df.add_column(
        "value".to_string(),
        Series::new(
            values.iter().map(|v| (*v).to_string()).collect::<Vec<_>>(),
            Some("value".to_string()),
        )
        .expect("series"),
    )
    .expect("add");
    df
}

#[test]
fn missing_values_never_satisfy_a_predicate() {
    let df = text_frame(vec!["10", "", "30"]);

    assert_eq!(query_ids(&df, "value > 5").expect("gt"), vec![0, 2]);
    assert_eq!(
        query_ids(&df, "value < 5").expect("lt"),
        Vec::<i64>::new(),
        "a missing value must not satisfy the opposite predicate either"
    );
    assert_eq!(query_ids(&df, "value == 10").expect("eq"), vec![0]);

    // The same holds for the row-by-row tier.
    let row_by_row = QueryContext::with_jit_settings(false, 100);
    let filtered = df
        .query_with_context("value > 5", &row_by_row)
        .expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![0, 2]);
}

#[test]
fn missing_values_propagate_through_eval_instead_of_becoming_zero() {
    let df = text_frame(vec!["10", "", "30"]);
    let doubled = df.eval("value * 2", "doubled").expect("eval");
    let values = doubled
        .get_column::<f64>("doubled")
        .expect("numeric result")
        .values();
    assert_eq!(values[0], 20.0);
    assert!(
        values[1].is_nan(),
        "a missing input must stay missing (got {}), never 0",
        values[1]
    );
    assert_eq!(values[2], 60.0);
}

#[test]
fn na_markers_in_a_numeric_column_are_missing_values() {
    let df = text_frame(vec!["1", "NA", "3"]);
    assert_eq!(query_ids(&df, "value >= 1").expect("ge"), vec![0, 2]);
}

#[test]
fn na_text_in_a_textual_column_stays_text() {
    // The column is not numeric, so 'NA' is just a string and compares as one.
    let df = text_frame(vec!["IT", "NA", "HR"]);
    assert_eq!(query_ids(&df, "value == 'NA'").expect("eq"), vec![1]);

    let row_by_row = QueryContext::with_jit_settings(false, 100);
    let filtered = df
        .query_with_context("value == 'NA'", &row_by_row)
        .expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![1]);
}

#[test]
fn string_ordering_comparison() {
    let df = fixture().expect("fixture");
    // 'Fin' < 'HR' < 'IT' lexicographically.
    assert_eq!(query_ids(&df, "dept < 'HR'").expect("lt"), vec![3]);
    assert_eq!(query_ids(&df, "dept >= 'IT'").expect("ge"), vec![0, 2]);
}

// ---------------------------------------------------------------------------
// 5. Errors must be errors (never a silent all-true mask)
// ---------------------------------------------------------------------------

#[test]
fn unknown_column_is_an_error() {
    let df = fixture().expect("fixture");
    let result = df.query("does_not_exist > 10");
    assert!(
        result.is_err(),
        "unknown column must error, never return every row"
    );
    let result = df.query("age > 30 && does_not_exist > 10");
    assert!(
        result.is_err(),
        "unknown column inside a compound expression must error"
    );
}

#[test]
fn non_boolean_query_is_an_error() {
    let df = fixture().expect("fixture");
    assert!(
        df.query("age && salary").is_err(),
        "numeric operands for && must error, not be coerced"
    );
    assert!(
        df.query("age + 1").is_err(),
        "a numeric expression is not a valid mask"
    );
}

#[test]
fn syntax_errors_are_reported() {
    let df = fixture().expect("fixture");
    assert!(df.query("age >").is_err(), "truncated expression");
    assert!(df.query("age $ 3").is_err(), "unknown character");
    assert!(df.query("(age > 3").is_err(), "unbalanced parenthesis");
}

#[test]
fn unknown_function_is_an_error() {
    let df = fixture().expect("fixture");
    assert!(df.query("mystery(age) > 1").is_err());
}

// ---------------------------------------------------------------------------
// 6. Context variables
// ---------------------------------------------------------------------------

#[test]
fn context_variables_resolve_for_unknown_identifiers() {
    let df = fixture().expect("fixture");
    let mut engine = QueryEngine::new();
    engine.set_variable("threshold".to_string(), LiteralValue::Number(40.0));
    let filtered = engine.query(&df, "age > threshold").expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![2, 4]);
}

#[test]
fn at_prefixed_variables_resolve() {
    let df = fixture().expect("fixture");
    let mut context = QueryContext::new();
    context.set_variable("min_salary".to_string(), LiteralValue::Number(80000.0));
    context.set_variable("target".to_string(), LiteralValue::String("IT".to_string()));
    let filtered = df
        .query_with_context("salary >= @min_salary && dept == @target", &context)
        .expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![2]);
}

#[test]
fn undefined_variable_is_an_error() {
    let df = fixture().expect("fixture");
    assert!(df.query("age > @nope").is_err());
    assert!(df.query("age > nope").is_err());
}

#[test]
fn column_shadows_nothing_when_variable_has_same_name() {
    // A real column always wins over a variable of the same name (the variable
    // lookup is only a fallback for identifiers that are not columns).
    let df = fixture().expect("fixture");
    let mut context = QueryContext::new();
    context.set_variable("age".to_string(), LiteralValue::Number(0.0));
    let filtered = df.query_with_context("age > 40", &context).expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![2, 4]);
}

// ---------------------------------------------------------------------------
// 7. Functions
// ---------------------------------------------------------------------------

#[test]
fn builtin_functions_in_predicates() {
    let df = fixture().expect("fixture");
    // sqrt(salary) > 300 -> salary > 90000 -> id 4
    assert_eq!(query_ids(&df, "sqrt(salary) > 300").expect("sqrt"), vec![4]);
    // abs applied to a negative expression: |age - 45| is 20,10,0,15,10
    assert_eq!(
        query_ids(&df, "abs(age - 45) <= 10").expect("abs"),
        vec![1, 2, 4]
    );
}

#[test]
fn custom_functions_are_usable() {
    let df = fixture().expect("fixture");
    let mut engine = QueryEngine::new();
    engine.add_function("double".to_string(), |args| {
        args.first().copied().unwrap_or(0.0) * 2.0
    });
    let filtered = engine.query(&df, "double(age) > 80").expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![2, 4]);
}

// ---------------------------------------------------------------------------
// 8. eval()
// ---------------------------------------------------------------------------

#[test]
fn eval_produces_typed_numeric_column() {
    let df = fixture().expect("fixture");
    let with_ratio = df.eval("salary / age", "ratio").expect("eval");
    assert_eq!(with_ratio.row_count(), df.row_count());
    let ratio = with_ratio
        .get_column::<f64>("ratio")
        .expect("numeric eval result must be Series<f64>, not Series<String>");
    assert_eq!(ratio.len(), 5);
    assert!((ratio.values()[0] - 2000.0).abs() < 1e-9);
}

#[test]
fn eval_produces_typed_boolean_column() {
    let df = fixture().expect("fixture");
    let flagged = df
        .eval("age > 30 && active == true", "senior")
        .expect("eval");
    let senior = flagged
        .get_column::<bool>("senior")
        .expect("boolean eval result must be Series<bool>");
    assert_eq!(senior.values(), &[false, false, true, false, true]);
}

#[test]
fn eval_produces_string_column_for_string_expressions() {
    let df = fixture().expect("fixture");
    let joined = df.eval("dept + '-x'", "tag").expect("eval");
    let tag = joined
        .get_column::<String>("tag")
        .expect("string eval result must be Series<String>");
    assert_eq!(tag.values()[0], "IT-x".to_string());
}

#[test]
fn eval_preserves_source_dtypes() {
    let df = fixture().expect("fixture");
    let extended = df.eval("age * 2", "double_age").expect("eval");
    assert_eq!(
        extended
            .get_column::<i64>("age")
            .expect("source i64 column survives eval")
            .values(),
        &[25i64, 35, 45, 30, 55]
    );
}

#[test]
fn eval_rejects_duplicate_target_column() {
    let df = fixture().expect("fixture");
    assert!(
        df.eval("age * 2", "age").is_err(),
        "overwriting an existing column must be an explicit error"
    );
}

// ---------------------------------------------------------------------------
// 9. Differential test: vectorized path vs the row-by-row interpreter
// ---------------------------------------------------------------------------

fn parse(expr: &'static str) -> TestResult<pandrs::dataframe::query::Expr> {
    let mut lexer = Lexer::new(expr);
    let mut tokens = Vec::new();
    loop {
        let token = lexer.next_token().map_err(|e| e.to_string())?;
        let is_eof = matches!(token, Token::Eof);
        tokens.push(token);
        if is_eof {
            break;
        }
    }
    Parser::new(tokens).parse().map_err(|e| e.to_string())
}

const DIFFERENTIAL_EXPRESSIONS: &[&str] = &[
    "age > 30",
    "age >= 35 && salary < 90000",
    "age < 28 || salary > 90000",
    "dept == 'IT'",
    "dept != 'HR'",
    "active == true",
    "active",
    "~active",
    "!(dept == 'IT')",
    "(age > 30 && salary > 80000) || (dept == 'Fin')",
    "salary / age > 1800 && age < 40",
    "age + 5 > 40",
    "-age < -30",
    "age ** 2 > 1600",
    "sqrt(salary) > 300",
    "age > 30 & salary < 90000",
    "age < 28 | salary > 90000",
    "true",
    "false",
    "age % 2 == 1",
];

#[test]
fn every_path_agrees_with_the_row_by_row_interpreter() {
    let df = fixture().expect("fixture");
    let context = QueryContext::new();

    for expr_str in DIFFERENTIAL_EXPRESSIONS {
        let expr = parse(expr_str).unwrap_or_else(|e| panic!("parse {expr_str}: {e}"));

        let reference = Evaluator::new(&df, &context)
            .evaluate_query(&expr)
            .unwrap_or_else(|e| panic!("interpreter {expr_str}: {e}"));

        let jit = JitEvaluator::new(&df, &context)
            .evaluate_query_jit(&expr)
            .unwrap_or_else(|e| panic!("jit evaluator {expr_str}: {e}"));
        assert_eq!(jit, reference, "JitEvaluator disagrees on `{expr_str}`");

        let optimized = OptimizedEvaluator::new(&df, &context)
            .evaluate_query_vectorized(&expr)
            .unwrap_or_else(|e| panic!("optimized evaluator {expr_str}: {e}"));
        assert_eq!(
            optimized, reference,
            "OptimizedEvaluator disagrees on `{expr_str}`"
        );

        let expected_ids: Vec<i64> = reference
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i as i64) } else { None })
            .collect();
        let got = query_ids(&df, expr_str).unwrap_or_else(|e| panic!("query {expr_str}: {e}"));
        assert_eq!(got, expected_ids, "df.query disagrees on `{expr_str}`");
    }
}

const TEXT_FRAME_EXPRESSIONS: &[&str] = &[
    "value > 5",
    "value < 5",
    "value == 10",
    "value != 10",
    "value >= 10 && value <= 30",
    "value + 1 > 11",
    "value == 'NA'",
    "value == ''",
];

#[test]
fn both_tiers_agree_on_text_columns_with_missing_values() {
    // String columns (what `read_csv` produces) are where the two tiers could
    // most easily disagree, because one classifies per column and the other
    // reads cell by cell.
    for cells in [
        vec!["10", "", "30"],
        vec!["1", "NA", "3"],
        vec!["IT", "NA", "HR"],
        vec!["true", "false", "true"],
    ] {
        let df = text_frame(cells.clone());
        let vectorized = QueryContext::with_jit_settings(true, 1);
        let row_by_row = QueryContext::with_jit_settings(false, 1);

        for expr in TEXT_FRAME_EXPRESSIONS {
            let fast = df.query_with_context(expr, &vectorized);
            let slow = df.query_with_context(expr, &row_by_row);

            match (fast, slow) {
                (Ok(fast), Ok(slow)) => assert_eq!(
                    ids(&fast).expect("ids"),
                    ids(&slow).expect("ids"),
                    "tiers disagree on `{expr}` for {cells:?}"
                ),
                (Err(_), Err(_)) => {}
                (fast, slow) => panic!(
                    "tiers disagree on whether `{expr}` is valid for {cells:?}: \
                     vectorized={:?}, row-by-row={:?}",
                    fast.map(|df| df.row_count()),
                    slow.map(|df| df.row_count())
                ),
            }
        }
    }
}

#[test]
fn disabling_jit_does_not_disable_filtering() {
    let df = fixture().expect("fixture");
    let context = QueryContext::with_jit_settings(false, 100);
    let filtered = df
        .query_with_context("age > 30 && salary < 90000", &context)
        .expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![1, 2]);

    let context = QueryContext::with_jit_settings(true, 1);
    let filtered = df
        .query_with_context("age > 30 && salary < 90000", &context)
        .expect("query");
    assert_eq!(ids(&filtered).expect("ids"), vec![1, 2]);
}

#[test]
fn empty_dataframe_query_returns_empty_frame() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::new(Vec::<i64>::new(), Some("a".to_string())).expect("series"),
    )
    .expect("add");
    let filtered = df.query("a > 1").expect("query");
    assert_eq!(filtered.row_count(), 0);
    assert_eq!(filtered.column_names().len(), 1);
}

#[test]
fn repeated_queries_are_stable() {
    // The JIT cache is shared through the context; repeated evaluation of the
    // same expression must keep returning the same rows.
    let df = fixture().expect("fixture");
    let context = QueryContext::with_jit_settings(true, 1);
    for _ in 0..8 {
        let filtered = df
            .query_with_context("age > 30 && dept == 'HR'", &context)
            .expect("query");
        assert_eq!(ids(&filtered).expect("ids"), vec![1, 4]);
    }
}

#[test]
fn jit_stats_do_not_report_uncompiled_executions_as_jit() {
    let df = fixture().expect("fixture");
    let context = QueryContext::with_jit_settings(true, 1);
    for _ in 0..4 {
        let _ = df
            .query_with_context("age > 30 && salary < 90000", &context)
            .expect("query");
    }
    let stats = context.jit_stats().expect("stats");
    assert_eq!(
        stats.jit_executions, 0,
        "no machine code is generated for query expressions, so nothing may be \
         recorded as a JIT execution"
    );
    assert!(
        stats.native_executions >= 4,
        "native executions must be recorded honestly (got {})",
        stats.native_executions
    );
    assert!((stats.jit_speedup_ratio() - 1.0).abs() < f64::EPSILON);
}
