//! Tests for expression support in distributed processing

#[cfg(feature = "distributed")]
mod tests {
    use pandrs::distributed::expr::{ColumnProjection, Expr, ExprDataType, UdfDefinition};
    use pandrs::distributed::DistributedContext;
    use pandrs::error::Result;
    use std::sync::Arc;

    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_select_expr() -> Result<()> {
        // Create test data
        let mut df = pandrs::dataframe::DataFrame::new();
        df.add_column(
            "a".to_string(),
            pandrs::series::Series::from_vec([1, 2, 3], Some("a")),
        )?;
        df.add_column(
            "b".to_string(),
            pandrs::series::Series::from_vec([4, 5, 6], Some("b")),
        )?;

        // Create context and register data
        let mut context = DistributedContext::new_local(2)?;
        context.register_dataframe("test", df)?;
        let test_df = context.dataset("test")?;

        // Test column selection
        let result = test_df
            .select_expr(&[
                ColumnProjection::column("a"),
                ColumnProjection::with_alias(Expr::col("b").mul(Expr::lit(2)), "b_doubled"),
            ])?
            .collect()?;

        assert_eq!(result.shape()?.0, 3); // 3 rows
        assert_eq!(result.shape()?.1, 2); // 2 columns

        // Verify column values
        let b_doubled = result.column("b_doubled")?.to_vec::<f64>()?;
        assert_eq!(b_doubled, [8.0, 10.0, 12.0]);

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_with_column() -> Result<()> {
        // Create test data
        let mut df = pandrs::dataframe::DataFrame::new();
        df.add_column(
            "a".to_string(),
            pandrs::series::Series::from_vec([1, 2, 3], Some("a")),
        )?;
        df.add_column(
            "b".to_string(),
            pandrs::series::Series::from_vec([4, 5, 6], Some("b")),
        )?;

        // Create context and register data
        let mut context = DistributedContext::new_local(2)?;
        context.register_dataframe("test", df)?;
        let test_df = context.dataset("test")?;

        // Test adding a calculated column
        let result = test_df
            .with_column("sum_ab", Expr::col("a").add(Expr::col("b")))?
            .collect()?;

        assert_eq!(result.shape()?.0, 3); // 3 rows
        assert_eq!(result.shape()?.1, 3); // 3 columns (a, b, sum_ab)

        // Verify column values
        let sum_ab = result.column("sum_ab")?.to_vec::<f64>()?;
        assert_eq!(sum_ab, [5.0, 7.0, 9.0]);

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_filter_expr() -> Result<()> {
        // Create test data
        let mut df = pandrs::dataframe::DataFrame::new();
        df.add_column(
            "a".to_string(),
            pandrs::series::Series::from_vec([1, 2, 3, 4, 5], Some("a")),
        )?;
        df.add_column(
            "b".to_string(),
            pandrs::series::Series::from_vec([5, 4, 3, 2, 1], Some("b")),
        )?;

        // Create context and register data
        let mut context = DistributedContext::new_local(2)?;
        context.register_dataframe("test", df)?;
        let test_df = context.dataset("test")?;

        // Test filtering with expression
        let result = test_df
            .filter_expr(Expr::col("a").add(Expr::col("b")).gt(Expr::lit(6)))
            .collect()?;

        assert_eq!(result.shape()?.0, 3); // 3 rows (where a + b > 6)

        // Verify filtered values
        let a_col = result.column("a")?.to_vec::<i32>()?;
        let b_col = result.column("b")?.to_vec::<i32>()?;

        assert!(a_col.iter().zip(b_col.iter()).all(|(a, b)| a + b > 6));

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_udf_creation() -> Result<()> {
        // Skip if not using local engine for tests
        if !cfg!(feature = "test_with_datafusion") {
            return Ok(());
        }

        // Create test data
        let mut df = pandrs::dataframe::DataFrame::new();
        df.add_column(
            "a".to_string(),
            pandrs::series::Series::from_vec([10, 20, 30], Some("a")),
        )?;
        df.add_column(
            "b".to_string(),
            pandrs::series::Series::from_vec([2, 4, 5], Some("b")),
        )?;

        // Create context and register data
        let mut context = DistributedContext::new_local(2)?;
        context.register_dataframe("test", df)?;
        let test_df = context.dataset("test")?;

        // Define a UDF
        let multiply_udf = UdfDefinition::new(
            "multiply_with_factor",
            ExprDataType::Float,
            [ExprDataType::Float, ExprDataType::Float],
            "param0 * param1 * 1.5", // multiply a and b, then multiply by 1.5
        );

        // Register the UDF and use it
        let result = test_df
            .create_udf(&[multiply_udf])?
            .select_expr(&[
                ColumnProjection::column("a"),
                ColumnProjection::column("b"),
                ColumnProjection::with_alias(
                    Expr::call("multiply_with_factor", [Expr::col("a"), Expr::col("b")]),
                    "result",
                ),
            ])?
            .collect()?;

        // Verify result column
        assert_eq!(result.shape()?.1, 3); // 3 columns

        let result_col = result.column("result")?.to_vec::<f64>()?;
        assert_eq!(result_col, [30.0, 120.0, 225.0]); // a * b * 1.5

        Ok(())
    }
}
