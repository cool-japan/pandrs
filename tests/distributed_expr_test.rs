//! Tests for expression support in distributed processing

#[cfg(feature = "distributed")]
mod tests {
    use pandrs::distributed::expr::{ColumnProjection, Expr, ExprDataType, UdfDefinition};
    use pandrs::distributed::{DistributedConfig, DistributedContext};
    use pandrs::error::Result;
    use pandrs::series::Series;

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_expr_creation() -> Result<()> {
        // Test basic expression creation
        let col_expr = Expr::col("a");
        let lit_expr = Expr::lit(42);
        let add_expr = col_expr.clone().add(lit_expr.clone());
        let mul_expr = col_expr.clone().mul(Expr::lit(2));
        let gt_expr = col_expr.gt(Expr::lit(10));

        // Verify expressions can be created without panic
        assert!(format!("{:?}", add_expr).contains("Add"));
        assert!(format!("{:?}", mul_expr).contains("Mul"));
        assert!(format!("{:?}", gt_expr).contains("GreaterThan"));

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_column_projection() -> Result<()> {
        // Test column projection creation
        let simple_col = ColumnProjection::column("a");
        let aliased_col =
            ColumnProjection::with_alias(Expr::col("b").mul(Expr::lit(2)), "b_doubled");

        // Verify projections can be created
        assert!(format!("{:?}", simple_col).len() > 0);
        assert!(format!("{:?}", aliased_col).len() > 0);

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_expr_data_type() -> Result<()> {
        // Test expression data types
        let int_type = ExprDataType::Integer;
        let float_type = ExprDataType::Float;
        let string_type = ExprDataType::String;
        let bool_type = ExprDataType::Boolean;

        // Verify data types can be created
        assert!(format!("{:?}", int_type).contains("Integer"));
        assert!(format!("{:?}", float_type).contains("Float"));
        assert!(format!("{:?}", string_type).contains("String"));
        assert!(format!("{:?}", bool_type).contains("Boolean"));

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_udf_definition() -> Result<()> {
        // Test UDF definition creation
        let udf = UdfDefinition::new(
            "multiply_with_factor",
            ExprDataType::Float,
            vec![ExprDataType::Float, ExprDataType::Float],
            "param0 * param1 * 1.5",
        );

        // Verify UDF definition can be created
        assert!(format!("{:?}", udf).len() > 0);

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_distributed_context_with_dataframe() -> Result<()> {
        // Create test data
        let mut df = pandrs::dataframe::DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1, 2, 3], Some("a".to_string()))?,
        )?;
        df.add_column(
            "b".to_string(),
            Series::new(vec![4, 5, 6], Some("b".to_string()))?,
        )?;

        // Create context with config
        let config = DistributedConfig::new()
            .with_executor("datafusion")
            .with_concurrency(2);

        let mut context = DistributedContext::new(config)?;
        context.register_dataframe("test", &df)?;

        // Verify registration
        let retrieved = context.get_dataset("test");
        assert!(retrieved.is_some());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_complex_expression_building() -> Result<()> {
        // Test building complex expressions
        let expr = Expr::col("a")
            .add(Expr::col("b"))
            .mul(Expr::lit(2))
            .sub(Expr::lit(10));

        // Test comparison expressions
        let filter_expr = Expr::col("a").add(Expr::col("b")).gt(Expr::lit(6));

        // Test logical expressions
        let and_expr = Expr::col("a")
            .gt(Expr::lit(0))
            .and(Expr::col("b").lt(Expr::lit(100)));

        let or_expr = Expr::col("a")
            .eq(Expr::lit(1))
            .or(Expr::col("a").eq(Expr::lit(2)));

        // Verify expressions can be created
        assert!(format!("{:?}", expr).len() > 0);
        assert!(format!("{:?}", filter_expr).len() > 0);
        assert!(format!("{:?}", and_expr).len() > 0);
        assert!(format!("{:?}", or_expr).len() > 0);

        Ok(())
    }
}
