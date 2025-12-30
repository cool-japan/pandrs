//! Tests for expression validation and type checking

#[cfg(feature = "distributed")]
mod tests {
    use pandrs::distributed::expr::{
        ColumnMeta, ColumnProjection, Expr, ExprDataType, ExprSchema, ExprValidator, InferredType,
        UdfDefinition,
    };
    use pandrs::error::Result;

    // Helper to create a test schema
    fn create_test_schema() -> ExprSchema {
        let mut schema = ExprSchema::new();

        schema
            .add_column(ColumnMeta::new("id", ExprDataType::Integer, false, None))
            .add_column(ColumnMeta::new("name", ExprDataType::String, false, None))
            .add_column(ColumnMeta::new("price", ExprDataType::Float, false, None))
            .add_column(ColumnMeta::new(
                "quantity",
                ExprDataType::Integer,
                false,
                None,
            ))
            .add_column(ColumnMeta::new(
                "in_stock",
                ExprDataType::Boolean,
                false,
                None,
            ))
            .add_column(ColumnMeta::new(
                "created_at",
                ExprDataType::Timestamp,
                false,
                None,
            ))
            .add_column(ColumnMeta::new(
                "nullable_value",
                ExprDataType::Float,
                true,
                None,
            ));

        schema
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_column_reference_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Valid column reference
        let expr = Expr::col("id");
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Integer);
        assert!(!result.nullable);

        // Invalid column reference
        let expr = Expr::col("nonexistent");
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_literal_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Integer literal
        let expr = Expr::lit(42);
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Integer);
        assert!(!result.nullable);

        // Float literal
        let expr = Expr::lit(42.5);
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);
        assert!(!result.nullable);

        // String literal
        let expr = Expr::lit("test");
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::String);
        assert!(!result.nullable);

        // Boolean literal
        let expr = Expr::lit(true);
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Boolean);
        assert!(!result.nullable);

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_arithmetic_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Integer + Integer = Integer
        let expr = Expr::col("id").add(Expr::col("quantity"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Integer);

        // Integer + Float = Float
        let expr = Expr::col("id").add(Expr::col("price"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);

        // Float * Integer = Float
        let expr = Expr::col("price").mul(Expr::col("quantity"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);

        // Integer / Integer = Float (division always results in float)
        let expr = Expr::col("id").div(Expr::col("quantity"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);

        // String + String should be invalid for arithmetic
        let expr = Expr::col("name").add(Expr::col("name"));
        assert!(validator.validate_expr(&expr).is_err());

        // Invalid combination
        let expr = Expr::col("name").mul(Expr::col("id"));
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_comparison_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Integer == Integer
        let expr = Expr::col("id").eq(Expr::col("quantity"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Boolean);

        // String == String
        let expr = Expr::col("name").eq(Expr::lit("test"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Boolean);

        // Float > Integer (allowed)
        let expr = Expr::col("price").gt(Expr::col("id"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Boolean);

        // Invalid comparison
        let expr = Expr::col("name").lt(Expr::col("price"));
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_logical_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Boolean AND Boolean
        let expr = Expr::col("in_stock").and(Expr::col("in_stock"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Boolean);

        // Boolean OR Boolean
        let expr = Expr::col("in_stock").or(Expr::col("in_stock"));
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Boolean);

        // Invalid logical operation
        let expr = Expr::col("id").and(Expr::col("in_stock"));
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_function_validation() -> Result<()> {
        let schema = create_test_schema();
        let mut validator = ExprValidator::new(&schema);

        // Add a custom function
        validator.add_udf(
            "calculate_total",
            ExprDataType::Float,
            vec![ExprDataType::Float, ExprDataType::Integer],
        );

        // Valid function call
        let expr = Expr::call(
            "calculate_total",
            vec![Expr::col("price"), Expr::col("quantity")],
        );
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);

        // Invalid function call (unknown function)
        let expr = Expr::call("unknown_function", vec![Expr::col("id")]);
        assert!(validator.validate_expr(&expr).is_err());

        // Invalid function call (wrong parameter types)
        let expr = Expr::call(
            "calculate_total",
            vec![Expr::col("name"), Expr::col("quantity")],
        );
        assert!(validator.validate_expr(&expr).is_err());

        // Invalid function call (wrong number of parameters)
        let expr = Expr::call("calculate_total", vec![Expr::col("price")]);
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_case_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Valid CASE expression
        let expr = Expr::case(
            vec![
                (Expr::col("price").lt(Expr::lit(50.0)), Expr::lit("Low")),
                (Expr::col("price").lt(Expr::lit(100.0)), Expr::lit("Medium")),
            ],
            Some(Expr::lit("High")),
        );
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::String);

        // Invalid CASE expression (mixed return types)
        let expr = Expr::case(
            vec![
                (Expr::col("price").lt(Expr::lit(50.0)), Expr::lit("Low")),
                (Expr::col("price").lt(Expr::lit(100.0)), Expr::lit(42)),
            ],
            None,
        );
        assert!(validator.validate_expr(&expr).is_err());

        // Invalid CASE expression (non-boolean conditions)
        let expr = Expr::case(
            vec![
                (Expr::col("price"), Expr::lit("Low")),
                (Expr::col("quantity"), Expr::lit("High")),
            ],
            None,
        );
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_cast_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Valid casts

        // Integer to Float
        let expr = Expr::col("id").to_float();
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);

        // Float to Integer
        let expr = Expr::col("price").to_integer();
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Integer);

        // Integer to String
        let expr = Expr::col("id").to_string();
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::String);

        // Boolean to String
        let expr = Expr::col("in_stock").to_string();
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::String);

        // Invalid casts

        // Boolean to Date
        let expr = Expr::cast(Expr::col("in_stock"), ExprDataType::Date);
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_coalesce_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Valid coalesce
        let expr = Expr::coalesce(vec![Expr::col("nullable_value"), Expr::lit(0.0)]);
        let result = validator.validate_expr(&expr)?;
        assert_eq!(result.data_type, ExprDataType::Float);
        assert!(!result.nullable); // Not nullable because second argument is not nullable

        // Invalid coalesce (mixed types)
        let expr = Expr::coalesce(vec![Expr::col("nullable_value"), Expr::col("id")]);
        assert!(validator.validate_expr(&expr).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_projection_validation() -> Result<()> {
        let schema = create_test_schema();
        let validator = ExprValidator::new(&schema);

        // Valid projections
        let projections = vec![
            ColumnProjection::column("id"),
            ColumnProjection::with_alias(
                Expr::col("price").mul(Expr::col("quantity")),
                "total_value",
            ),
            ColumnProjection::with_alias(Expr::col("price").gt(Expr::lit(100.0)), "is_expensive"),
        ];

        let result = validator.validate_projections(&projections)?;
        assert_eq!(result.len(), 3);
        assert_eq!(result.get("id").unwrap().data_type, ExprDataType::Integer);
        assert_eq!(
            result.get("total_value").unwrap().data_type,
            ExprDataType::Float
        );
        assert_eq!(
            result.get("is_expensive").unwrap().data_type,
            ExprDataType::Boolean
        );

        // Invalid projections
        let projections = [ColumnProjection::column("nonexistent")];

        assert!(validator.validate_projections(&projections).is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_arrow_schema_conversion() -> Result<()> {
        let schema = create_test_schema();

        let arrow_schema = schema.to_arrow_schema()?;

        // Check that all columns are present
        assert_eq!(arrow_schema.fields().len(), schema.len());

        // Check types of specific columns
        let id_field = arrow_schema.field_with_name("id").unwrap();
        assert_eq!(id_field.data_type(), &arrow::datatypes::DataType::Int64);

        let name_field = arrow_schema.field_with_name("name").unwrap();
        assert_eq!(name_field.data_type(), &arrow::datatypes::DataType::Utf8);

        let price_field = arrow_schema.field_with_name("price").unwrap();
        assert_eq!(
            price_field.data_type(),
            &arrow::datatypes::DataType::Float64
        );

        // Convert back to ExprSchema
        let round_trip_schema = ExprSchema::from_arrow_schema(&arrow_schema)?;

        // Check that all columns are present
        assert_eq!(round_trip_schema.len(), schema.len());

        // Check specific column types
        let id_col = round_trip_schema.column("id").unwrap();
        assert_eq!(id_col.data_type, ExprDataType::Integer);

        let name_col = round_trip_schema.column("name").unwrap();
        assert_eq!(name_col.data_type, ExprDataType::String);

        let price_col = round_trip_schema.column("price").unwrap();
        assert_eq!(price_col.data_type, ExprDataType::Float);

        Ok(())
    }
}
