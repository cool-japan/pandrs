//! Example demonstrating schema validation and type checking for expressions

#[cfg(feature = "distributed")]
use pandrs::distributed::expr::{
    ColumnMeta, ColumnProjection, Expr, ExprDataType, ExprSchema, ExprValidator, InferredType,
    UdfDefinition,
};
#[cfg(feature = "distributed")]
use pandrs::distributed::DistributedContext;
#[cfg(feature = "distributed")]
use pandrs::error::Result;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Create a schema for validation
    let mut schema = ExprSchema::new();

    // Add columns to the schema
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
            "category",
            ExprDataType::String,
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
        ));

    // Create expression validator
    let mut validator = ExprValidator::new(&schema);

    // Add user-defined function
    validator.add_udf(
        "calculate_discount",
        ExprDataType::Float,
        vec![ExprDataType::Float, ExprDataType::Integer],
    );

    println!("Schema created with the following columns:");
    for col_name in schema.column_names() {
        let col = schema.column(col_name).unwrap();
        println!("  - {} ({:?})", col.name, col.data_type);
    }

    // Test some expressions
    let expressions = vec![
        // Valid expressions
        ("Basic column reference", Expr::col("name"), true),
        (
            "Arithmetic expression",
            Expr::col("price").mul(Expr::col("quantity")),
            true,
        ),
        (
            "Boolean expression",
            Expr::col("price").gt(Expr::lit(100.0)),
            true,
        ),
        (
            "CASE expression",
            Expr::case(
                vec![
                    (
                        Expr::col("price").lt(Expr::lit(50.0)),
                        Expr::lit("Low".to_string()),
                    ),
                    (
                        Expr::col("price").lt(Expr::lit(100.0)),
                        Expr::lit("Medium".to_string()),
                    ),
                ],
                Some(Expr::lit("High".to_string())),
            ),
            true,
        ),
        (
            "Function call",
            Expr::call(
                "calculate_discount",
                vec![Expr::col("price"), Expr::col("quantity")],
            ),
            true,
        ),
        (
            "String operations",
            Expr::col("name")
                .concat(Expr::lit(" - "))
                .concat(Expr::col("category")),
            true,
        ),
        // Invalid expressions
        (
            "Invalid column reference",
            Expr::col("missing_column"),
            false,
        ),
        (
            "Type mismatch in arithmetic",
            Expr::col("name").mul(Expr::col("quantity")),
            false,
        ),
        (
            "Type mismatch in comparison",
            Expr::col("name").eq(Expr::col("price")),
            false,
        ),
        (
            "Invalid function call",
            Expr::call("unknown_function", vec![Expr::col("price")]),
            false,
        ),
    ];

    // Validate expressions
    println!("\nValidating expressions:");
    for (desc, expr, expected_valid) in expressions {
        match validator.validate_expr(&expr) {
            Ok(inferred_type) => {
                if expected_valid {
                    println!(
                        "✅ [PASS] {}: Valid with inferred type {:?}",
                        desc, inferred_type.data_type
                    );
                } else {
                    println!("❌ [FAIL] {}: Expected to be invalid but was valid", desc);
                }
            }
            Err(err) => {
                if expected_valid {
                    println!(
                        "❌ [FAIL] {}: Expected to be valid but got error: {}",
                        desc, err
                    );
                } else {
                    println!("✅ [PASS] {}: Invalid with error: {}", desc, err);
                }
            }
        }
    }

    // Test validating projections
    println!("\nValidating projections:");

    let projections = vec![
        ColumnProjection::column("id"),
        ColumnProjection::column("name"),
        ColumnProjection::with_alias(Expr::col("price").mul(Expr::col("quantity")), "total_value"),
        ColumnProjection::with_alias(Expr::col("price").gt(Expr::lit(100.0)), "is_expensive"),
    ];

    match validator.validate_projections(&projections) {
        Ok(inferred_types) => {
            println!("✅ All projections are valid");
            println!("Inferred projection output types:");
            for (name, inferred_type) in inferred_types {
                println!("  - {} ({:?})", name, inferred_type.data_type);
            }
        }
        Err(err) => {
            println!("❌ Projection validation failed with error: {}", err);
        }
    }

    // Test invalid projections
    let invalid_projections = vec![
        ColumnProjection::column("id"),
        ColumnProjection::column("missing_column"), // This column doesn't exist
        ColumnProjection::with_alias(
            Expr::col("name").mul(Expr::col("quantity")), // Type mismatch
            "invalid_operation",
        ),
    ];

    match validator.validate_projections(&invalid_projections) {
        Ok(_) => {
            println!("❌ Invalid projections passed validation (unexpected)");
        }
        Err(err) => {
            println!(
                "✅ Invalid projections correctly failed validation with error: {}",
                err
            );
        }
    }

    // Example of using schema to create DataFusion Arrow schema
    #[cfg(feature = "distributed")]
    {
        let arrow_schema = schema.to_arrow_schema()?;
        println!("\nConverted to Arrow schema:");
        println!("{}", arrow_schema);
    }

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_expr_validation_example --features distributed");
}
