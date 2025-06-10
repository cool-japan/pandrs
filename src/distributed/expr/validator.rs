//! # Expression Validation
//!
//! This module provides validation capabilities for expressions, ensuring type safety
//! before execution.

use super::core::{BinaryOperator, Expr, Literal, UnaryOperator};
use super::projection::ColumnProjection;
use super::schema::{ColumnMeta, ExprSchema};
use super::ExprDataType;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Inferred type for an expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredType {
    /// Data type
    pub data_type: ExprDataType,
    /// Whether the expression can be null
    pub nullable: bool,
}

impl InferredType {
    /// Creates a new inferred type
    pub fn new(data_type: ExprDataType, nullable: bool) -> Self {
        Self {
            data_type,
            nullable,
        }
    }

    /// Gets a boolean type
    pub fn boolean(nullable: bool) -> Self {
        Self::new(ExprDataType::Boolean, nullable)
    }

    /// Gets an integer type
    pub fn integer(nullable: bool) -> Self {
        Self::new(ExprDataType::Integer, nullable)
    }

    /// Gets a float type
    pub fn float(nullable: bool) -> Self {
        Self::new(ExprDataType::Float, nullable)
    }

    /// Gets a string type
    pub fn string(nullable: bool) -> Self {
        Self::new(ExprDataType::String, nullable)
    }

    /// Gets a date type
    pub fn date(nullable: bool) -> Self {
        Self::new(ExprDataType::Date, nullable)
    }

    /// Gets a timestamp type
    pub fn timestamp(nullable: bool) -> Self {
        Self::new(ExprDataType::Timestamp, nullable)
    }
}

/// Validates expressions using schema information
pub struct ExprValidator<'a> {
    /// Schema to validate against
    schema: &'a ExprSchema,
    /// Function return types
    function_types: HashMap<String, (ExprDataType, Vec<ExprDataType>)>,
}

impl<'a> ExprValidator<'a> {
    /// Creates a new validator
    pub fn new(schema: &'a ExprSchema) -> Self {
        // Initialize with standard functions and their return types
        let mut function_types = HashMap::new();

        // String functions
        function_types.insert(
            "lower".to_string(),
            (ExprDataType::String, vec![ExprDataType::String]),
        );
        function_types.insert(
            "upper".to_string(),
            (ExprDataType::String, vec![ExprDataType::String]),
        );
        function_types.insert(
            "concat".to_string(),
            (
                ExprDataType::String,
                vec![ExprDataType::String, ExprDataType::String],
            ),
        );
        function_types.insert(
            "trim".to_string(),
            (ExprDataType::String, vec![ExprDataType::String]),
        );

        // Numeric functions
        function_types.insert(
            "abs".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );
        function_types.insert(
            "round".to_string(),
            (
                ExprDataType::Float,
                vec![ExprDataType::Float, ExprDataType::Integer],
            ),
        );
        function_types.insert(
            "floor".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );
        function_types.insert(
            "ceiling".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );

        // Date/Time functions
        function_types.insert(
            "date_trunc".to_string(),
            (
                ExprDataType::Timestamp,
                vec![ExprDataType::String, ExprDataType::Timestamp],
            ),
        );
        function_types.insert(
            "extract".to_string(),
            (
                ExprDataType::Integer,
                vec![ExprDataType::String, ExprDataType::Timestamp],
            ),
        );

        // Aggregate functions
        function_types.insert(
            "min".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );
        function_types.insert(
            "max".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );
        function_types.insert(
            "sum".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );
        function_types.insert(
            "avg".to_string(),
            (ExprDataType::Float, vec![ExprDataType::Float]),
        );
        function_types.insert(
            "count".to_string(),
            (ExprDataType::Integer, vec![ExprDataType::String]),
        );

        Self {
            schema,
            function_types,
        }
    }

    /// Adds a user-defined function
    pub fn add_udf(
        &mut self,
        name: impl Into<String>,
        return_type: ExprDataType,
        parameter_types: Vec<ExprDataType>,
    ) -> &mut Self {
        self.function_types
            .insert(name.into(), (return_type, parameter_types));
        self
    }

    /// Validates an expression and infers its type
    pub fn validate_expr(&self, expr: &Expr) -> Result<InferredType> {
        match expr {
            Expr::Column(name) => {
                // Check if column exists in schema
                if let Some(col_meta) = self.schema.column(name) {
                    Ok(InferredType::new(
                        col_meta.data_type.clone(),
                        col_meta.nullable,
                    ))
                } else {
                    Err(Error::InvalidOperation(format!(
                        "Column '{}' not found in schema",
                        name
                    )))
                }
            }
            Expr::Literal(lit) => {
                // Infer type from literal
                match lit {
                    Literal::Null => Err(Error::InvalidOperation(
                        "Cannot infer type for NULL literal without context".to_string(),
                    )),
                    Literal::Boolean(_) => Ok(InferredType::boolean(false)),
                    Literal::Integer(_) => Ok(InferredType::integer(false)),
                    Literal::Float(_) => Ok(InferredType::float(false)),
                    Literal::String(_) => Ok(InferredType::string(false)),
                }
            }
            Expr::BinaryOp { left, op, right } => {
                // Validate operands
                let left_type = self.validate_expr(left)?;
                let right_type = self.validate_expr(right)?;

                // Result is nullable if either operand is nullable
                let nullable = left_type.nullable || right_type.nullable;

                // Determine result type based on operator and operand types
                match op {
                    // Arithmetic operations
                    BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo => {
                        // For arithmetic operations, result is numeric
                        match (left_type.data_type.clone(), right_type.data_type.clone()) {
                            // If both are integers, result is integer (except division)
                            (ExprDataType::Integer, ExprDataType::Integer) => {
                                if *op == BinaryOperator::Divide {
                                    Ok(InferredType::float(nullable))
                                } else {
                                    Ok(InferredType::integer(nullable))
                                }
                            }
                            // If either is float, result is float
                            (ExprDataType::Float, _) | (_, ExprDataType::Float) => {
                                Ok(InferredType::float(nullable))
                            }
                            // String concatenation is a special case
                            (ExprDataType::String, ExprDataType::String)
                                if *op == BinaryOperator::Add =>
                            {
                                Ok(InferredType::string(nullable))
                            }
                            // Other combinations are invalid
                            _ => Err(Error::InvalidOperation(format!(
                                "Invalid operand types for arithmetic operation: {:?} {} {:?}",
                                left_type.data_type, op, right_type.data_type
                            ))),
                        }
                    }
                    // Comparison operations
                    BinaryOperator::Equal
                    | BinaryOperator::NotEqual
                    | BinaryOperator::LessThan
                    | BinaryOperator::LessThanOrEqual
                    | BinaryOperator::GreaterThan
                    | BinaryOperator::GreaterThanOrEqual => {
                        // For comparison operations, result is boolean
                        // Check if operand types are comparable
                        match (left_type.data_type.clone(), right_type.data_type.clone()) {
                            // Same types are comparable
                            (a, b) if a == b => Ok(InferredType::boolean(nullable)),
                            // Integer and float are comparable
                            (ExprDataType::Integer, ExprDataType::Float)
                            | (ExprDataType::Float, ExprDataType::Integer) => {
                                Ok(InferredType::boolean(nullable))
                            }
                            // Date and timestamp are comparable
                            (ExprDataType::Date, ExprDataType::Timestamp)
                            | (ExprDataType::Timestamp, ExprDataType::Date) => {
                                Ok(InferredType::boolean(nullable))
                            }
                            // Other combinations are invalid
                            _ => Err(Error::InvalidOperation(format!(
                                "Invalid operand types for comparison: {:?} {} {:?}",
                                left_type.data_type.clone(),
                                op,
                                right_type.data_type.clone()
                            ))),
                        }
                    }
                    // Logical operations
                    BinaryOperator::And | BinaryOperator::Or => {
                        // Both operands must be boolean
                        if left_type.data_type == ExprDataType::Boolean
                            && right_type.data_type == ExprDataType::Boolean
                        {
                            Ok(InferredType::boolean(nullable))
                        } else {
                            Err(Error::InvalidOperation(format!(
                                "Logical operations require boolean operands, got {:?} and {:?}",
                                left_type.data_type, right_type.data_type
                            )))
                        }
                    }
                    // Bitwise operations
                    BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseOr
                    | BinaryOperator::BitwiseXor => {
                        // Both operands must be integers
                        if left_type.data_type == ExprDataType::Integer
                            && right_type.data_type == ExprDataType::Integer
                        {
                            Ok(InferredType::integer(nullable))
                        } else {
                            Err(Error::InvalidOperation(format!(
                                "Bitwise operations require integer operands, got {:?} and {:?}",
                                left_type.data_type, right_type.data_type
                            )))
                        }
                    }
                    // Like pattern matching
                    BinaryOperator::Like => {
                        // Left operand must be string, right operand must be string
                        if left_type.data_type == ExprDataType::String
                            && right_type.data_type == ExprDataType::String
                        {
                            Ok(InferredType::boolean(nullable))
                        } else {
                            Err(Error::InvalidOperation(format!(
                                "LIKE operation requires string operands, got {:?} and {:?}",
                                left_type.data_type, right_type.data_type
                            )))
                        }
                    }
                    // String concatenation
                    BinaryOperator::Concat => {
                        // Both operands must be strings
                        if left_type.data_type == ExprDataType::String
                            && right_type.data_type == ExprDataType::String
                        {
                            Ok(InferredType::string(nullable))
                        } else {
                            Err(Error::InvalidOperation(format!(
                                "String concatenation requires string operands, got {:?} and {:?}",
                                left_type.data_type, right_type.data_type
                            )))
                        }
                    }
                }
            }
            Expr::UnaryOp { op, expr } => {
                // Validate operand
                let expr_type = self.validate_expr(expr)?;

                match op {
                    // Negation
                    UnaryOperator::Negate => {
                        // Operand must be numeric
                        match expr_type.data_type {
                            ExprDataType::Integer => Ok(InferredType::integer(expr_type.nullable)),
                            ExprDataType::Float => Ok(InferredType::float(expr_type.nullable)),
                            _ => Err(Error::InvalidOperation(format!(
                                "Negation requires numeric operand, got {:?}",
                                expr_type.data_type
                            ))),
                        }
                    }
                    // Logical NOT
                    UnaryOperator::Not => {
                        // Operand must be boolean
                        if expr_type.data_type == ExprDataType::Boolean {
                            Ok(InferredType::boolean(expr_type.nullable))
                        } else {
                            Err(Error::InvalidOperation(format!(
                                "Logical NOT requires boolean operand, got {:?}",
                                expr_type.data_type
                            )))
                        }
                    }
                    // IS NULL and IS NOT NULL
                    UnaryOperator::IsNull | UnaryOperator::IsNotNull => {
                        // Can be applied to any type, result is boolean
                        Ok(InferredType::boolean(false))
                    }
                }
            }
            Expr::Function { name, args } => {
                // Check if function exists
                if let Some((return_type, param_types)) = self.function_types.get(name) {
                    // Validate number of arguments
                    if args.len() != param_types.len() {
                        return Err(Error::InvalidOperation(format!(
                            "Function '{}' expects {} arguments, got {}",
                            name,
                            param_types.len(),
                            args.len()
                        )));
                    }

                    // Validate each argument
                    for (i, (arg, expected_type)) in args.iter().zip(param_types.iter()).enumerate()
                    {
                        let arg_type = self.validate_expr(arg)?;

                        // Check if argument type matches expected type
                        if arg_type.data_type != *expected_type {
                            return Err(Error::InvalidOperation(
                                format!("Function '{}' argument {} has invalid type: expected {:?}, got {:?}",
                                    name, i + 1, expected_type, arg_type.data_type)
                            ));
                        }
                    }

                    // Return function's return type
                    Ok(InferredType::new(return_type.clone(), true))
                } else {
                    Err(Error::InvalidOperation(format!(
                        "Function '{}' not found",
                        name
                    )))
                }
            }
            Expr::Case {
                when_then,
                else_expr,
            } => {
                if when_then.is_empty() {
                    return Err(Error::InvalidOperation(
                        "CASE expression must have at least one WHEN clause".to_string(),
                    ));
                }

                // Validate WHEN conditions (must be boolean)
                for (when, _) in when_then.iter() {
                    let when_type = self.validate_expr(when)?;
                    if when_type.data_type != ExprDataType::Boolean {
                        return Err(Error::InvalidOperation(format!(
                            "CASE WHEN condition must be boolean, got {:?}",
                            when_type.data_type
                        )));
                    }
                }

                // Get type of first THEN expression
                let first_then_type = self.validate_expr(&when_then[0].1)?;
                let mut nullable = first_then_type.nullable;

                // Validate that all THEN expressions have the same type
                for (_, then) in when_then.iter().skip(1) {
                    let then_type = self.validate_expr(then)?;
                    if then_type.data_type != first_then_type.data_type {
                        return Err(Error::InvalidOperation(format!(
                            "CASE THEN expressions must have the same type: {:?} vs {:?}",
                            first_then_type.data_type, then_type.data_type
                        )));
                    }
                    nullable = nullable || then_type.nullable;
                }

                // If ELSE expression exists, validate it has the same type as THEN expressions
                if let Some(else_expr) = else_expr {
                    let else_type = self.validate_expr(else_expr)?;
                    if else_type.data_type != first_then_type.data_type {
                        return Err(Error::InvalidOperation(
                            format!("CASE ELSE expression must have the same type as THEN expressions: {:?} vs {:?}",
                                first_then_type.data_type, else_type.data_type)
                        ));
                    }
                    nullable = nullable || else_type.nullable;
                } else {
                    // If no ELSE expression, result is nullable (missing case)
                    nullable = true;
                }

                Ok(InferredType::new(first_then_type.data_type, nullable))
            }
            Expr::Cast {
                expr: inner,
                data_type,
            } => {
                // Validate inner expression
                let inner_type = self.validate_expr(inner)?;

                // Check if cast is valid
                let valid_cast = match (inner_type.data_type.clone(), data_type) {
                    // Numeric conversions are valid
                    (ExprDataType::Integer, ExprDataType::Float)
                    | (ExprDataType::Float, ExprDataType::Integer) => true,

                    // String to/from numeric conversions are valid
                    (ExprDataType::String, ExprDataType::Integer)
                    | (ExprDataType::String, ExprDataType::Float)
                    | (ExprDataType::Integer, ExprDataType::String)
                    | (ExprDataType::Float, ExprDataType::String) => true,

                    // Date/Timestamp conversions are valid
                    (ExprDataType::Date, ExprDataType::Timestamp)
                    | (ExprDataType::Timestamp, ExprDataType::Date) => true,

                    // String to/from date/timestamp conversions are valid
                    (ExprDataType::String, ExprDataType::Date)
                    | (ExprDataType::String, ExprDataType::Timestamp)
                    | (ExprDataType::Date, ExprDataType::String)
                    | (ExprDataType::Timestamp, ExprDataType::String) => true,

                    // Boolean conversions
                    (ExprDataType::Boolean, ExprDataType::Integer)
                    | (ExprDataType::Boolean, ExprDataType::String)
                    | (ExprDataType::Integer, ExprDataType::Boolean)
                    | (ExprDataType::String, ExprDataType::Boolean) => true,

                    // Same type is always valid
                    (a, b) if a == *b => true,

                    // Other conversions are invalid
                    _ => false,
                };

                if valid_cast {
                    Ok(InferredType::new(data_type.clone(), inner_type.nullable))
                } else {
                    Err(Error::InvalidOperation(format!(
                        "Invalid cast from {:?} to {:?}",
                        inner_type.data_type.clone(),
                        data_type
                    )))
                }
            }
            Expr::Coalesce { exprs } => {
                if exprs.is_empty() {
                    return Err(Error::InvalidOperation(
                        "COALESCE expression must have at least one argument".to_string(),
                    ));
                }

                // Get type of first expression
                let first_type = self.validate_expr(&exprs[0])?;
                let mut nullable = first_type.nullable;

                // Validate that all expressions have the same type
                for expr in exprs.iter().skip(1) {
                    let expr_type = self.validate_expr(expr)?;
                    if expr_type.data_type != first_type.data_type {
                        return Err(Error::InvalidOperation(format!(
                            "COALESCE expressions must have the same type: {:?} vs {:?}",
                            first_type.data_type, expr_type.data_type
                        )));
                    }
                    nullable = nullable && expr_type.nullable;
                }

                // Result is nullable only if all expressions are nullable
                Ok(InferredType::new(first_type.data_type, nullable))
            }
        }
    }

    /// Validates a list of projections
    pub fn validate_projections(
        &self,
        projections: &[ColumnProjection],
    ) -> Result<HashMap<String, InferredType>> {
        let mut result = HashMap::new();

        for projection in projections {
            // Validate the expression
            let expr_type = self.validate_expr(&projection.expr)?;

            // Get the output name
            let output_name = projection.output_name();

            // Add to result
            result.insert(output_name, expr_type);
        }

        Ok(result)
    }
}
