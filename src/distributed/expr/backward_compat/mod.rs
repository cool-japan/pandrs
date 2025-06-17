//! # Expression Support for Distributed Processing (Legacy)
//!
//! DEPRECATED: This file is maintained for backward compatibility only.
//! Please use the `distributed::expr` module directory structure instead.
//!
//! This module provides support for complex expressions and user-defined functions
//! for distributed DataFrames, enabling more complex transformations and calculations.
//!
//! @deprecated

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::distributed::core::dataframe::DistributedDataFrame;
use crate::distributed::execution::{ExecutionPlan, Operation};
use crate::error::Result;

/// Represents an expression that can be used in a distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// Column reference
    Column(String),
    /// Literal value
    Literal(Literal),
    /// Binary operation
    BinaryOp {
        /// Left operand
        left: Box<Expr>,
        /// Operator
        op: BinaryOperator,
        /// Right operand
        right: Box<Expr>,
    },
    /// Unary operation
    UnaryOp {
        /// Operator
        op: UnaryOperator,
        /// Operand
        expr: Box<Expr>,
    },
    /// Function call
    Function {
        /// Function name
        name: String,
        /// Arguments
        args: Vec<Expr>,
    },
    /// Case statement
    Case {
        /// When conditions and results
        when_then: Vec<(Expr, Expr)>,
        /// Default result
        else_expr: Option<Box<Expr>>,
    },
    /// CAST expression
    Cast {
        /// Input expression
        expr: Box<Expr>,
        /// Target data type
        data_type: ExprDataType,
    },
    /// COALESCE expression
    Coalesce {
        /// Expressions to check
        exprs: Vec<Expr>,
    },
}

/// Represents a literal value in an expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
}

/// Types of binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOperator {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Subtract,
    /// Multiplication (*)
    Multiply,
    /// Division (/)
    Divide,
    /// Modulo (%)
    Modulo,
    /// Equality (=)
    Equal,
    /// Inequality (<>)
    NotEqual,
    /// Less than (<)
    LessThan,
    /// Less than or equal to (<=)
    LessThanOrEqual,
    /// Greater than (>)
    GreaterThan,
    /// Greater than or equal to (>=)
    GreaterThanOrEqual,
    /// Logical AND
    And,
    /// Logical OR
    Or,
    /// Bitwise AND (&)
    BitwiseAnd,
    /// Bitwise OR (|)
    BitwiseOr,
    /// Bitwise XOR (^)
    BitwiseXor,
    /// Like pattern matching
    Like,
    /// String concatenation (||)
    Concat,
}

/// Types of unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    /// Negation (-)
    Negate,
    /// Logical NOT
    Not,
    /// Is NULL
    IsNull,
    /// Is NOT NULL
    IsNotNull,
}

/// Data types for expressions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExprDataType {
    /// Boolean type
    Boolean,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// String type
    String,
    /// Date type
    Date,
    /// Timestamp type
    Timestamp,
}

impl fmt::Display for ExprDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::Integer => write!(f, "BIGINT"),
            Self::Float => write!(f, "DOUBLE"),
            Self::String => write!(f, "VARCHAR"),
            Self::Date => write!(f, "DATE"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Boolean(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(fl) => write!(f, "{}", fl),
            Self::String(s) => write!(f, "'{}'", s.replace('\'', "''")), // Escape single quotes
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Subtract => write!(f, "-"),
            Self::Multiply => write!(f, "*"),
            Self::Divide => write!(f, "/"),
            Self::Modulo => write!(f, "%"),
            Self::Equal => write!(f, "="),
            Self::NotEqual => write!(f, "<>"),
            Self::LessThan => write!(f, "<"),
            Self::LessThanOrEqual => write!(f, "<="),
            Self::GreaterThan => write!(f, ">"),
            Self::GreaterThanOrEqual => write!(f, ">="),
            Self::And => write!(f, "AND"),
            Self::Or => write!(f, "OR"),
            Self::BitwiseAnd => write!(f, "&"),
            Self::BitwiseOr => write!(f, "|"),
            Self::BitwiseXor => write!(f, "^"),
            Self::Like => write!(f, "LIKE"),
            Self::Concat => write!(f, "||"),
        }
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Negate => write!(f, "-"),
            Self::Not => write!(f, "NOT "),
            Self::IsNull => write!(f, "IS NULL"),
            Self::IsNotNull => write!(f, "IS NOT NULL"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column(name) => write!(f, "{}", name),
            Self::Literal(value) => write!(f, "{}", value),
            Self::BinaryOp { left, op, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
            Self::UnaryOp { op, expr } => match op {
                UnaryOperator::IsNull | UnaryOperator::IsNotNull => {
                    write!(f, "({} {})", expr, op)
                }
                _ => write!(f, "{}({})", op, expr),
            },
            Self::Function { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Self::Case {
                when_then,
                else_expr,
            } => {
                write!(f, "CASE")?;
                for (when, then) in when_then {
                    write!(f, " WHEN {} THEN {}", when, then)?;
                }
                if let Some(else_expr) = else_expr {
                    write!(f, " ELSE {}", else_expr)?;
                }
                write!(f, " END")
            }
            Self::Cast { expr, data_type } => {
                write!(f, "CAST({} AS {})", expr, data_type)
            }
            Self::Coalesce { exprs } => {
                write!(f, "COALESCE(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl Expr {
    /// Creates a column reference
    pub fn col(name: impl Into<String>) -> Self {
        Self::Column(name.into())
    }

    /// Creates a literal value
    pub fn lit<T: Into<Literal>>(value: T) -> Self {
        Self::Literal(value.into())
    }

    /// Creates a literal NULL value
    pub fn null() -> Self {
        Self::Literal(Literal::Null)
    }

    /// Creates a function call
    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Self::Function {
            name: name.into(),
            args,
        }
    }

    /// Creates a binary operation
    pub fn binary(left: Expr, op: BinaryOperator, right: Expr) -> Self {
        Self::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Creates a unary operation
    pub fn unary(op: UnaryOperator, expr: Expr) -> Self {
        Self::UnaryOp {
            op,
            expr: Box::new(expr),
        }
    }

    /// Creates a CASE expression
    pub fn case(when_then: Vec<(Expr, Expr)>, else_expr: Option<Expr>) -> Self {
        Self::Case {
            when_then,
            else_expr: else_expr.map(Box::new),
        }
    }

    /// Creates a CAST expression
    pub fn cast(expr: Expr, data_type: ExprDataType) -> Self {
        Self::Cast {
            expr: Box::new(expr),
            data_type,
        }
    }

    /// Creates a COALESCE expression
    pub fn coalesce(exprs: Vec<Expr>) -> Self {
        Self::Coalesce { exprs }
    }

    /// Adds two expressions
    pub fn add(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Add, other)
    }

    /// Subtracts an expression from this one
    pub fn sub(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Subtract, other)
    }

    /// Multiplies this expression by another
    pub fn mul(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Multiply, other)
    }

    /// Divides this expression by another
    pub fn div(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Divide, other)
    }

    /// Applies modulo operation
    pub fn modulo(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Modulo, other)
    }

    /// Checks if this expression equals another
    pub fn eq(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Equal, other)
    }

    /// Checks if this expression does not equal another
    pub fn neq(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::NotEqual, other)
    }

    /// Checks if this expression is less than another
    pub fn lt(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::LessThan, other)
    }

    /// Checks if this expression is less than or equal to another
    pub fn lte(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::LessThanOrEqual, other)
    }

    /// Checks if this expression is greater than another
    pub fn gt(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::GreaterThan, other)
    }

    /// Checks if this expression is greater than or equal to another
    pub fn gte(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::GreaterThanOrEqual, other)
    }

    /// Applies logical AND
    pub fn and(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::And, other)
    }

    /// Applies logical OR
    pub fn or(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Or, other)
    }

    /// Applies LIKE pattern matching
    pub fn like(self, pattern: impl Into<String>) -> Self {
        Self::binary(self, BinaryOperator::Like, Self::lit(pattern.into()))
    }

    /// Concatenates with another string expression
    pub fn concat(self, other: Expr) -> Self {
        Self::binary(self, BinaryOperator::Concat, other)
    }

    /// Negates this expression
    pub fn negate(self) -> Self {
        Self::unary(UnaryOperator::Negate, self)
    }

    /// Applies logical NOT
    pub fn not(self) -> Self {
        Self::unary(UnaryOperator::Not, self)
    }

    /// Checks if this expression is NULL
    pub fn is_null(self) -> Self {
        Self::unary(UnaryOperator::IsNull, self)
    }

    /// Checks if this expression is NOT NULL
    pub fn is_not_null(self) -> Self {
        Self::unary(UnaryOperator::IsNotNull, self)
    }

    /// Casts this expression to boolean
    pub fn to_boolean(self) -> Self {
        Self::cast(self, ExprDataType::Boolean)
    }

    /// Casts this expression to integer
    pub fn to_integer(self) -> Self {
        Self::cast(self, ExprDataType::Integer)
    }

    /// Casts this expression to float
    pub fn to_float(self) -> Self {
        Self::cast(self, ExprDataType::Float)
    }

    /// Casts this expression to string
    pub fn to_string(self) -> Self {
        Self::cast(self, ExprDataType::String)
    }

    /// Casts this expression to date
    pub fn to_date(self) -> Self {
        Self::cast(self, ExprDataType::Date)
    }

    /// Casts this expression to timestamp
    pub fn to_timestamp(self) -> Self {
        Self::cast(self, ExprDataType::Timestamp)
    }
}

// Conversions to Literal
impl From<bool> for Literal {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i64> for Literal {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<i32> for Literal {
    fn from(value: i32) -> Self {
        Self::Integer(value as i64)
    }
}

impl From<f64> for Literal {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Self::Float(value as f64)
    }
}

impl From<String> for Literal {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for Literal {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

/// A user-defined function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UdfDefinition {
    /// Name of the function
    pub name: String,
    /// Return type
    pub return_type: ExprDataType,
    /// Parameter types
    pub parameter_types: Vec<ExprDataType>,
    /// SQL function body
    pub body: String,
}

impl UdfDefinition {
    /// Creates a new UDF definition
    pub fn new(
        name: impl Into<String>,
        return_type: ExprDataType,
        parameter_types: Vec<ExprDataType>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            return_type,
            parameter_types,
            body: body.into(),
        }
    }

    /// Converts the UDF definition to SQL CREATE FUNCTION statement
    pub fn to_sql(&self) -> String {
        let mut params = Vec::with_capacity(self.parameter_types.len());
        for (i, param_type) in self.parameter_types.iter().enumerate() {
            params.push(format!("param{} {}", i, param_type));
        }

        format!(
            "CREATE FUNCTION {} ({}) RETURNS {} AS '{}'",
            self.name,
            params.join(", "),
            self.return_type,
            self.body
        )
    }
}

/// Represents a column projection with optional alias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnProjection {
    /// Expression to project
    pub expr: Expr,
    /// Optional alias
    pub alias: Option<String>,
}

impl ColumnProjection {
    /// Creates a new column projection
    pub fn new(expr: Expr, alias: Option<impl Into<String>>) -> Self {
        Self {
            expr,
            alias: alias.map(|a| a.into()),
        }
    }

    /// Creates a column projection with alias
    pub fn with_alias(expr: Expr, alias: impl Into<String>) -> Self {
        Self {
            expr,
            alias: Some(alias.into()),
        }
    }

    /// Creates a simple column projection without alias
    pub fn column(name: impl Into<String>) -> Self {
        Self {
            expr: Expr::col(name),
            alias: None,
        }
    }

    /// Converts the column projection to SQL
    pub fn to_sql(&self) -> String {
        match &self.alias {
            Some(alias) => format!("{} AS {}", self.expr, alias),
            None => format!("{}", self.expr),
        }
    }
}

/// Extension trait for projection operations
pub trait ProjectionExt {
    /// Selects expressions from the DataFrame
    fn select_expr(&self, projections: &[ColumnProjection]) -> Result<DistributedDataFrame>;

    /// Creates a new calculated column
    fn with_column(&self, name: impl Into<String>, expr: Expr) -> Result<DistributedDataFrame>;

    /// Filters the DataFrame using an expression
    fn filter_expr(&self, expr: Expr) -> Result<DistributedDataFrame>;

    /// Creates user-defined functions
    fn create_udf(&self, udfs: &[UdfDefinition]) -> Result<DistributedDataFrame>;
}

impl ProjectionExt for DistributedDataFrame {
    fn select_expr(&self, projections: &[ColumnProjection]) -> Result<DistributedDataFrame> {
        // Create a custom operation for expressions
        let operation = Operation::Custom {
            name: "select_expr".to_string(),
            params: [(
                "projections".to_string(),
                serde_json::to_string(projections).unwrap_or_default(),
            )]
            .iter()
            .cloned()
            .collect(),
        };

        if self.is_lazy() {
            let mut new_df = self.clone_empty();
            let mut plan = ExecutionPlan::new(&self.id().to_string());
            plan.add_operation(operation);
            new_df.add_pending_operation(plan, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            let mut plan = ExecutionPlan::new(&self.id().to_string());
            plan.add_operation(operation);
            self.execute_operation(plan, vec![self.id().to_string()])
        }
    }

    fn with_column(&self, name: impl Into<String>, expr: Expr) -> Result<DistributedDataFrame> {
        let name = name.into();
        let projection = ColumnProjection::with_alias(expr, name.clone());

        // Create a custom operation for adding a column
        let operation = Operation::Custom {
            name: "with_column".to_string(),
            params: [
                ("column_name".to_string(), name),
                (
                    "projection".to_string(),
                    serde_json::to_string(&projection).unwrap_or_default(),
                ),
            ]
            .iter()
            .cloned()
            .collect(),
        };

        if self.is_lazy() {
            let mut new_df = self.clone_empty();
            let mut plan = ExecutionPlan::new(&self.id().to_string());
            plan.add_operation(operation);
            new_df.add_pending_operation(plan, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            let mut plan = ExecutionPlan::new(&self.id().to_string());
            plan.add_operation(operation);
            self.execute_operation(plan, vec![self.id().to_string()])
        }
    }

    fn filter_expr(&self, expr: Expr) -> Result<DistributedDataFrame> {
        // Convert the expression to SQL
        let filter_sql = format!("{:?}", expr);

        // Use the existing filter operation with the SQL expression
        self.filter(&filter_sql)
    }

    fn create_udf(&self, udfs: &[UdfDefinition]) -> Result<DistributedDataFrame> {
        // Create a custom operation for UDFs
        let operation = Operation::Custom {
            name: "create_udf".to_string(),
            params: [(
                "udfs".to_string(),
                serde_json::to_string(udfs).unwrap_or_default(),
            )]
            .iter()
            .cloned()
            .collect(),
        };

        if self.is_lazy() {
            let mut new_df = self.clone_empty();
            let mut plan = ExecutionPlan::new(&self.id().to_string());
            plan.add_operation(operation);
            new_df.add_pending_operation(plan, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            let mut plan = ExecutionPlan::new(&self.id().to_string());
            plan.add_operation(operation);
            self.execute_operation(plan, vec![self.id().to_string()])
        }
    }
}

// Re-export all the types from the new modules for backward compatibility
pub use crate::distributed::expr::{
    ColumnMeta,
    // ExprDataType is already defined in this file, so don't re-export it
    ExprSchema,
    ExprValidator,
    InferredType,
};

// The following is just to ensure documentation clarity in case someone is using
// the old path, but implementation is delegated to the new modules
#[deprecated(
    since = "0.1.0",
    note = "Use the specific modules instead: core::Expr, projection::ColumnProjection, etc."
)]
pub type Deprecated = ();
