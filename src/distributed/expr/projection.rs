//! # Projection Operations
//!
//! This module provides functionality for column projections and user-defined functions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core::Expr;
use super::{schema::ExprSchema, ExprDataType};
use crate::distributed::core::dataframe::DistributedDataFrame;
use crate::distributed::execution::{ExecutionPlan, Operation};
use crate::error::Result;

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

    /// Gets the output name of this projection
    pub fn output_name(&self) -> String {
        match &self.alias {
            Some(alias) => alias.clone(),
            None => match &self.expr {
                Expr::Column(name) => name.clone(),
                _ => {
                    let expr_str = format!("{:?}", self.expr);
                    format!(
                        "expr_{}",
                        expr_str
                            .chars()
                            .filter(|c| c.is_alphanumeric())
                            .collect::<String>()
                    )
                }
            },
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

    /// Validates a set of projections against the schema
    fn validate_projections(
        &self,
        projections: &[ColumnProjection],
        schema: &ExprSchema,
    ) -> Result<()>;
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
            new_df.add_pending_operation(operation, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id().to_string()])
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
            new_df.add_pending_operation(operation, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id().to_string()])
        }
    }

    fn filter_expr(&self, expr: Expr) -> Result<DistributedDataFrame> {
        // Convert the expression to SQL
        let filter_sql = expr.to_string();

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
            new_df.add_pending_operation(operation, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id().to_string()])
        }
    }

    fn validate_projections(
        &self,
        projections: &[ColumnProjection],
        schema: &ExprSchema,
    ) -> Result<()> {
        // Will be implemented later with schema validation
        Ok(())
    }
}
