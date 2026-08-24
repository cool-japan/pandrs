//! # Projection Operations
//!
//! This module provides functionality for column projections and user-defined functions.

use serde::{Deserialize, Serialize};

use std::collections::HashSet;

use super::core::Expr;
use super::validator::ExprValidator;
use super::{schema::ExprSchema, ExprDataType};
use crate::distributed::core::dataframe::DistributedDataFrame;
use crate::distributed::execution::{ExecutionPlan, Operation};
use crate::error::{Error, Result};

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

    /// Converts the column projection to SQL.
    ///
    /// The expression renders column references as quoted identifiers (see
    /// `Expr`'s `Display`), and the alias is quoted here, so the whole
    /// projection is safe to splice into a `SELECT` list.
    pub fn to_sql(&self) -> String {
        match &self.alias {
            Some(alias) => format!("{} AS \"{}\"", self.expr, alias.replace('"', "\"\"")),
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
    fn filter_expr(&mut self, expr: Expr) -> Result<DistributedDataFrame>;

    /// Creates user-defined functions
    fn create_udf(&self, udfs: &[UdfDefinition]) -> Result<DistributedDataFrame>;

    /// Validates a set of projections against the schema.
    ///
    /// Every projection is checked, and the first problem found is returned as
    /// `Err(Error::InvalidOperation)`:
    ///
    /// * each projection expression is type-checked against `schema` (via
    ///   [`ExprValidator`]), so a reference to a column absent from the schema
    ///   or a type-incompatible operation is rejected;
    /// * output names (alias, or the bare column name when no alias is given)
    ///   must be unique across the projection set — a collision is rejected;
    /// * a supplied alias must be well-formed: neither empty nor whitespace.
    ///
    /// Note that `ExprValidator::validate_expr` cannot infer a type for a bare
    /// `NULL` literal without context, so a projection such as `NULL AS x` is
    /// reported as invalid rather than silently accepted.
    ///
    /// This is a caller-invoked, client-side pre-check: you supply the input
    /// `schema` and call it explicitly (for example before `select_expr`) to
    /// reject an obviously-malformed projection set early. It is deliberately
    /// NOT applied automatically inside `select_expr`/`with_column`, because
    /// those accept user-defined functions (registered via `create_udf`) and
    /// the full SQL function catalog, whereas the [`ExprValidator`] built here
    /// knows only a fixed set of built-in functions and no registered UDFs — so
    /// gating those methods on it would reject valid projections. Authoritative
    /// validation of a projection happens when the operation is executed by the
    /// engine, which has the complete schema, function catalog, and registered
    /// UDFs. To pre-validate expressions that use UDFs, construct an
    /// [`ExprValidator`] directly and register them with `ExprValidator::add_udf`
    /// before validating.
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
            let mut plan = ExecutionPlan::new(self.id());
            plan.add_operation(operation);
            new_df.add_pending_operation(plan, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            let mut plan = ExecutionPlan::new(self.id());
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
            let mut plan = ExecutionPlan::new(self.id());
            plan.add_operation(operation);
            new_df.add_pending_operation(plan, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            let mut plan = ExecutionPlan::new(self.id());
            plan.add_operation(operation);
            self.execute_operation(plan, vec![self.id().to_string()])
        }
    }

    fn filter_expr(&mut self, expr: Expr) -> Result<DistributedDataFrame> {
        // Render the expression as SQL via its `Display` impl. The previous
        // `format!("{:?}", expr)` emitted Rust debug syntax
        // (`BinaryOp { left: Column("a"), .. }`), which is not valid SQL and
        // always failed to parse.
        //
        // NOTE: `expr.to_string()` would resolve to `Expr`'s inherent
        // `to_string(self) -> Expr` (a CAST-to-string builder), not the
        // `Display`-based `ToString`, so we format explicitly.
        let filter_sql = format!("{}", expr);

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
            let mut plan = ExecutionPlan::new(self.id());
            plan.add_operation(operation);
            new_df.add_pending_operation(plan, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            let mut plan = ExecutionPlan::new(self.id());
            plan.add_operation(operation);
            self.execute_operation(plan, vec![self.id().to_string()])
        }
    }

    fn validate_projections(
        &self,
        projections: &[ColumnProjection],
        schema: &ExprSchema,
    ) -> Result<()> {
        // Type-check every projection expression against the schema.
        // `ExprValidator` resolves column references (rejecting any column
        // absent from `schema`) and verifies operand/argument types, returning
        // `Err` on the first problem it finds.
        let validator = ExprValidator::new(schema);

        // Track output names to reject collisions. Unlike
        // `ExprValidator::validate_projections` (which keys a `HashMap` by
        // output name and therefore silently overwrites duplicates), we detect
        // the collision explicitly so a genuinely-invalid projection set is
        // rejected instead of quietly dropping a column.
        let mut output_names: HashSet<String> = HashSet::with_capacity(projections.len());

        for projection in projections {
            validator.validate_expr(&projection.expr)?;

            // A supplied alias must be well-formed: non-empty and not pure
            // whitespace, otherwise the projected column would be unnameable.
            if let Some(alias) = &projection.alias {
                if alias.trim().is_empty() {
                    return Err(Error::InvalidOperation(
                        "Projection alias must not be empty or whitespace".to_string(),
                    ));
                }
            }

            // Output names must be unique across the projection set; two
            // columns resolving to the same name would collide in the result
            // schema.
            let output_name = projection.output_name();
            if !output_names.insert(output_name.clone()) {
                return Err(Error::InvalidOperation(format!(
                    "Duplicate projection output name: '{}'",
                    output_name
                )));
            }
        }

        Ok(())
    }
}
