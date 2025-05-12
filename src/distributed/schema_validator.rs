//! # Execution Schema Validation (Legacy)
//!
//! DEPRECATED: This file is maintained for backward compatibility only.
//! Please use the `distributed::schema_validator` module directory structure instead.
//!
//! This module provides validation of execution plans against schemas,
//! ensuring type safety and preventing runtime errors.
//!
//! @deprecated

use std::collections::HashMap;
use std::sync::Arc;
use arrow::datatypes::SchemaRef;

use crate::error::{Result, Error};
use crate::distributed::execution::{ExecutionPlan, Operation};
use crate::distributed::expr::{
    Expr, ExprSchema, ExprValidator, ExprDataType, ColumnProjection
};
use crate::distributed::partition::PartitionSet;

/// Schema validator for execution plans
pub struct SchemaValidator {
    /// Schemas of registered datasets
    schemas: HashMap<String, ExprSchema>,
}

impl SchemaValidator {
    /// Creates a new schema validator
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }
    
    /// Registers a schema for a dataset
    pub fn register_schema(&mut self, name: impl Into<String>, schema: ExprSchema) -> &mut Self {
        self.schemas.insert(name.into(), schema);
        self
    }
    
    /// Registers a schema from an Arrow schema
    #[cfg(feature = "distributed")]
    pub fn register_arrow_schema(&mut self, name: impl Into<String>, schema: SchemaRef) -> Result<&mut Self> {
        let expr_schema = ExprSchema::from_arrow_schema(schema.as_ref())?;
        self.schemas.insert(name.into(), expr_schema);
        Ok(self)
    }
    
    /// Gets a schema by name
    pub fn schema(&self, name: &str) -> Option<&ExprSchema> {
        self.schemas.get(name)
    }
    
    /// Validates an execution plan
    pub fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        // Get schemas for input datasets
        let mut input_schemas = Vec::new();
        for input in plan.inputs() {
            if let Some(schema) = self.schemas.get(input) {
                input_schemas.push(schema);
            } else {
                return Err(Error::InvalidOperation(
                    format!("Schema not found for input dataset: {}", input)
                ));
            }
        }
        
        if input_schemas.is_empty() {
            return Err(Error::InvalidOperation(
                "No input schemas available for validation".to_string()
            ));
        }
        
        // Validate operation against schemas
        match plan.operation() {
            Operation::Select { columns } => {
                self.validate_select(input_schemas[0], columns)
            },
            Operation::Filter { predicate } => {
                self.validate_filter(input_schemas[0], predicate)
            },
            Operation::Join {
                join_type,
                left_keys,
                right_keys,
                ..
            } => {
                if input_schemas.len() < 2 {
                    return Err(Error::InvalidOperation(
                        "Join operation requires at least two input schemas".to_string()
                    ));
                }
                self.validate_join(input_schemas[0], input_schemas[1], left_keys, right_keys)
            },
            Operation::GroupBy {
                keys,
                aggregates,
            } => {
                self.validate_groupby(input_schemas[0], keys, aggregates)
            },
            Operation::OrderBy {
                sort_exprs,
            } => {
                self.validate_orderby(input_schemas[0], sort_exprs)
            },
            Operation::Window {
                window_functions,
            } => {
                self.validate_window(input_schemas[0], window_functions)
            },
            Operation::Custom {
                name,
                params,
            } => {
                match name.as_str() {
                    "select_expr" => {
                        if let Some(projections_json) = params.get("projections") {
                            // Parse projections from JSON
                            let projections: Vec<ColumnProjection> = serde_json::from_str(projections_json)
                                .map_err(|e| Error::DistributedProcessing(
                                    format!("Failed to parse projections: {}", e)
                                ))?;
                            
                            self.validate_select_expr(input_schemas[0], &projections)
                        } else {
                            Err(Error::InvalidOperation(
                                "select_expr operation requires projections parameter".to_string()
                            ))
                        }
                    },
                    "with_column" => {
                        let column_name = params.get("column_name")
                            .ok_or_else(|| Error::InvalidOperation(
                                "with_column operation requires column_name parameter".to_string()
                            ))?;
                        
                        if let Some(projection_json) = params.get("projection") {
                            let projection: ColumnProjection = serde_json::from_str(projection_json)
                                .map_err(|e| Error::DistributedProcessing(
                                    format!("Failed to parse projection: {}", e)
                                ))?;
                            
                            self.validate_with_column(input_schemas[0], column_name, &projection)
                        } else {
                            Err(Error::InvalidOperation(
                                "with_column operation requires projection parameter".to_string()
                            ))
                        }
                    },
                    "create_udf" => {
                        // UDF creation doesn't require schema validation
                        Ok(())
                    },
                    _ => {
                        // Unknown custom operation
                        Err(Error::NotImplemented(
                            format!("Schema validation for custom operation '{}' is not implemented", name)
                        ))
                    }
                }
            },
            Operation::Limit { .. } => {
                // Limit doesn't require schema validation
                Ok(())
            },
        }
    }
    
    /// Validates a SELECT operation
    fn validate_select(&self, schema: &ExprSchema, columns: &[String]) -> Result<()> {
        for column in columns {
            if !schema.has_column(column) {
                return Err(Error::InvalidOperation(
                    format!("Column not found in schema: {}", column)
                ));
            }
        }
        Ok(())
    }
    
    /// Validates a SELECT_EXPR operation
    fn validate_select_expr(&self, schema: &ExprSchema, projections: &[ColumnProjection]) -> Result<()> {
        let validator = ExprValidator::new(schema);
        validator.validate_projections(projections)?;
        Ok(())
    }
    
    /// Validates a WITH_COLUMN operation
    fn validate_with_column(&self, schema: &ExprSchema, column_name: &str, projection: &ColumnProjection) -> Result<()> {
        let validator = ExprValidator::new(schema);
        validator.validate_expr(&projection.expr)?;
        Ok(())
    }
    
    /// Validates a FILTER operation
    fn validate_filter(&self, schema: &ExprSchema, predicate: &str) -> Result<()> {
        // For a simple implementation, we'll just check if it's a valid SQL predicate
        // In a more advanced implementation, we'd parse the predicate into an Expr
        // and validate it against the schema
        
        // Placeholder for SQL predicate validation
        // This is simplified, but can be enhanced with a proper SQL parser
        if predicate.is_empty() {
            return Err(Error::InvalidOperation(
                "Empty predicate in filter operation".to_string()
            ));
        }
        
        // Basic check for balanced parentheses
        let mut paren_count = 0;
        for c in predicate.chars() {
            if c == '(' {
                paren_count += 1;
            } else if c == ')' {
                paren_count -= 1;
                if paren_count < 0 {
                    return Err(Error::InvalidOperation(
                        format!("Unbalanced parentheses in predicate: {}", predicate)
                    ));
                }
            }
        }
        
        if paren_count != 0 {
            return Err(Error::InvalidOperation(
                format!("Unbalanced parentheses in predicate: {}", predicate)
            ));
        }
        
        Ok(())
    }
    
    /// Validates a JOIN operation
    fn validate_join(&self, left_schema: &ExprSchema, right_schema: &ExprSchema, 
                     left_keys: &[String], right_keys: &[String]) -> Result<()> {
        if left_keys.len() != right_keys.len() {
            return Err(Error::InvalidOperation(
                format!("Number of left keys ({}) does not match number of right keys ({})",
                    left_keys.len(), right_keys.len())
            ));
        }
        
        for (left_key, right_key) in left_keys.iter().zip(right_keys.iter()) {
            // Check that keys exist in schemas
            let left_col = left_schema.column(left_key)
                .ok_or_else(|| Error::InvalidOperation(
                    format!("Left join key not found in schema: {}", left_key)
                ))?;
                
            let right_col = right_schema.column(right_key)
                .ok_or_else(|| Error::InvalidOperation(
                    format!("Right join key not found in schema: {}", right_key)
                ))?;
                
            // Check that keys have compatible types
            if !are_join_compatible(&left_col.data_type, &right_col.data_type) {
                return Err(Error::InvalidOperation(
                    format!("Incompatible join key types: {:?} and {:?}",
                        left_col.data_type, right_col.data_type)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validates a GROUP BY operation
    fn validate_groupby(&self, schema: &ExprSchema, keys: &[String], 
                        aggregates: &[crate::distributed::execution::AggregateExpr]) -> Result<()> {
        // Check that keys exist in schema
        for key in keys {
            if !schema.has_column(key) {
                return Err(Error::InvalidOperation(
                    format!("Grouping key not found in schema: {}", key)
                ));
            }
        }
        
        // Check that aggregated columns exist in schema
        for agg in aggregates {
            if !schema.has_column(&agg.input) {
                return Err(Error::InvalidOperation(
                    format!("Aggregated column not found in schema: {}", agg.input)
                ));
            }
            
            // Check that aggregation function is valid for column type
            let col = schema.column(&agg.input).unwrap();
            
            match agg.function.as_str() {
                "min" | "max" | "sum" | "avg" => {
                    // These functions require numeric or date columns
                    match col.data_type {
                        ExprDataType::Integer | ExprDataType::Float | 
                        ExprDataType::Date | ExprDataType::Timestamp => {
                            // Valid types for these aggregations
                        },
                        _ => {
                            return Err(Error::InvalidOperation(
                                format!("Aggregation function '{}' not supported for column type {:?}",
                                    agg.function, col.data_type)
                            ));
                        }
                    }
                },
                "count" => {
                    // Count can be applied to any column
                },
                _ => {
                    // Unknown aggregation function
                    return Err(Error::InvalidOperation(
                        format!("Unknown aggregation function: {}", agg.function)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Validates an ORDER BY operation
    fn validate_orderby(&self, schema: &ExprSchema, 
                      sort_exprs: &[crate::distributed::execution::SortExpr]) -> Result<()> {
        // Check that sort columns exist in schema
        for sort_expr in sort_exprs {
            if !schema.has_column(&sort_expr.column) {
                return Err(Error::InvalidOperation(
                    format!("Sort column not found in schema: {}", sort_expr.column)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validates a WINDOW operation
    fn validate_window(&self, schema: &ExprSchema, 
                     window_functions: &[crate::distributed::window::WindowFunction]) -> Result<()> {
        for wf in window_functions {
            // Check that output column doesn't already exist in schema
            if schema.has_column(&wf.output_column) {
                return Err(Error::InvalidOperation(
                    format!("Output column already exists in schema: {}", wf.output_column)
                ));
            }
            
            // Check that input column exists in schema
            if !schema.has_column(&wf.input_column) {
                return Err(Error::InvalidOperation(
                    format!("Input column not found in schema: {}", wf.input_column)
                ));
            }
            
            // Check that partition columns exist in schema
            for col in &wf.partition_by {
                if !schema.has_column(col) {
                    return Err(Error::InvalidOperation(
                        format!("Partition column not found in schema: {}", col)
                    ));
                }
            }
            
            // Check that order columns exist in schema
            for (col, _) in &wf.order_by {
                if !schema.has_column(col) {
                    return Err(Error::InvalidOperation(
                        format!("Order column not found in schema: {}", col)
                    ));
                }
            }
            
            // Check that function is compatible with column type
            let col = schema.column(&wf.input_column).unwrap();
            
            match wf.function.as_str() {
                "row_number" | "rank" | "dense_rank" => {
                    // These functions can be applied to any column
                },
                "sum" | "avg" | "min" | "max" => {
                    // These functions require numeric or date columns
                    match col.data_type {
                        ExprDataType::Integer | ExprDataType::Float | 
                        ExprDataType::Date | ExprDataType::Timestamp => {
                            // Valid types for these functions
                        },
                        _ => {
                            return Err(Error::InvalidOperation(
                                format!("Window function '{}' not supported for column type {:?}",
                                    wf.function, col.data_type)
                            ));
                        }
                    }
                },
                "first_value" | "last_value" | "nth_value" | "lead" | "lag" => {
                    // These functions can be applied to any column
                },
                _ => {
                    // Unknown window function
                    return Err(Error::InvalidOperation(
                        format!("Unknown window function: {}", wf.function)
                    ));
                }
            }
        }
        
        Ok(())
    }
}

/// Checks if two data types are compatible for joins
fn are_join_compatible(left_type: &ExprDataType, right_type: &ExprDataType) -> bool {
    match (left_type, right_type) {
        // Same types are always compatible
        (a, b) if a == b => true,
        
        // Integer and Float are compatible
        (ExprDataType::Integer, ExprDataType::Float) |
        (ExprDataType::Float, ExprDataType::Integer) => true,
        
        // Date and Timestamp are compatible
        (ExprDataType::Date, ExprDataType::Timestamp) |
        (ExprDataType::Timestamp, ExprDataType::Date) => true,
        
        // Other combinations are not compatible
        _ => false,
    }
}