//! # Schema Validation for Expressions
//!
//! This module provides schema validation capabilities for the expression system,
//! ensuring type safety and preventing runtime errors in distributed operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::ExprDataType;
use crate::error::{Error, Result};

/// Represents a column's metadata including its data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMeta {
    /// Name of the column
    pub name: String,
    /// Data type of the column
    pub data_type: ExprDataType,
    /// Whether the column can contain null values
    pub nullable: bool,
    /// Optional description of the column
    pub description: Option<String>,
}

impl ColumnMeta {
    /// Creates a new column metadata object
    pub fn new(
        name: impl Into<String>,
        data_type: ExprDataType,
        nullable: bool,
        description: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
            description,
        }
    }
}

/// Schema information for validating expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprSchema {
    /// Metadata for each column in the schema
    columns: HashMap<String, ColumnMeta>,
}

impl ExprSchema {
    /// Creates a new empty schema
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
        }
    }

    /// Adds a column to the schema
    pub fn add_column(&mut self, meta: ColumnMeta) -> &mut Self {
        self.columns.insert(meta.name.clone(), meta);
        self
    }

    /// Gets the metadata for a column
    pub fn column(&self, name: &str) -> Option<&ColumnMeta> {
        self.columns.get(name)
    }

    /// Gets all column names
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    /// Checks if the schema contains a column
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Gets the number of columns in the schema
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Checks if the schema is empty
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Gets all columns in the schema
    pub fn columns(&self) -> &HashMap<String, ColumnMeta> {
        &self.columns
    }

    /// Converts from an Arrow schema
    #[cfg(feature = "distributed")]
    pub fn from_arrow_schema(schema: &arrow::datatypes::Schema) -> Result<Self> {
        let mut result = Self::new();

        for field in schema.fields() {
            let name = field.name().clone();
            let data_type = arrow_type_to_expr_type(field.data_type())?;
            let nullable = field.is_nullable();

            let meta = ColumnMeta::new(name, data_type, nullable, None);
            result.add_column(meta);
        }

        Ok(result)
    }

    /// Converts to an Arrow schema
    #[cfg(feature = "distributed")]
    pub fn to_arrow_schema(&self) -> Result<arrow::datatypes::Schema> {
        let mut fields = Vec::with_capacity(self.columns.len());

        for meta in self.columns.values() {
            let arrow_type = expr_type_to_arrow_type(&meta.data_type)?;
            let field = arrow::datatypes::Field::new(&meta.name, arrow_type, meta.nullable);
            fields.push(field);
        }

        Ok(arrow::datatypes::Schema::new(fields))
    }
}

/// Converts from an Arrow data type to an expression data type
#[cfg(feature = "distributed")]
pub fn arrow_type_to_expr_type(arrow_type: &arrow::datatypes::DataType) -> Result<ExprDataType> {
    match arrow_type {
        arrow::datatypes::DataType::Boolean => Ok(ExprDataType::Boolean),
        arrow::datatypes::DataType::Int8
        | arrow::datatypes::DataType::Int16
        | arrow::datatypes::DataType::Int32
        | arrow::datatypes::DataType::Int64
        | arrow::datatypes::DataType::UInt8
        | arrow::datatypes::DataType::UInt16
        | arrow::datatypes::DataType::UInt32
        | arrow::datatypes::DataType::UInt64 => Ok(ExprDataType::Integer),
        arrow::datatypes::DataType::Float16
        | arrow::datatypes::DataType::Float32
        | arrow::datatypes::DataType::Float64 => Ok(ExprDataType::Float),
        arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => {
            Ok(ExprDataType::String)
        }
        arrow::datatypes::DataType::Date32 | arrow::datatypes::DataType::Date64 => {
            Ok(ExprDataType::Date)
        }
        arrow::datatypes::DataType::Timestamp(_, _) => Ok(ExprDataType::Timestamp),
        _ => Err(Error::NotImplemented(format!(
            "Conversion of Arrow data type {:?} to expression type is not implemented",
            arrow_type
        ))),
    }
}

/// Converts from an expression data type to an Arrow data type
#[cfg(feature = "distributed")]
pub fn expr_type_to_arrow_type(expr_type: &ExprDataType) -> Result<arrow::datatypes::DataType> {
    match expr_type {
        ExprDataType::Boolean => Ok(arrow::datatypes::DataType::Boolean),
        ExprDataType::Integer => Ok(arrow::datatypes::DataType::Int64),
        ExprDataType::Float => Ok(arrow::datatypes::DataType::Float64),
        ExprDataType::String => Ok(arrow::datatypes::DataType::Utf8),
        ExprDataType::Date => Ok(arrow::datatypes::DataType::Date32),
        ExprDataType::Timestamp => Ok(arrow::datatypes::DataType::Timestamp(
            arrow::datatypes::TimeUnit::Microsecond,
            None,
        )),
    }
}
