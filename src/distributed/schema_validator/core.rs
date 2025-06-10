//! # Core Schema Validator Types
//!
//! This module provides core types and structures for schema validation
//! in distributed processing.

use arrow::datatypes::SchemaRef;
use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::expr::ExprSchema;
use crate::error::{Error, Result};

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
    pub fn register_arrow_schema(
        &mut self,
        name: impl Into<String>,
        schema: SchemaRef,
    ) -> Result<&mut Self> {
        let expr_schema = ExprSchema::from_arrow_schema(schema.as_ref())?;
        self.schemas.insert(name.into(), expr_schema);
        Ok(self)
    }

    /// Gets a schema by name
    pub fn schema(&self, name: &str) -> Option<&ExprSchema> {
        self.schemas.get(name)
    }

    /// Gets all registered schemas
    pub fn schemas(&self) -> &HashMap<String, ExprSchema> {
        &self.schemas
    }
}
