use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// View of a DataFrame column
#[derive(Debug, Clone)]
pub struct ColumnView<'a> {
    /// Reference to the original column
    values: &'a [String],
    /// Column name
    name: Option<String>,
    /// Is the view read-only?
    read_only: bool,
}

impl<'a> ColumnView<'a> {
    /// Create a new column view
    pub fn new(values: &'a [String], name: Option<String>, read_only: bool) -> Self {
        Self {
            values,
            name,
            read_only,
        }
    }

    /// Get the name of the column
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the values of the column
    pub fn values(&self) -> &[String] {
        self.values
    }

    /// Is the view read-only?
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }
}

/// View functionality for DataFrames
pub trait ViewExt {
    /// Get a column view
    fn get_column_view<'a>(&'a self, column_name: &str) -> Result<ColumnView<'a>>;

    /// Get a head view (first n rows)
    fn head(&self, n: usize) -> Result<Self>
    where
        Self: Sized;

    /// Get a tail view (last n rows)
    fn tail(&self, n: usize) -> Result<Self>
    where
        Self: Sized;
}

impl ViewExt for DataFrame {
    fn get_column_view<'a>(&'a self, column_name: &str) -> Result<ColumnView<'a>> {
        // This would be implemented later
        Err(Error::NotImplemented(
            "get_column_view not implemented yet".to_string(),
        ))
    }

    fn head(&self, n: usize) -> Result<Self> {
        // This would be implemented later
        Ok(DataFrame::new())
    }

    fn tail(&self, n: usize) -> Result<Self> {
        // This would be implemented later
        Ok(DataFrame::new())
    }
}
