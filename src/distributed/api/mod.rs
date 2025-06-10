//! # High-Level API for Distributed Processing
//!
//! This module provides high-level functions and utilities for working with
//! distributed data processing in PandRS.

use crate::dataframe::base::DataFrame;
use crate::distributed::core::{
    DistributedConfig, DistributedContext, DistributedDataFrame, ToDistributed,
};
use crate::error::Result;

/// Creates a new distributed context with the default configuration
pub fn create_default_context() -> Result<DistributedContext> {
    let config = DistributedConfig::default();
    DistributedContext::new(config)
}

/// Creates a new distributed context with the specified configuration
pub fn create_context(config: DistributedConfig) -> Result<DistributedContext> {
    DistributedContext::new(config)
}

/// Converts a DataFrame to a distributed DataFrame using default configuration
pub fn to_distributed(df: &DataFrame) -> Result<DistributedDataFrame> {
    let config = DistributedConfig::default();
    df.to_distributed(config)
}

/// Reads a CSV file into a distributed DataFrame
pub fn read_csv(
    path: &str,
    context: &DistributedContext,
    options: Option<crate::io::csv::CsvReadOptions>,
) -> Result<DistributedDataFrame> {
    context.read_csv(path, options)
}

/// Reads a Parquet file into a distributed DataFrame
#[cfg(feature = "parquet")]
pub fn read_parquet(path: &str, context: &DistributedContext) -> Result<DistributedDataFrame> {
    context.read_parquet(path)
}

/// Reads from a SQL table into a distributed DataFrame
#[cfg(feature = "sql")]
pub fn read_sql(
    query: &str,
    url: &str,
    context: &DistributedContext,
) -> Result<DistributedDataFrame> {
    context.read_sql(query, url)
}

/// Creates an empty distributed DataFrame
pub fn empty_distributed(context: &DistributedContext) -> Result<DistributedDataFrame> {
    context.empty()
}
