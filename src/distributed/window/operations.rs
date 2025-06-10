//! # Window Operations for Distributed DataFrames
//!
//! This module provides operations for applying window functions to distributed DataFrames.

use super::core::WindowFunction;
use crate::distributed::dataframe::DistributedDataFrame;
use crate::distributed::execution::{ExecutionPlan, Operation};
use crate::error::Result;

/// Extension trait to add window function capabilities to DistributedDataFrame
pub trait WindowFunctionExt {
    /// Applies window functions to the DataFrame
    fn window(&self, window_functions: &[WindowFunction]) -> Result<DistributedDataFrame>;
}

impl WindowFunctionExt for DistributedDataFrame {
    fn window(&self, window_functions: &[WindowFunction]) -> Result<DistributedDataFrame> {
        // Create a Window operation with the window functions
        let window_functions = window_functions.to_vec();
        let operation = Operation::Window { window_functions };

        if self.is_lazy() {
            let mut new_df = self.clone_empty();
            new_df.add_pending_operation(operation, vec![self.id().to_string()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id().to_string()])
        }
    }
}
