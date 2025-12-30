//! # Core Plan Explanation Types
//!
//! This module provides core types and structures for query plan explanation
//! in distributed processing.

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::distributed::statistics::TableStatistics;

/// Formats for query plan explanation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainFormat {
    /// Text format
    Text,
    /// JSON format
    Json,
    /// Dot format (for visualization)
    Dot,
}

/// Options for query plan explanation
#[derive(Debug, Clone)]
pub struct ExplainOptions {
    /// Format of the explanation
    pub format: ExplainFormat,
    /// Whether to include statistics
    pub with_statistics: bool,
    /// Whether to show the optimized plan
    pub optimized: bool,
    /// Whether to analyze the plan
    pub analyze: bool,
}

impl Default for ExplainOptions {
    fn default() -> Self {
        Self {
            format: ExplainFormat::Text,
            with_statistics: false,
            optimized: true,
            analyze: false,
        }
    }
}

/// Query plan node types
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Scan operation
    Scan {
        /// Table name
        table: String,
        /// Columns to scan
        columns: Vec<String>,
        /// Statistics for the table
        statistics: Option<TableStatistics>,
    },
    /// Project operation
    Project {
        /// Output columns
        columns: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Filter operation
    Filter {
        /// Filter predicate
        predicate: String,
        /// Input node
        input: Box<PlanNode>,
        /// Estimated selectivity (0.0-1.0)
        selectivity: Option<f64>,
    },
    /// Join operation
    Join {
        /// Join type
        join_type: String,
        /// Left input
        left: Box<PlanNode>,
        /// Right input
        right: Box<PlanNode>,
        /// Join keys
        keys: Vec<(String, String)>,
    },
    /// Aggregate operation
    Aggregate {
        /// Grouping keys
        keys: Vec<String>,
        /// Aggregation expressions
        aggregates: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Sort operation
    Sort {
        /// Sort expressions
        sort_exprs: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Limit operation
    Limit {
        /// Maximum number of rows
        limit: usize,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Window operation
    Window {
        /// Window functions
        window_functions: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Custom operation
    Custom {
        /// Operation name
        name: String,
        /// Parameters
        params: HashMap<String, String>,
        /// Input node
        input: Box<PlanNode>,
    },
}

impl Display for PlanNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain(&ExplainOptions::default()))
    }
}

impl PlanNode {
    /// Explains the plan in the given format
    pub fn explain(&self, options: &ExplainOptions) -> String {
        match options.format {
            ExplainFormat::Text => self.explain_text(options, 0),
            ExplainFormat::Json => self.explain_json(options),
            ExplainFormat::Dot => self.explain_dot(options),
        }
    }
}
