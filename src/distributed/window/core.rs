//! # Core Window Function Types and Structures
//!
//! This module defines the core types and structures for window functions
//! in distributed processing.

use std::fmt;

/// Types of window frame boundaries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameBoundary {
    /// Current row
    CurrentRow,
    /// Number of rows before current row
    Preceding(usize),
    /// Number of rows after current row
    Following(usize),
    /// Unbounded preceding (all rows from start)
    UnboundedPreceding,
    /// Unbounded following (all rows to end)
    UnboundedFollowing,
}

impl fmt::Display for WindowFrameBoundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrentRow => write!(f, "CURRENT ROW"),
            Self::Preceding(n) => write!(f, "{} PRECEDING", n),
            Self::Following(n) => write!(f, "{} FOLLOWING", n),
            Self::UnboundedPreceding => write!(f, "UNBOUNDED PRECEDING"),
            Self::UnboundedFollowing => write!(f, "UNBOUNDED FOLLOWING"),
        }
    }
}

/// Window frame type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameType {
    /// Rows-based window (count of rows)
    Rows,
    /// Range-based window (value range)
    Range,
}

impl fmt::Display for WindowFrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rows => write!(f, "ROWS"),
            Self::Range => write!(f, "RANGE"),
        }
    }
}

/// Definition of a window frame
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowFrame {
    /// Frame type (rows or range)
    frame_type: WindowFrameType,
    /// Start boundary
    start: WindowFrameBoundary,
    /// End boundary
    end: WindowFrameBoundary,
}

impl WindowFrame {
    /// Creates a new window frame
    pub fn new(
        frame_type: WindowFrameType,
        start: WindowFrameBoundary,
        end: WindowFrameBoundary,
    ) -> Self {
        Self {
            frame_type,
            start,
            end,
        }
    }

    /// Creates a rows-based window frame
    pub fn rows(start: WindowFrameBoundary, end: WindowFrameBoundary) -> Self {
        Self::new(WindowFrameType::Rows, start, end)
    }

    /// Creates a range-based window frame
    pub fn range(start: WindowFrameBoundary, end: WindowFrameBoundary) -> Self {
        Self::new(WindowFrameType::Range, start, end)
    }

    /// Creates a frame for n preceding rows
    pub fn preceding(n: usize) -> Self {
        Self::rows(
            WindowFrameBoundary::Preceding(n),
            WindowFrameBoundary::CurrentRow,
        )
    }

    /// Creates a frame for all preceding rows
    pub fn unbounded_preceding() -> Self {
        Self::rows(
            WindowFrameBoundary::UnboundedPreceding,
            WindowFrameBoundary::CurrentRow,
        )
    }

    /// Creates a frame for current row only
    pub fn current_row() -> Self {
        Self::rows(
            WindowFrameBoundary::CurrentRow,
            WindowFrameBoundary::CurrentRow,
        )
    }

    /// Creates a frame that spans n preceding and m following rows
    pub fn surrounding(preceding: usize, following: usize) -> Self {
        Self::rows(
            WindowFrameBoundary::Preceding(preceding),
            WindowFrameBoundary::Following(following),
        )
    }

    /// Creates a frame that spans the entire partition
    pub fn entire_partition() -> Self {
        Self::rows(
            WindowFrameBoundary::UnboundedPreceding,
            WindowFrameBoundary::UnboundedFollowing,
        )
    }

    /// Converts the window frame to SQL syntax
    pub fn to_sql(&self) -> String {
        format!(
            "{} BETWEEN {} AND {}",
            self.frame_type, self.start, self.end
        )
    }
}

/// Represents a window function specification
#[derive(Debug, Clone)]
pub struct WindowFunction {
    /// Function name
    pub function: String,
    /// Input columns
    pub inputs: Vec<String>,
    /// Output column name
    pub output: String,
    /// Partition by columns
    pub partition_by: Vec<String>,
    /// Order by columns
    pub order_by: Vec<(String, bool)>, // (column, ascending)
    /// Window frame
    pub frame: Option<WindowFrame>,
}

impl WindowFunction {
    /// Creates a new window function
    pub fn new(
        function: &str,
        inputs: &[&str],
        output: &str,
        partition_by: &[&str],
        order_by: &[(&str, bool)],
        frame: Option<WindowFrame>,
    ) -> Self {
        Self {
            function: function.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            output: output.to_string(),
            partition_by: partition_by.iter().map(|s| s.to_string()).collect(),
            order_by: order_by.iter().map(|(s, b)| (s.to_string(), *b)).collect(),
            frame,
        }
    }

    /// Creates a window function with a single input column
    pub fn single_input(
        function: &str,
        input: &str,
        output: &str,
        partition_by: &[&str],
        order_by: &[(&str, bool)],
        frame: Option<WindowFrame>,
    ) -> Self {
        Self::new(function, &[input], output, partition_by, order_by, frame)
    }

    /// Converts the window function to an SQL expression
    pub fn to_sql(&self) -> String {
        let inputs = if self.inputs.is_empty() {
            "*".to_string()
        } else {
            self.inputs.join(", ")
        };

        let mut result = format!("{}({}) OVER (", self.function, inputs);

        if !self.partition_by.is_empty() {
            result.push_str(&format!("PARTITION BY {} ", self.partition_by.join(", ")));
        }

        if !self.order_by.is_empty() {
            result.push_str("ORDER BY ");
            let mut order_parts = Vec::new();

            for (col, asc) in &self.order_by {
                let direction = if *asc { "ASC" } else { "DESC" };
                order_parts.push(format!("{} {}", col, direction));
            }

            result.push_str(&order_parts.join(", "));
            result.push(' ');
        }

        if let Some(frame) = &self.frame {
            result.push_str(&frame.to_sql());
        }

        result.push_str(")");

        format!("{} AS {}", result, self.output)
    }
}
