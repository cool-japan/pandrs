//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Descriptive statistics from describe()
#[derive(Debug, Clone)]
pub struct DescribeStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub max: f64,
}
/// Axis for apply operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Apply function along rows (across columns)
    Rows,
    /// Apply function along columns (down rows)
    Columns,
}
/// Correlation or covariance matrix
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    pub values: Vec<Vec<f64>>,
}
impl CorrelationMatrix {
    /// Get correlation value between two columns
    pub fn get(&self, col1: &str, col2: &str) -> Option<f64> {
        let idx1 = self.columns.iter().position(|c| c == col1)?;
        let idx2 = self.columns.iter().position(|c| c == col2)?;
        Some(self.values[idx1][idx2])
    }
    /// Get the matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (
            self.values.len(),
            self.values.first().map(|v| v.len()).unwrap_or(0),
        )
    }
}
/// Rank computation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankMethod {
    /// Average rank of tied values
    Average,
    /// Minimum rank of tied values
    Min,
    /// Maximum rank of tied values
    Max,
    /// First occurrence gets lower rank
    First,
    /// Dense ranking (no gaps)
    Dense,
}
/// Value types for assign operations
#[derive(Debug, Clone)]
pub enum SeriesValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}
