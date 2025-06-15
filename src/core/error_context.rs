//! Enhanced error context system for PandRS
//!
//! This module provides rich error context, recovery suggestions, and debugging information
//! to help users understand and resolve errors more effectively.

use std::collections::HashMap;
use std::time::SystemTime;

/// Error context with rich debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The operation that failed
    pub operation: &'static str,
    /// DataFrame shape when error occurred
    pub dataframe_shape: Option<(usize, usize)>,
    /// Column names involved in the operation
    pub column_names: Vec<String>,
    /// Suggested fixes for the error
    pub suggested_fixes: Vec<String>,
    /// Performance hint related to the error
    pub performance_hint: Option<String>,
    /// Stack trace information
    pub stack_trace: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Error severity level
    pub severity: ErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Informational - operation might continue
    Info,
    /// Warning - operation can continue but with reduced functionality
    Warning,
    /// Error - operation cannot continue
    Error,
    /// Critical - system state may be compromised
    Critical,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: &'static str) -> Self {
        Self {
            operation,
            dataframe_shape: None,
            column_names: Vec::new(),
            suggested_fixes: Vec::new(),
            performance_hint: None,
            stack_trace: Vec::new(),
            metadata: HashMap::new(),
            timestamp: SystemTime::now(),
            severity: ErrorSeverity::Error,
        }
    }

    /// Add DataFrame shape context
    pub fn with_shape(mut self, shape: (usize, usize)) -> Self {
        self.dataframe_shape = Some(shape);
        self
    }

    /// Add column names context
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.column_names = columns;
        self
    }

    /// Add a suggested fix
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggested_fixes.push(suggestion);
        self
    }

    /// Add multiple suggested fixes
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggested_fixes.extend(suggestions);
        self
    }

    /// Add performance hint
    pub fn with_performance_hint(mut self, hint: String) -> Self {
        self.performance_hint = Some(hint);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set error severity
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add stack trace entry
    pub fn with_stack_entry(mut self, entry: String) -> Self {
        self.stack_trace.push(entry);
        self
    }

    /// Generate human-readable error summary
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Error in operation '{}' at {:?}\n",
            self.operation, self.timestamp
        );

        if let Some(shape) = self.dataframe_shape {
            summary.push_str(&format!("DataFrame shape: {:?}\n", shape));
        }

        if !self.column_names.is_empty() {
            summary.push_str(&format!("Columns involved: {:?}\n", self.column_names));
        }

        if !self.suggested_fixes.is_empty() {
            summary.push_str("\nSuggested fixes:\n");
            for (i, fix) in self.suggested_fixes.iter().enumerate() {
                summary.push_str(&format!("  {}. {}\n", i + 1, fix));
            }
        }

        if let Some(ref hint) = self.performance_hint {
            summary.push_str(&format!("\nPerformance hint: {}\n", hint));
        }

        if !self.metadata.is_empty() {
            summary.push_str("\nAdditional context:\n");
            for (key, value) in &self.metadata {
                summary.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        summary
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new("unknown")
    }
}

/// Trait for error recovery and suggestion
pub trait ErrorRecovery {
    /// Suggest fixes for the error
    fn suggest_fixes(&self) -> Vec<String>;

    /// Check if error can be automatically recovered
    fn can_auto_recover(&self) -> bool;

    /// Attempt automatic recovery (returns recovered data or new error)
    fn attempt_recovery(
        &self,
    ) -> std::result::Result<Option<Box<dyn std::any::Any>>, crate::core::error::Error>;

    /// Get error context
    fn error_context(&self) -> Option<&ErrorContext>;
}

/// Error recovery helpers for common PandRS errors
pub struct ErrorRecoveryHelper;

impl ErrorRecoveryHelper {
    /// Generate suggestions for column not found errors
    pub fn column_not_found_suggestions(
        missing_column: &str,
        available_columns: &[String],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        suggestions.push(format!("Column '{}' not found", missing_column));
        suggestions.push(format!("Available columns: {:?}", available_columns));

        // Find similar column names (simple similarity)
        let similar_columns: Vec<&String> = available_columns
            .iter()
            .filter(|col| {
                let similarity = Self::string_similarity(missing_column, col);
                similarity > 0.5
            })
            .collect();

        if !similar_columns.is_empty() {
            suggestions.push(format!("Did you mean one of: {:?}", similar_columns));
        }

        suggestions.push("Use .columns() to list all available columns".to_string());
        suggestions.push("Check for typos in column name".to_string());

        suggestions
    }

    /// Generate suggestions for shape mismatch errors
    pub fn shape_mismatch_suggestions(
        expected: (usize, usize),
        actual: (usize, usize),
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        suggestions.push(format!("Expected shape {:?}, got {:?}", expected, actual));

        if expected.0 != actual.0 {
            suggestions.push("Row count mismatch - check data alignment".to_string());
            if actual.0 > expected.0 {
                suggestions.push("Consider using .head() to limit rows".to_string());
            } else {
                suggestions.push("Data may be incomplete - check data source".to_string());
            }
        }

        if expected.1 != actual.1 {
            suggestions.push("Column count mismatch - check column selection".to_string());
            suggestions.push("Use .select() to choose specific columns".to_string());
        }

        suggestions.push("Use .shape() to check DataFrame dimensions".to_string());

        suggestions
    }

    /// Generate suggestions for type mismatch errors
    pub fn type_mismatch_suggestions(column: &str, expected: &str, found: &str) -> Vec<String> {
        let mut suggestions = vec![
            format!(
                "Column '{}' type mismatch: expected {}, found {}",
                column, expected, found
            ),
            format!("Use .dtypes() to check column types"),
            format!("Consider type conversion with .astype()"),
        ];

        // Specific suggestions based on types
        match (expected, found) {
            ("int64", "float64") => {
                suggestions.push("Data contains floating point values - consider using .round() or .astype('int64')".to_string());
            }
            ("float64", "int64") => {
                suggestions.push("Converting to float - this is usually safe".to_string());
            }
            ("string", _) => {
                suggestions.push("Use .astype('string') to convert to string type".to_string());
            }
            (_, "string") => {
                suggestions
                    .push("Parse string data with appropriate conversion functions".to_string());
            }
            _ => {}
        }

        suggestions
    }

    /// Generate performance hints based on operation and data characteristics
    pub fn performance_hints(
        operation: &str,
        data_size: Option<usize>,
        column_count: Option<usize>,
    ) -> Option<String> {
        match operation {
            "groupby" => {
                if let Some(size) = data_size {
                    if size > 1_000_000 {
                        return Some("Large dataset detected - consider using .sample() for testing or distributed processing".to_string());
                    }
                }
                if let Some(cols) = column_count {
                    if cols > 50 {
                        return Some("Many columns detected - consider selecting only needed columns before groupby".to_string());
                    }
                }
                None
            }
            "join" => {
                if let Some(size) = data_size {
                    if size > 10_000_000 {
                        return Some("Large join detected - ensure join columns are indexed for better performance".to_string());
                    }
                }
                None
            }
            "sort" => {
                if let Some(size) = data_size {
                    if size > 1_000_000 {
                        return Some("Large sort operation - consider using partial sorting with .nlargest() or .nsmallest()".to_string());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Simple string similarity using Levenshtein distance
    fn string_similarity(s1: &str, s2: &str) -> f64 {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        let max_len = len1.max(len2);
        let distance = Self::levenshtein_distance(s1, s2);

        1.0 - (distance as f64 / max_len as f64)
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }
}

/// Builder for creating error contexts
pub struct ErrorContextBuilder {
    context: ErrorContext,
}

impl ErrorContextBuilder {
    /// Start building error context for an operation
    pub fn new(operation: &'static str) -> Self {
        Self {
            context: ErrorContext::new(operation),
        }
    }

    /// Add DataFrame shape
    pub fn shape(mut self, rows: usize, cols: usize) -> Self {
        self.context.dataframe_shape = Some((rows, cols));
        self
    }

    /// Add column information
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.context.column_names = columns;
        self
    }

    /// Add a suggestion
    pub fn suggest(mut self, suggestion: &str) -> Self {
        self.context.suggested_fixes.push(suggestion.to_string());
        self
    }

    /// Add performance hint
    pub fn hint(mut self, hint: &str) -> Self {
        self.context.performance_hint = Some(hint.to_string());
        self
    }

    /// Add metadata
    pub fn meta(mut self, key: &str, value: &str) -> Self {
        self.context
            .metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Set severity
    pub fn severity(mut self, severity: ErrorSeverity) -> Self {
        self.context.severity = severity;
        self
    }

    /// Build the context
    pub fn build(self) -> ErrorContext {
        self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContextBuilder::new("test_operation")
            .shape(100, 5)
            .columns(vec!["col1".to_string(), "col2".to_string()])
            .suggest("Try using .select() to choose columns")
            .hint("Large dataset - consider sampling")
            .meta("data_type", "numeric")
            .severity(ErrorSeverity::Warning)
            .build();

        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.dataframe_shape, Some((100, 5)));
        assert_eq!(context.column_names.len(), 2);
        assert_eq!(context.suggested_fixes.len(), 1);
        assert!(context.performance_hint.is_some());
        assert_eq!(context.severity, ErrorSeverity::Warning);
    }

    #[test]
    fn test_column_not_found_suggestions() {
        let available = vec!["name".to_string(), "age".to_string(), "salary".to_string()];
        let suggestions = ErrorRecoveryHelper::column_not_found_suggestions("nam", &available);

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("nam")));
        assert!(suggestions.iter().any(|s| s.contains("name")));
    }

    #[test]
    fn test_string_similarity() {
        assert!(ErrorRecoveryHelper::string_similarity("hello", "helo") >= 0.8);
        assert!(ErrorRecoveryHelper::string_similarity("hello", "world") < 0.5);
        assert_eq!(ErrorRecoveryHelper::string_similarity("same", "same"), 1.0);
    }
}
