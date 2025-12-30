//! Query engine and DataFrame integration
//!
//! This module provides the main query engine that integrates all components
//! and the extension traits for DataFrame query functionality.

use super::ast::{Expr, LiteralValue};
use super::evaluator::{Evaluator, JitEvaluator, QueryContext};
use super::lexer_parser::{Lexer, Parser};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Query engine for DataFrames
pub struct QueryEngine {
    context: QueryContext,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self {
            context: QueryContext::new(),
        }
    }

    /// Create a query engine with custom context
    pub fn with_context(context: QueryContext) -> Self {
        Self { context }
    }

    /// Execute a query on a DataFrame
    pub fn query(&self, dataframe: &DataFrame, query_str: &str) -> Result<DataFrame> {
        // Tokenize the query string
        let input_str: &'static str = unsafe { std::mem::transmute(query_str) };
        let mut lexer = Lexer::new(input_str);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token()?;
            let is_eof = matches!(token, super::ast::Token::Eof);
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        // Parse tokens into AST
        let mut parser = Parser::new(tokens);
        let expr = parser.parse()?;

        // Use JIT evaluator for best performance
        let evaluator = JitEvaluator::new(dataframe, &self.context);
        let mask = evaluator.evaluate_query_jit(&expr)?;

        // Filter DataFrame based on mask
        self.filter_dataframe_by_mask(dataframe, &mask)
    }

    /// Filter DataFrame using boolean mask
    fn filter_dataframe_by_mask(&self, dataframe: &DataFrame, mask: &[bool]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        // Get indices where mask is true
        let selected_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &include)| if include { Some(idx) } else { None })
            .collect();

        // Create filtered columns
        for col_name in dataframe.column_names() {
            let column_values = dataframe.get_column_string_values(&col_name)?;
            let filtered_values: Vec<String> = selected_indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            let filtered_series = Series::new(filtered_values, Some(col_name.clone()))?;
            result.add_column(col_name, filtered_series)?;
        }

        Ok(result)
    }

    /// Add a variable to the query context
    pub fn set_variable(&mut self, name: String, value: LiteralValue) {
        self.context.set_variable(name, value);
    }

    /// Add a custom function to the query context
    pub fn add_function<F>(&mut self, name: String, func: F)
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.context.add_function(name, func);
    }
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait to add query functionality to DataFrame
pub trait QueryExt {
    /// Execute a query expression on the DataFrame
    fn query(&self, query_str: &str) -> Result<DataFrame>;

    /// Execute a query with custom context
    fn query_with_context(&self, query_str: &str, context: &QueryContext) -> Result<DataFrame>;

    /// Evaluate an expression and return the result as a new column
    fn eval(&self, expr_str: &str, result_column: &str) -> Result<DataFrame>;
}

impl QueryExt for DataFrame {
    fn query(&self, query_str: &str) -> Result<DataFrame> {
        let engine = QueryEngine::new();
        engine.query(self, query_str)
    }

    fn query_with_context(&self, query_str: &str, context: &QueryContext) -> Result<DataFrame> {
        let engine = QueryEngine::with_context(context.clone());
        engine.query(self, query_str)
    }

    fn eval(&self, expr_str: &str, result_column: &str) -> Result<DataFrame> {
        // This would evaluate an expression and add it as a new column
        // For now, implement basic version that parses and evaluates
        let mut result = self.clone();

        // Parse and evaluate expression for each row
        let engine = QueryEngine::new();
        let input_str: &'static str = unsafe { std::mem::transmute(expr_str) };
        let mut lexer = Lexer::new(input_str);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token()?;
            let is_eof = matches!(token, super::ast::Token::Eof);
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        let mut parser = Parser::new(tokens);
        let expr = parser.parse()?;

        let evaluator = Evaluator::new(self, &engine.context);
        let mut result_values = Vec::new();

        for row_idx in 0..self.row_count() {
            let value = evaluator.evaluate_expression_for_row(&expr, row_idx)?;
            match value {
                LiteralValue::Number(n) => result_values.push(n.to_string()),
                LiteralValue::String(s) => result_values.push(s),
                LiteralValue::Boolean(b) => result_values.push(b.to_string()),
            }
        }

        let result_series = Series::new(result_values, Some(result_column.to_string()))?;
        result.add_column(result_column.to_string(), result_series)?;

        Ok(result)
    }
}
