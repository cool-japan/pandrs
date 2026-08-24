//! Query engine and DataFrame integration
//!
//! This module provides the main query engine that integrates all components
//! and the extension traits for DataFrame query functionality.

use super::ast::{Expr, LiteralValue, Token};
use super::evaluator::{Evaluator, QueryContext};
use super::filter::take_rows;
use super::lexer_parser::{Lexer, Parser};
use super::vectorized::{ValueVec, VectorizedEvaluator};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;

/// Tokenize and parse a query expression.
///
/// The expression is borrowed for the duration of the call; no lifetime
/// laundering is involved.
fn parse_expression(query_str: &str) -> Result<Expr> {
    let mut lexer = Lexer::new(query_str);
    let mut tokens = Vec::new();

    loop {
        let token = lexer.next_token()?;
        let is_eof = matches!(token, Token::Eof);
        tokens.push(token);
        if is_eof {
            break;
        }
    }

    Parser::new(tokens).parse()
}

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

    /// Execute a query on a DataFrame.
    ///
    /// The expression must evaluate to a boolean per row. Rows where it is true
    /// are kept, with every column's element type preserved.
    pub fn query(&self, dataframe: &DataFrame, query_str: &str) -> Result<DataFrame> {
        let expr = parse_expression(query_str)?;
        let mask = Evaluator::new(dataframe, &self.context).evaluate_query_with_jit(&expr)?;
        self.filter_dataframe_by_mask(dataframe, &mask)
    }

    /// Filter DataFrame using boolean mask
    fn filter_dataframe_by_mask(&self, dataframe: &DataFrame, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != dataframe.row_count() {
            return Err(Error::InconsistentRowCount {
                expected: dataframe.row_count(),
                found: mask.len(),
            });
        }

        let selected_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &include)| if include { Some(idx) } else { None })
            .collect();

        take_rows(dataframe, &selected_indices)
    }

    /// Evaluate an expression for every row and append it as a new column.
    ///
    /// The new column keeps the expression's own type: a numeric expression
    /// produces a `Series<f64>`, a boolean expression a `Series<bool>` and a
    /// text expression a `Series<String>`.
    pub fn eval(
        &self,
        dataframe: &DataFrame,
        expr_str: &str,
        result_column: &str,
    ) -> Result<DataFrame> {
        if dataframe.contains_column(result_column) {
            return Err(Error::DuplicateColumnName(result_column.to_string()));
        }

        let expr = parse_expression(expr_str)?;
        let evaluated = VectorizedEvaluator::new(dataframe, &self.context).evaluate(&expr)?;

        let mut result = dataframe.clone();
        let row_count = dataframe.row_count();
        let name = result_column.to_string();

        match &*evaluated {
            ValueVec::Num(values) => {
                result.add_column(name.clone(), Series::new(values.clone(), Some(name))?)?;
            }
            ValueVec::Bool(values) => {
                result.add_column(name.clone(), Series::new(values.clone(), Some(name))?)?;
            }
            ValueVec::Str(values) => {
                result.add_column(name.clone(), Series::new(values.clone(), Some(name))?)?;
            }
            // A constant expression is broadcast over every row.
            ValueVec::Scalar(LiteralValue::Number(value)) => {
                result.add_column(
                    name.clone(),
                    Series::new(vec![*value; row_count], Some(name))?,
                )?;
            }
            ValueVec::Scalar(LiteralValue::Boolean(value)) => {
                result.add_column(
                    name.clone(),
                    Series::new(vec![*value; row_count], Some(name))?,
                )?;
            }
            ValueVec::Scalar(LiteralValue::String(value)) => {
                result.add_column(
                    name.clone(),
                    Series::new(vec![value.clone(); row_count], Some(name))?,
                )?;
            }
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

    /// The context this engine evaluates with
    pub fn context(&self) -> &QueryContext {
        &self.context
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

    /// Evaluate `expr_str` for every row and append the result as
    /// `result_column`.
    ///
    /// This is a DataFrame expression evaluator (columns, literals, arithmetic
    /// and the registered numeric functions); it executes no external code.
    /// The new column keeps the expression's own type: a numeric expression
    /// produces a `Series<f64>`, a boolean expression a `Series<bool>` and a
    /// text expression a `Series<String>`.
    fn eval(&self, expr_str: &str, result_column: &str) -> Result<DataFrame> {
        QueryEngine::new().eval(self, expr_str, result_column)
    }
}
