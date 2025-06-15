//! Query and evaluation engine for DataFrames
//!
//! This module provides pandas-like query functionality and expression evaluation:
//! - String-based query expressions (.query() method)
//! - Expression evaluation (.eval() method) 
//! - Boolean indexing with complex conditions
//! - Support for mathematical operations and comparisons
//! - Variable substitution and context evaluation

use std::collections::HashMap;
use std::str::Chars;
use std::iter::Peekable;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;
use crate::optimized::jit::jit_core::{JitFunction, JitResult, JitError};

/// Token types for expression parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Column identifier
    Identifier(String),
    /// Numeric literal
    Number(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Boolean(bool),
    /// Comparison operators
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    /// Logical operators
    And,
    Or,
    Not,
    /// Arithmetic operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Power,
    /// Parentheses
    LeftParen,
    RightParen,
    /// Functions
    Function(String),
    /// Comma separator
    Comma,
    /// End of input
    Eof,
}

/// Expression AST node types
#[derive(Debug, Clone)]
pub enum Expr {
    /// Column reference
    Column(String),
    /// Literal values
    Literal(LiteralValue),
    /// Binary operations
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// Unary operations
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Function calls
    Function {
        name: String,
        args: Vec<Expr>,
    },
}

/// Literal value types
#[derive(Debug, Clone)]
pub enum LiteralValue {
    Number(f64),
    String(String),
    Boolean(bool),
}

/// Binary operators
#[derive(Debug, Clone)]
pub enum BinaryOp {
    // Comparison
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    // Logical
    And,
    Or,
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
}

/// Unary operators
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Not,
    Negate,
}

/// Statistics for JIT compilation
#[derive(Debug, Clone, Default)]
pub struct JitQueryStats {
    /// Number of expression compilations
    pub compilations: u64,
    /// Number of JIT executions
    pub jit_executions: u64,
    /// Number of native executions
    pub native_executions: u64,
    /// Total compilation time in nanoseconds
    pub compilation_time_ns: u64,
    /// Total JIT execution time in nanoseconds
    pub jit_execution_time_ns: u64,
    /// Total native execution time in nanoseconds
    pub native_execution_time_ns: u64,
}

impl JitQueryStats {
    pub fn record_compilation(&mut self, duration_ns: u64) {
        self.compilations += 1;
        self.compilation_time_ns += duration_ns;
    }
    
    pub fn record_jit_execution(&mut self, duration_ns: u64) {
        self.jit_executions += 1;
        self.jit_execution_time_ns += duration_ns;
    }
    
    pub fn record_native_execution(&mut self, duration_ns: u64) {
        self.native_executions += 1;
        self.native_execution_time_ns += duration_ns;
    }
    
    pub fn average_compilation_time_ns(&self) -> f64 {
        if self.compilations > 0 {
            self.compilation_time_ns as f64 / self.compilations as f64
        } else {
            0.0
        }
    }
    
    pub fn jit_speedup_ratio(&self) -> f64 {
        if self.jit_executions > 0 && self.native_executions > 0 {
            let avg_native = self.native_execution_time_ns as f64 / self.native_executions as f64;
            let avg_jit = self.jit_execution_time_ns as f64 / self.jit_executions as f64;
            if avg_jit > 0.0 {
                avg_native / avg_jit
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
}

/// Compiled expression cache entry
#[derive(Clone)]
struct CompiledExpression {
    /// Expression signature for cache lookup
    signature: String,
    /// JIT-compiled function
    jit_function: Option<Arc<JitFunction>>,
    /// Number of times this expression has been executed
    execution_count: u64,
    /// Last execution time
    last_execution: std::time::SystemTime,
}

/// Query execution context with JIT compilation support
pub struct QueryContext {
    /// Variable bindings for substitution
    pub variables: HashMap<String, LiteralValue>,
    /// Available functions
    pub functions: HashMap<String, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    /// JIT compilation cache
    compiled_expressions: Arc<Mutex<HashMap<String, CompiledExpression>>>,
    /// JIT compilation statistics
    jit_stats: Arc<Mutex<JitQueryStats>>,
    /// JIT compilation threshold (compile after N executions)
    jit_threshold: u64,
    /// Enable JIT compilation
    jit_enabled: bool,
}

impl std::fmt::Debug for QueryContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryContext")
            .field("variables", &self.variables)
            .field("functions", &format!("{} functions", self.functions.len()))
            .finish()
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        let mut context = Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            compiled_expressions: Arc::new(Mutex::new(HashMap::new())),
            jit_stats: Arc::new(Mutex::new(JitQueryStats::default())),
            jit_threshold: 5, // Compile after 5 executions
            jit_enabled: true,
        };
        
        // Add built-in mathematical functions
        context.add_builtin_functions();
        context
    }
}

impl QueryContext {
    /// Create a new query context
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a new query context with JIT settings
    pub fn with_jit_settings(jit_enabled: bool, jit_threshold: u64) -> Self {
        let mut context = Self::default();
        context.jit_enabled = jit_enabled;
        context.jit_threshold = jit_threshold;
        context
    }
    
    /// Add a variable binding
    pub fn set_variable(&mut self, name: String, value: LiteralValue) {
        self.variables.insert(name, value);
    }
    
    /// Add a custom function
    pub fn add_function<F>(&mut self, name: String, func: F)
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.functions.insert(name, Box::new(func));
    }
    
    /// Get JIT compilation statistics
    pub fn jit_stats(&self) -> JitQueryStats {
        self.jit_stats.lock().unwrap().clone()
    }
    
    /// Enable or disable JIT compilation
    pub fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = enabled;
    }
    
    /// Set JIT compilation threshold
    pub fn set_jit_threshold(&mut self, threshold: u64) {
        self.jit_threshold = threshold;
    }
    
    /// Clear JIT compilation cache
    pub fn clear_jit_cache(&mut self) {
        let mut cache = self.compiled_expressions.lock().unwrap();
        cache.clear();
    }
    
    /// Get the number of compiled expressions in cache
    pub fn compiled_expressions_count(&self) -> usize {
        self.compiled_expressions.lock().unwrap().len()
    }
    
    /// Add built-in mathematical functions
    fn add_builtin_functions(&mut self) {
        // Basic math functions
        self.add_function("abs".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].abs() }
        });
        
        self.add_function("sqrt".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].sqrt() }
        });
        
        self.add_function("log".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].ln() }
        });
        
        self.add_function("log10".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].log10() }
        });
        
        self.add_function("exp".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].exp() }
        });
        
        // Trigonometric functions
        self.add_function("sin".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].sin() }
        });
        
        self.add_function("cos".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].cos() }
        });
        
        self.add_function("tan".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args[0].tan() }
        });
        
        // Statistical functions
        self.add_function("min".to_string(), |args| {
            args.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        });
        
        self.add_function("max".to_string(), |args| {
            args.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        });
        
        self.add_function("sum".to_string(), |args| {
            args.iter().sum()
        });
        
        self.add_function("mean".to_string(), |args| {
            if args.is_empty() { 0.0 } else { args.iter().sum::<f64>() / args.len() as f64 }
        });
    }
}

/// Lexer for tokenizing query expressions
pub struct Lexer {
    chars: Peekable<Chars<'static>>,
    input: &'static str,
}

impl Lexer {
    /// Create a new lexer
    pub fn new(input: &'static str) -> Self {
        Self {
            chars: input.chars().peekable(),
            input,
        }
    }
    
    /// Get the next token
    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();
        
        match self.chars.peek() {
            None => Ok(Token::Eof),
            Some(&ch) => {
                match ch {
                    '(' => {
                        self.chars.next();
                        Ok(Token::LeftParen)
                    }
                    ')' => {
                        self.chars.next();
                        Ok(Token::RightParen)
                    }
                    ',' => {
                        self.chars.next();
                        Ok(Token::Comma)
                    }
                    '+' => {
                        self.chars.next();
                        Ok(Token::Plus)
                    }
                    '-' => {
                        self.chars.next();
                        Ok(Token::Minus)
                    }
                    '*' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'*') {
                            self.chars.next();
                            Ok(Token::Power)
                        } else {
                            Ok(Token::Multiply)
                        }
                    }
                    '/' => {
                        self.chars.next();
                        Ok(Token::Divide)
                    }
                    '%' => {
                        self.chars.next();
                        Ok(Token::Modulo)
                    }
                    '=' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'=') {
                            self.chars.next();
                            Ok(Token::Equal)
                        } else {
                            Err(Error::InvalidValue("Expected '==' for equality comparison".to_string()))
                        }
                    }
                    '!' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'=') {
                            self.chars.next();
                            Ok(Token::NotEqual)
                        } else {
                            Ok(Token::Not)
                        }
                    }
                    '<' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'=') {
                            self.chars.next();
                            Ok(Token::LessThanOrEqual)
                        } else {
                            Ok(Token::LessThan)
                        }
                    }
                    '>' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'=') {
                            self.chars.next();
                            Ok(Token::GreaterThanOrEqual)
                        } else {
                            Ok(Token::GreaterThan)
                        }
                    }
                    '&' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'&') {
                            self.chars.next();
                            Ok(Token::And)
                        } else {
                            Err(Error::InvalidValue("Expected '&&' for logical AND".to_string()))
                        }
                    }
                    '|' => {
                        self.chars.next();
                        if self.chars.peek() == Some(&'|') {
                            self.chars.next();
                            Ok(Token::Or)
                        } else {
                            Err(Error::InvalidValue("Expected '||' for logical OR".to_string()))
                        }
                    }
                    '\'' | '"' => self.read_string(),
                    '0'..='9' => self.read_number(),
                    'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
                    _ => Err(Error::InvalidValue(format!("Unexpected character: {}", ch))),
                }
            }
        }
    }
    
    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }
    
    /// Read a string literal
    fn read_string(&mut self) -> Result<Token> {
        let quote = self.chars.next().unwrap(); // consume opening quote
        let mut value = String::new();
        
        while let Some(ch) = self.chars.next() {
            if ch == quote {
                return Ok(Token::String(value));
            } else if ch == '\\' {
                // Handle escape sequences
                if let Some(escaped) = self.chars.next() {
                    match escaped {
                        'n' => value.push('\n'),
                        't' => value.push('\t'),
                        'r' => value.push('\r'),
                        '\\' => value.push('\\'),
                        '\'' => value.push('\''),
                        '"' => value.push('"'),
                        _ => {
                            value.push('\\');
                            value.push(escaped);
                        }
                    }
                }
            } else {
                value.push(ch);
            }
        }
        
        Err(Error::InvalidValue("Unterminated string literal".to_string()))
    }
    
    /// Read a number literal
    fn read_number(&mut self) -> Result<Token> {
        let mut number = String::new();
        
        while let Some(&ch) = self.chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                number.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }
        
        match number.parse::<f64>() {
            Ok(value) => Ok(Token::Number(value)),
            Err(_) => Err(Error::InvalidValue(format!("Invalid number: {}", number))),
        }
    }
    
    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> Result<Token> {
        let mut identifier = String::new();
        
        while let Some(&ch) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                identifier.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }
        
        // Check for keywords
        match identifier.as_str() {
            "true" => Ok(Token::Boolean(true)),
            "false" => Ok(Token::Boolean(false)),
            "and" => Ok(Token::And),
            "or" => Ok(Token::Or),
            "not" => Ok(Token::Not),
            _ => {
                // Check if it's followed by '(' to determine if it's a function
                if self.chars.peek() == Some(&'(') {
                    Ok(Token::Function(identifier))
                } else {
                    Ok(Token::Identifier(identifier))
                }
            }
        }
    }
}

/// Parser for building expression AST
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    /// Create a new parser with tokens
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }
    
    /// Parse the tokens into an expression AST
    pub fn parse(&mut self) -> Result<Expr> {
        self.parse_or_expression()
    }
    
    /// Parse OR expressions
    fn parse_or_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_and_expression()?;
        
        while self.match_token(&Token::Or) {
            let op = BinaryOp::Or;
            let right = self.parse_and_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse AND expressions
    fn parse_and_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_equality_expression()?;
        
        while self.match_token(&Token::And) {
            let op = BinaryOp::And;
            let right = self.parse_equality_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse equality expressions (==, !=)
    fn parse_equality_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_comparison_expression()?;
        
        while let Some(op) = self.match_equality_operator() {
            let right = self.parse_comparison_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse comparison expressions (<, <=, >, >=)
    fn parse_comparison_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_additive_expression()?;
        
        while let Some(op) = self.match_comparison_operator() {
            let right = self.parse_additive_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse additive expressions (+, -)
    fn parse_additive_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative_expression()?;
        
        while let Some(op) = self.match_additive_operator() {
            let right = self.parse_multiplicative_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse multiplicative expressions (*, /, %)
    fn parse_multiplicative_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_power_expression()?;
        
        while let Some(op) = self.match_multiplicative_operator() {
            let right = self.parse_power_expression()?;
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse power expressions (**)
    fn parse_power_expression(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary_expression()?;
        
        if self.match_token(&Token::Power) {
            let right = self.parse_power_expression()?; // Right associative
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::Power,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse unary expressions (!, -, not)
    fn parse_unary_expression(&mut self) -> Result<Expr> {
        if self.match_token(&Token::Not) {
            let operand = self.parse_unary_expression()?;
            Ok(Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            })
        } else if self.match_token(&Token::Minus) {
            let operand = self.parse_unary_expression()?;
            Ok(Expr::Unary {
                op: UnaryOp::Negate,
                operand: Box::new(operand),
            })
        } else {
            self.parse_primary_expression()
        }
    }
    
    /// Parse primary expressions (literals, identifiers, function calls, parentheses)
    fn parse_primary_expression(&mut self) -> Result<Expr> {
        if let Some(token) = self.current_token().cloned() {
            match token {
                Token::Number(value) => {
                    self.advance();
                    Ok(Expr::Literal(LiteralValue::Number(value)))
                }
                Token::String(value) => {
                    self.advance();
                    Ok(Expr::Literal(LiteralValue::String(value)))
                }
                Token::Boolean(value) => {
                    self.advance();
                    Ok(Expr::Literal(LiteralValue::Boolean(value)))
                }
                Token::Identifier(name) => {
                    self.advance();
                    Ok(Expr::Column(name))
                }
                Token::Function(name) => {
                    let func_name = name;
                    self.advance();
                    
                    if !self.match_token(&Token::LeftParen) {
                        return Err(Error::InvalidValue("Expected '(' after function name".to_string()));
                    }
                    
                    let mut args = Vec::new();
                    
                    if !self.check_token(&Token::RightParen) {
                        loop {
                            args.push(self.parse_or_expression()?);
                            
                            if !self.match_token(&Token::Comma) {
                                break;
                            }
                        }
                    }
                    
                    if !self.match_token(&Token::RightParen) {
                        return Err(Error::InvalidValue("Expected ')' after function arguments".to_string()));
                    }
                    
                    Ok(Expr::Function {
                        name: func_name,
                        args,
                    })
                }
                Token::LeftParen => {
                    self.advance();
                    let expr = self.parse_or_expression()?;
                    
                    if !self.match_token(&Token::RightParen) {
                        return Err(Error::InvalidValue("Expected ')' after expression".to_string()));
                    }
                    
                    Ok(expr)
                }
                _ => Err(Error::InvalidValue(format!("Unexpected token: {:?}", token))),
            }
        } else {
            Err(Error::InvalidValue("Unexpected end of input".to_string()))
        }
    }
    
    /// Helper methods for parsing
    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }
    
    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }
    
    fn match_token(&mut self, expected: &Token) -> bool {
        if self.check_token(expected) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    fn check_token(&self, expected: &Token) -> bool {
        if let Some(token) = self.current_token() {
            std::mem::discriminant(token) == std::mem::discriminant(expected)
        } else {
            false
        }
    }
    
    fn match_equality_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::Equal) => {
                self.advance();
                Some(BinaryOp::Equal)
            }
            Some(Token::NotEqual) => {
                self.advance();
                Some(BinaryOp::NotEqual)
            }
            _ => None,
        }
    }
    
    fn match_comparison_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::LessThan) => {
                self.advance();
                Some(BinaryOp::LessThan)
            }
            Some(Token::LessThanOrEqual) => {
                self.advance();
                Some(BinaryOp::LessThanOrEqual)
            }
            Some(Token::GreaterThan) => {
                self.advance();
                Some(BinaryOp::GreaterThan)
            }
            Some(Token::GreaterThanOrEqual) => {
                self.advance();
                Some(BinaryOp::GreaterThanOrEqual)
            }
            _ => None,
        }
    }
    
    fn match_additive_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::Plus) => {
                self.advance();
                Some(BinaryOp::Add)
            }
            Some(Token::Minus) => {
                self.advance();
                Some(BinaryOp::Subtract)
            }
            _ => None,
        }
    }
    
    fn match_multiplicative_operator(&mut self) -> Option<BinaryOp> {
        match self.current_token() {
            Some(Token::Multiply) => {
                self.advance();
                Some(BinaryOp::Multiply)
            }
            Some(Token::Divide) => {
                self.advance();
                Some(BinaryOp::Divide)
            }
            Some(Token::Modulo) => {
                self.advance();
                Some(BinaryOp::Modulo)
            }
            _ => None,
        }
    }
}

/// Expression evaluator with optimization support
pub struct Evaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    /// Cache for column data to avoid repeated parsing
    column_cache: std::cell::RefCell<HashMap<String, Vec<LiteralValue>>>,
    /// Optimization flags
    enable_short_circuit: bool,
    enable_constant_folding: bool,
}

/// JIT-compiled expression evaluator
pub struct JitEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    /// Cache for column data
    column_cache: std::cell::RefCell<HashMap<String, Vec<LiteralValue>>>,
}

/// Optimized expression evaluator with short-circuiting and constant folding
pub struct OptimizedEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    column_cache: std::cell::RefCell<HashMap<String, Vec<LiteralValue>>>,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self { 
            dataframe, 
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
            enable_short_circuit: true,
            enable_constant_folding: true,
        }
    }
    
    /// Create a new evaluator with optimization settings
    pub fn with_optimizations(dataframe: &'a DataFrame, context: &'a QueryContext, 
                             short_circuit: bool, constant_folding: bool) -> Self {
        Self { 
            dataframe, 
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
            enable_short_circuit: short_circuit,
            enable_constant_folding: constant_folding,
        }
    }
    
    /// Evaluate an expression and return a boolean mask for filtering
    pub fn evaluate_query(&self, expr: &Expr) -> Result<Vec<bool>> {
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);
        
        // Pre-optimize expression if constant folding is enabled
        let optimized_expr = if self.enable_constant_folding {
            self.optimize_expression(expr)?
        } else {
            expr.clone()
        };
        
        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(&optimized_expr, row_idx)?;
            match value {
                LiteralValue::Boolean(b) => result.push(b),
                _ => return Err(Error::InvalidValue("Query expression must evaluate to boolean".to_string())),
            }
        }
        
        Ok(result)
    }
    
    /// Evaluate query with JIT compilation support
    pub fn evaluate_query_with_jit(&self, expr: &Expr) -> Result<Vec<bool>> {
        let expr_signature = self.expression_signature(expr);
        
        // Check if we should use JIT compilation
        if self.context.jit_enabled {
            let should_compile = {
                let mut cache = self.context.compiled_expressions.lock().unwrap();
                if let Some(compiled_expr) = cache.get_mut(&expr_signature) {
                    compiled_expr.execution_count += 1;
                    compiled_expr.last_execution = std::time::SystemTime::now();
                    
                    // Use JIT if available, otherwise check if we should compile
                    compiled_expr.jit_function.is_some() || 
                    (compiled_expr.execution_count >= self.context.jit_threshold && compiled_expr.jit_function.is_none())
                } else {
                    // First execution - add to cache
                    cache.insert(expr_signature.clone(), CompiledExpression {
                        signature: expr_signature.clone(),
                        jit_function: None,
                        execution_count: 1,
                        last_execution: std::time::SystemTime::now(),
                    });
                    false
                }
            };
            
            if should_compile {
                // Try to compile the expression
                if let Ok(jit_func) = self.compile_expression_to_jit(expr) {
                    let mut cache = self.context.compiled_expressions.lock().unwrap();
                    if let Some(compiled_expr) = cache.get_mut(&expr_signature) {
                        compiled_expr.jit_function = Some(Arc::new(jit_func));
                    }
                }
            }
            
            // Try to execute with JIT
            {
                let cache = self.context.compiled_expressions.lock().unwrap();
                if let Some(compiled_expr) = cache.get(&expr_signature) {
                    if let Some(jit_func) = &compiled_expr.jit_function {
                        return self.execute_jit_compiled_query(expr, jit_func);
                    }
                }
            }
        }
        
        // Fall back to regular evaluation
        self.evaluate_query(expr)
    }
    
    /// Generate a signature for an expression for caching
    fn expression_signature(&self, expr: &Expr) -> String {
        format!("{:?}", expr) // Simple signature based on Debug output
    }
    
    /// Compile an expression to JIT-compiled function
    fn compile_expression_to_jit(&self, expr: &Expr) -> JitResult<JitFunction> {
        let start = Instant::now();
        
        let signature = self.expression_signature(expr);
        
        // For this implementation, we'll create a JIT function that encapsulates
        // the expression evaluation logic
        let jit_func = match expr {
            // Simple numeric operations can be JIT compiled
            Expr::Binary { left, op, right } if self.is_jit_compilable_binary(left, op, right) => {
                self.compile_binary_expression(left, op, right)?
            }
            
            // Column comparisons can be vectorized
            Expr::Binary { left, op, right } if self.is_column_comparison(left, right) => {
                self.compile_column_comparison(left, op, right)?
            }
            
            // Other expressions fall back to interpreted evaluation
            _ => {
                return Err(JitError::CompilationError("Expression not JIT-compilable".to_string()));
            }
        };
        
        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_compilation(duration.as_nanos() as u64);
        }
        
        Ok(jit_func)
    }
    
    /// Check if a binary expression can be JIT compiled
    fn is_jit_compilable_binary(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> bool {
        // Simple arithmetic operations on numeric literals or columns
        matches!(op, BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide) &&
        self.is_numeric_expression(left) && self.is_numeric_expression(right)
    }
    
    /// Check if expression is numeric (literal or column)
    fn is_numeric_expression(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Literal(LiteralValue::Number(_)) | Expr::Column(_))
    }
    
    /// Check if this is a column comparison suitable for vectorization
    fn is_column_comparison(&self, left: &Expr, right: &Expr) -> bool {
        matches!((left, right), (Expr::Column(_), Expr::Literal(_)) | (Expr::Literal(_), Expr::Column(_)))
    }
    
    /// Compile a binary arithmetic expression
    fn compile_binary_expression(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> JitResult<JitFunction> {
        let op_name = match op {
            BinaryOp::Add => "add",
            BinaryOp::Subtract => "sub", 
            BinaryOp::Multiply => "mul",
            BinaryOp::Divide => "div",
            _ => return Err(JitError::CompilationError("Unsupported binary operation for JIT".to_string())),
        };
        
        let func_name = format!("jit_binary_{}_{:?}_{:?}", op_name, left, right);
        
        // Create a JIT function that performs the binary operation
        let jit_func = match op {
            BinaryOp::Add => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        args[0] + args[1]
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::Subtract => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        args[0] - args[1]
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::Multiply => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        args[0] * args[1]
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::Divide => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 && args[1] != 0.0 {
                        args[0] / args[1]
                    } else {
                        f64::NAN
                    }
                })
            }
            _ => unreachable!(),
        };
        
        Ok(jit_func)
    }
    
    /// Compile a column comparison expression
    fn compile_column_comparison(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> JitResult<JitFunction> {
        let func_name = format!("jit_comparison_{:?}_{:?}_{:?}", left, op, right);
        
        // Create a vectorized comparison function
        let jit_func = match op {
            BinaryOp::Equal => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        if (args[0] - args[1]).abs() < f64::EPSILON { 1.0 } else { 0.0 }
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::LessThan => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        if args[0] < args[1] { 1.0 } else { 0.0 }
                    } else {
                        0.0
                    }
                })
            }
            BinaryOp::GreaterThan => {
                crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                    if args.len() >= 2 {
                        if args[0] > args[1] { 1.0 } else { 0.0 }
                    } else {
                        0.0
                    }
                })
            }
            _ => return Err(JitError::CompilationError("Unsupported comparison operation for JIT".to_string())),
        };
        
        Ok(jit_func)
    }
    
    /// Execute a JIT-compiled query
    fn execute_jit_compiled_query(&self, expr: &Expr, jit_func: &JitFunction) -> Result<Vec<bool>> {
        let start = Instant::now();
        
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);
        
        // For column-based operations, we can vectorize
        if let Expr::Binary { left, op: _, right } = expr {
            if self.is_column_comparison(left, right) {
                // Vectorized execution path
                let (column_values, literal_value) = self.extract_column_and_literal(left, right)?;
                
                for &col_val in &column_values {
                    let args = vec![col_val, literal_value];
                    use crate::optimized::jit::jit_core::JitCompilable;
                    let jit_result = jit_func.execute(args);
                    result.push(jit_result != 0.0);
                }
                
                let duration = start.elapsed();
                {
                    let mut stats = self.context.jit_stats.lock().unwrap();
                    stats.record_jit_execution(duration.as_nanos() as u64);
                }
                
                return Ok(result);
            }
        }
        
        // Fall back to row-by-row evaluation
        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(expr, row_idx)?;
            match value {
                LiteralValue::Boolean(b) => result.push(b),
                _ => return Err(Error::InvalidValue("Query expression must evaluate to boolean".to_string())),
            }
        }
        
        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_jit_execution(duration.as_nanos() as u64);
        }
        
        Ok(result)
    }
    
    /// Extract column values and literal value from a column comparison
    fn extract_column_and_literal(&self, left: &Expr, right: &Expr) -> Result<(Vec<f64>, f64)> {
        match (left, right) {
            (Expr::Column(col_name), Expr::Literal(LiteralValue::Number(lit_val))) => {
                let col_values = self.dataframe.get_column_numeric_values(col_name)?;
                Ok((col_values, *lit_val))
            }
            (Expr::Literal(LiteralValue::Number(lit_val)), Expr::Column(col_name)) => {
                let col_values = self.dataframe.get_column_numeric_values(col_name)?;
                Ok((col_values, *lit_val))
            }
            _ => Err(Error::InvalidValue("Invalid column comparison expression".to_string())),
        }
    }
    
    /// Optimize expression through constant folding and algebraic simplifications
    fn optimize_expression(&self, expr: &Expr) -> Result<Expr> {
        match expr {
            Expr::Binary { left, op, right } => {
                let optimized_left = self.optimize_expression(left)?;
                let optimized_right = self.optimize_expression(right)?;
                
                // Constant folding: if both operands are literals, evaluate at compile time
                if let (Expr::Literal(l), Expr::Literal(r)) = (&optimized_left, &optimized_right) {
                    let result = self.apply_binary_operation(l, op, r)?;
                    return Ok(Expr::Literal(result));
                }
                
                // Algebraic simplifications
                match (&optimized_left, op, &optimized_right) {
                    // AND optimizations: x && true = x, x && false = false
                    (expr, BinaryOp::And, Expr::Literal(LiteralValue::Boolean(true))) => Ok(expr.clone()),
                    (Expr::Literal(LiteralValue::Boolean(true)), BinaryOp::And, expr) => Ok(expr.clone()),
                    (_, BinaryOp::And, Expr::Literal(LiteralValue::Boolean(false))) => Ok(Expr::Literal(LiteralValue::Boolean(false))),
                    (Expr::Literal(LiteralValue::Boolean(false)), BinaryOp::And, _) => Ok(Expr::Literal(LiteralValue::Boolean(false))),
                    
                    // OR optimizations: x || true = true, x || false = x
                    (_, BinaryOp::Or, Expr::Literal(LiteralValue::Boolean(true))) => Ok(Expr::Literal(LiteralValue::Boolean(true))),
                    (Expr::Literal(LiteralValue::Boolean(true)), BinaryOp::Or, _) => Ok(Expr::Literal(LiteralValue::Boolean(true))),
                    (expr, BinaryOp::Or, Expr::Literal(LiteralValue::Boolean(false))) => Ok(expr.clone()),
                    (Expr::Literal(LiteralValue::Boolean(false)), BinaryOp::Or, expr) => Ok(expr.clone()),
                    
                    // Arithmetic optimizations: x + 0 = x, x * 1 = x, x * 0 = 0
                    (expr, BinaryOp::Add, Expr::Literal(LiteralValue::Number(n))) if *n == 0.0 => Ok(expr.clone()),
                    (Expr::Literal(LiteralValue::Number(n)), BinaryOp::Add, expr) if *n == 0.0 => Ok(expr.clone()),
                    (expr, BinaryOp::Multiply, Expr::Literal(LiteralValue::Number(n))) if *n == 1.0 => Ok(expr.clone()),
                    (Expr::Literal(LiteralValue::Number(n)), BinaryOp::Multiply, expr) if *n == 1.0 => Ok(expr.clone()),
                    (_, BinaryOp::Multiply, Expr::Literal(LiteralValue::Number(n))) if *n == 0.0 => Ok(Expr::Literal(LiteralValue::Number(0.0))),
                    (Expr::Literal(LiteralValue::Number(n)), BinaryOp::Multiply, _) if *n == 0.0 => Ok(Expr::Literal(LiteralValue::Number(0.0))),
                    
                    _ => Ok(Expr::Binary {
                        left: Box::new(optimized_left),
                        op: op.clone(),
                        right: Box::new(optimized_right),
                    })
                }
            }
            
            Expr::Unary { op, operand } => {
                let optimized_operand = self.optimize_expression(operand)?;
                
                // Constant folding for unary operations
                if let Expr::Literal(val) = &optimized_operand {
                    let result = self.apply_unary_operation(op, val)?;
                    return Ok(Expr::Literal(result));
                }
                
                // Double negation elimination: !!x = x
                if let (UnaryOp::Not, Expr::Unary { op: UnaryOp::Not, operand }) = (op, &optimized_operand) {
                    return Ok((**operand).clone());
                }
                
                Ok(Expr::Unary {
                    op: op.clone(),
                    operand: Box::new(optimized_operand),
                })
            }
            
            Expr::Function { name, args } => {
                let optimized_args: Result<Vec<Expr>> = args.iter()
                    .map(|arg| self.optimize_expression(arg))
                    .collect();
                
                Ok(Expr::Function {
                    name: name.clone(),
                    args: optimized_args?,
                })
            }
            
            _ => Ok(expr.clone())
        }
    }
    
    /// Evaluate an expression for a specific row
    fn evaluate_expression_for_row(&self, expr: &Expr, row_idx: usize) -> Result<LiteralValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),
            
            Expr::Column(name) => {
                if !self.dataframe.contains_column(name) {
                    return Err(Error::ColumnNotFound(name.clone()));
                }
                
                // Use cached column data if available
                {
                    let cache = self.column_cache.borrow();
                    if let Some(cached_values) = cache.get(name) {
                        if row_idx < cached_values.len() {
                            return Ok(cached_values[row_idx].clone());
                        } else {
                            return Err(Error::IndexOutOfBounds { index: row_idx, size: cached_values.len() });
                        }
                    }
                }
                
                // Cache miss - load and parse column data
                let column_values = self.dataframe.get_column_string_values(name)?;
                let parsed_values: Vec<LiteralValue> = column_values.iter()
                    .map(|str_value| {
                        // Try to parse as number first, then boolean, then keep as string
                        if let Ok(num) = str_value.parse::<f64>() {
                            LiteralValue::Number(num)
                        } else if let Ok(bool_val) = str_value.parse::<bool>() {
                            LiteralValue::Boolean(bool_val)
                        } else {
                            LiteralValue::String(str_value.clone())
                        }
                    })
                    .collect();
                
                // Cache the parsed values
                {
                    let mut cache = self.column_cache.borrow_mut();
                    cache.insert(name.clone(), parsed_values.clone());
                }
                
                if row_idx < parsed_values.len() {
                    Ok(parsed_values[row_idx].clone())
                } else {
                    Err(Error::IndexOutOfBounds { index: row_idx, size: parsed_values.len() })
                }
            }
            
            Expr::Binary { left, op, right } => {
                // Implement short-circuiting for logical operations
                if self.enable_short_circuit {
                    match op {
                        BinaryOp::And => {
                            let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                            if let LiteralValue::Boolean(false) = left_val {
                                return Ok(LiteralValue::Boolean(false)); // Short-circuit: false && x = false
                            }
                            let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                            self.apply_binary_operation(&left_val, op, &right_val)
                        }
                        BinaryOp::Or => {
                            let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                            if let LiteralValue::Boolean(true) = left_val {
                                return Ok(LiteralValue::Boolean(true)); // Short-circuit: true || x = true
                            }
                            let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                            self.apply_binary_operation(&left_val, op, &right_val)
                        }
                        _ => {
                            let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                            let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                            self.apply_binary_operation(&left_val, op, &right_val)
                        }
                    }
                } else {
                    let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                    let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                    self.apply_binary_operation(&left_val, op, &right_val)
                }
            }
            
            Expr::Unary { op, operand } => {
                let operand_val = self.evaluate_expression_for_row(operand, row_idx)?;
                self.apply_unary_operation(op, &operand_val)
            }
            
            Expr::Function { name, args } => {
                let arg_values: Result<Vec<f64>> = args.iter()
                    .map(|arg| {
                        let val = self.evaluate_expression_for_row(arg, row_idx)?;
                        match val {
                            LiteralValue::Number(n) => Ok(n),
                            _ => Err(Error::InvalidValue("Function arguments must be numeric".to_string())),
                        }
                    })
                    .collect();
                
                let arg_values = arg_values?;
                
                if let Some(func) = self.context.functions.get(name) {
                    let result = func(&arg_values);
                    Ok(LiteralValue::Number(result))
                } else {
                    Err(Error::InvalidValue(format!("Unknown function: {}", name)))
                }
            }
        }
    }
    
    /// Apply binary operation
    fn apply_binary_operation(&self, left: &LiteralValue, op: &BinaryOp, right: &LiteralValue) -> Result<LiteralValue> {
        match (left, right, op) {
            // Numeric operations
            (LiteralValue::Number(l), LiteralValue::Number(r), op) => {
                match op {
                    BinaryOp::Add => Ok(LiteralValue::Number(l + r)),
                    BinaryOp::Subtract => Ok(LiteralValue::Number(l - r)),
                    BinaryOp::Multiply => Ok(LiteralValue::Number(l * r)),
                    BinaryOp::Divide => {
                        if *r == 0.0 {
                            Err(Error::InvalidValue("Division by zero".to_string()))
                        } else {
                            Ok(LiteralValue::Number(l / r))
                        }
                    }
                    BinaryOp::Modulo => Ok(LiteralValue::Number(l % r)),
                    BinaryOp::Power => Ok(LiteralValue::Number(l.powf(*r))),
                    BinaryOp::Equal => Ok(LiteralValue::Boolean((l - r).abs() < f64::EPSILON)),
                    BinaryOp::NotEqual => Ok(LiteralValue::Boolean((l - r).abs() >= f64::EPSILON)),
                    BinaryOp::LessThan => Ok(LiteralValue::Boolean(l < r)),
                    BinaryOp::LessThanOrEqual => Ok(LiteralValue::Boolean(l <= r)),
                    BinaryOp::GreaterThan => Ok(LiteralValue::Boolean(l > r)),
                    BinaryOp::GreaterThanOrEqual => Ok(LiteralValue::Boolean(l >= r)),
                    BinaryOp::And | BinaryOp::Or => Err(Error::InvalidValue("Logical operations require boolean operands".to_string())),
                }
            }
            
            // String operations
            (LiteralValue::String(l), LiteralValue::String(r), op) => {
                match op {
                    BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                    BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                    BinaryOp::Add => Ok(LiteralValue::String(format!("{}{}", l, r))),
                    _ => Err(Error::InvalidValue("Unsupported operation for strings".to_string())),
                }
            }
            
            // Boolean operations
            (LiteralValue::Boolean(l), LiteralValue::Boolean(r), op) => {
                match op {
                    BinaryOp::And => Ok(LiteralValue::Boolean(*l && *r)),
                    BinaryOp::Or => Ok(LiteralValue::Boolean(*l || *r)),
                    BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                    BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                    _ => Err(Error::InvalidValue("Unsupported operation for booleans".to_string())),
                }
            }
            
            // Mixed type comparisons (try to convert to common type)
            (LiteralValue::Number(l), LiteralValue::String(r), op) => {
                if let Ok(r_num) = r.parse::<f64>() {
                    self.apply_binary_operation(&LiteralValue::Number(*l), op, &LiteralValue::Number(r_num))
                } else {
                    Err(Error::InvalidValue("Cannot compare number with non-numeric string".to_string()))
                }
            }
            
            (LiteralValue::String(l), LiteralValue::Number(r), op) => {
                if let Ok(l_num) = l.parse::<f64>() {
                    self.apply_binary_operation(&LiteralValue::Number(l_num), op, &LiteralValue::Number(*r))
                } else {
                    Err(Error::InvalidValue("Cannot compare non-numeric string with number".to_string()))
                }
            }
            
            _ => Err(Error::InvalidValue("Unsupported operand types for operation".to_string())),
        }
    }
    
    /// Apply unary operation
    fn apply_unary_operation(&self, op: &UnaryOp, operand: &LiteralValue) -> Result<LiteralValue> {
        match (op, operand) {
            (UnaryOp::Not, LiteralValue::Boolean(b)) => Ok(LiteralValue::Boolean(!b)),
            (UnaryOp::Negate, LiteralValue::Number(n)) => Ok(LiteralValue::Number(-n)),
            _ => Err(Error::InvalidValue("Unsupported unary operation".to_string())),
        }
    }
}

impl<'a> JitEvaluator<'a> {
    /// Create a new JIT evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self {
            dataframe,
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }
    
    /// Evaluate query with aggressive JIT compilation
    pub fn evaluate_query_jit(&self, expr: &Expr) -> Result<Vec<bool>> {
        // Force JIT compilation for all suitable expressions
        let expr_signature = format!("{:?}", expr);
        
        if self.context.jit_enabled {
            // Try to compile immediately
            if let Ok(jit_func) = self.compile_expression_to_jit(expr) {
                return self.execute_jit_compiled_query(expr, &jit_func);
            }
        }
        
        // Fall back to regular evaluation
        self.evaluate_query_fallback(expr)
    }
    
    /// Compile expression to JIT (same logic as Evaluator but more aggressive)
    fn compile_expression_to_jit(&self, expr: &Expr) -> JitResult<JitFunction> {
        let start = Instant::now();
        
        let jit_func = match expr {
            // All numeric binary operations
            Expr::Binary { left, op, right } if self.is_jit_compilable_binary(left, op, right) => {
                self.compile_binary_expression(left, op, right)?
            }
            
            // All comparison operations
            Expr::Binary { left, op, right } if self.is_comparison_op(op) => {
                self.compile_comparison_expression(left, op, right)?
            }
            
            // Function calls
            Expr::Function { name, args } => {
                self.compile_function_expression(name, args)?
            }
            
            _ => {
                return Err(JitError::CompilationError("Expression not JIT-compilable".to_string()));
            }
        };
        
        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_compilation(duration.as_nanos() as u64);
        }
        
        Ok(jit_func)
    }
    
    fn is_jit_compilable_binary(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> bool {
        matches!(op, BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Power)
    }
    
    fn is_comparison_op(&self, op: &BinaryOp) -> bool {
        matches!(op, BinaryOp::Equal | BinaryOp::NotEqual | BinaryOp::LessThan | 
                     BinaryOp::LessThanOrEqual | BinaryOp::GreaterThan | BinaryOp::GreaterThanOrEqual)
    }
    
    fn compile_binary_expression(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> JitResult<JitFunction> {
        let func_name = format!("jit_binary_{:?}", op);
        
        let jit_func = match op {
            BinaryOp::Add => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                args.iter().sum()
            }),
            BinaryOp::Subtract => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 { args[0] - args[1] } else { 0.0 }
            }),
            BinaryOp::Multiply => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                args.iter().product()
            }),
            BinaryOp::Divide => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && args[1] != 0.0 { args[0] / args[1] } else { f64::NAN }
            }),
            BinaryOp::Power => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 { args[0].powf(args[1]) } else { 0.0 }
            }),
            _ => return Err(JitError::CompilationError("Unsupported binary operation".to_string())),
        };
        
        Ok(jit_func)
    }
    
    fn compile_comparison_expression(&self, left: &Expr, op: &BinaryOp, right: &Expr) -> JitResult<JitFunction> {
        let func_name = format!("jit_comparison_{:?}", op);
        
        let jit_func = match op {
            BinaryOp::Equal => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && (args[0] - args[1]).abs() < f64::EPSILON { 1.0 } else { 0.0 }
            }),
            BinaryOp::NotEqual => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && (args[0] - args[1]).abs() >= f64::EPSILON { 1.0 } else { 0.0 }
            }),
            BinaryOp::LessThan => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && args[0] < args[1] { 1.0 } else { 0.0 }
            }),
            BinaryOp::LessThanOrEqual => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && args[0] <= args[1] { 1.0 } else { 0.0 }
            }),
            BinaryOp::GreaterThan => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && args[0] > args[1] { 1.0 } else { 0.0 }
            }),
            BinaryOp::GreaterThanOrEqual => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if args.len() >= 2 && args[0] >= args[1] { 1.0 } else { 0.0 }
            }),
            _ => return Err(JitError::CompilationError("Unsupported comparison operation".to_string())),
        };
        
        Ok(jit_func)
    }
    
    fn compile_function_expression(&self, name: &str, args: &[Expr]) -> JitResult<JitFunction> {
        let func_name = format!("jit_function_{}", name);
        
        // Compile built-in mathematical functions
        let jit_func = match name {
            "abs" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() { args[0].abs() } else { 0.0 }
            }),
            "sqrt" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() { args[0].sqrt() } else { 0.0 }
            }),
            "sin" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() { args[0].sin() } else { 0.0 }
            }),
            "cos" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() { args[0].cos() } else { 0.0 }
            }),
            "sum" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                args.iter().sum()
            }),
            "mean" => crate::optimized::jit::jit_core::jit(func_name, |args: Vec<f64>| {
                if !args.is_empty() { args.iter().sum::<f64>() / args.len() as f64 } else { 0.0 }
            }),
            _ => return Err(JitError::CompilationError(format!("Function {} not JIT-compilable", name))),
        };
        
        Ok(jit_func)
    }
    
    fn execute_jit_compiled_query(&self, expr: &Expr, jit_func: &JitFunction) -> Result<Vec<bool>> {
        let start = Instant::now();
        
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);
        
        // Execute JIT function for each row
        for row_idx in 0..row_count {
            let args = self.extract_arguments_for_row(expr, row_idx)?;
            use crate::optimized::jit::jit_core::JitCompilable;
            let jit_result = jit_func.execute(args);
            
            // Convert numeric result to boolean (0.0 = false, non-zero = true)
            result.push(jit_result != 0.0);
        }
        
        let duration = start.elapsed();
        {
            let mut stats = self.context.jit_stats.lock().unwrap();
            stats.record_jit_execution(duration.as_nanos() as u64);
        }
        
        Ok(result)
    }
    
    fn extract_arguments_for_row(&self, expr: &Expr, row_idx: usize) -> Result<Vec<f64>> {
        match expr {
            Expr::Binary { left, op: _, right } => {
                let left_val = self.extract_numeric_value(left, row_idx)?;
                let right_val = self.extract_numeric_value(right, row_idx)?;
                Ok(vec![left_val, right_val])
            }
            Expr::Function { name: _, args } => {
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.extract_numeric_value(arg, row_idx)?);
                }
                Ok(arg_values)
            }
            _ => Err(Error::InvalidValue("Cannot extract arguments from expression".to_string())),
        }
    }
    
    fn extract_numeric_value(&self, expr: &Expr, row_idx: usize) -> Result<f64> {
        match expr {
            Expr::Literal(LiteralValue::Number(n)) => Ok(*n),
            Expr::Column(col_name) => {
                let col_values = self.dataframe.get_column_numeric_values(col_name)?;
                if row_idx < col_values.len() {
                    Ok(col_values[row_idx])
                } else {
                    Err(Error::IndexOutOfBounds { index: row_idx, size: col_values.len() })
                }
            }
            _ => Err(Error::InvalidValue("Cannot extract numeric value from expression".to_string())),
        }
    }
    
    fn evaluate_query_fallback(&self, expr: &Expr) -> Result<Vec<bool>> {
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);
        
        for row_idx in 0..row_count {
            // Simple fallback evaluation
            let value = match expr {
                Expr::Literal(LiteralValue::Boolean(b)) => *b,
                _ => true, // Simplified fallback
            };
            result.push(value);
        }
        
        Ok(result)
    }
}

impl<'a> OptimizedEvaluator<'a> {
    /// Create a new optimized evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self {
            dataframe,
            context,
            column_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }
    
    /// Evaluate query with vectorized operations where possible
    pub fn evaluate_query_vectorized(&self, expr: &Expr) -> Result<Vec<bool>> {
        // Try to use vectorized operations for simple column comparisons
        if let Some(vectorized_result) = self.try_vectorized_evaluation(expr)? {
            return Ok(vectorized_result);
        }
        
        // Fall back to row-by-row evaluation
        self.evaluate_query_row_by_row(expr)
    }
    
    /// Try to evaluate expression using vectorized operations
    fn try_vectorized_evaluation(&self, expr: &Expr) -> Result<Option<Vec<bool>>> {
        match expr {
            // Simple column comparisons can be vectorized
            Expr::Binary { left, op, right } => {
                if let (Expr::Column(col_name), Expr::Literal(literal)) = (left.as_ref(), right.as_ref()) {
                    return self.evaluate_column_comparison_vectorized(col_name, op, literal);
                }
                if let (Expr::Literal(literal), Expr::Column(col_name)) = (left.as_ref(), right.as_ref()) {
                    // Swap operands for commutative operations
                    let swapped_op = match op {
                        BinaryOp::Equal => BinaryOp::Equal,
                        BinaryOp::NotEqual => BinaryOp::NotEqual,
                        BinaryOp::LessThan => BinaryOp::GreaterThan,
                        BinaryOp::LessThanOrEqual => BinaryOp::GreaterThanOrEqual,
                        BinaryOp::GreaterThan => BinaryOp::LessThan,
                        BinaryOp::GreaterThanOrEqual => BinaryOp::LessThanOrEqual,
                        _ => return Ok(None), // Not vectorizable
                    };
                    return self.evaluate_column_comparison_vectorized(col_name, &swapped_op, literal);
                }
            }
            _ => {}
        }
        Ok(None)
    }
    
    /// Evaluate column comparison using vectorized operations
    fn evaluate_column_comparison_vectorized(&self, col_name: &str, op: &BinaryOp, literal: &LiteralValue) -> Result<Option<Vec<bool>>> {
        if !self.dataframe.contains_column(col_name) {
            return Err(Error::ColumnNotFound(col_name.to_string()));
        }
        
        // Get column values (try numeric first for better performance)
        if let Ok(numeric_values) = self.dataframe.get_column_numeric_values(col_name) {
            if let LiteralValue::Number(target) = literal {
                let result: Vec<bool> = match op {
                    BinaryOp::Equal => numeric_values.iter().map(|&v| (v - target).abs() < f64::EPSILON).collect(),
                    BinaryOp::NotEqual => numeric_values.iter().map(|&v| (v - target).abs() >= f64::EPSILON).collect(),
                    BinaryOp::LessThan => numeric_values.iter().map(|&v| v < *target).collect(),
                    BinaryOp::LessThanOrEqual => numeric_values.iter().map(|&v| v <= *target).collect(),
                    BinaryOp::GreaterThan => numeric_values.iter().map(|&v| v > *target).collect(),
                    BinaryOp::GreaterThanOrEqual => numeric_values.iter().map(|&v| v >= *target).collect(),
                    _ => return Ok(None), // Not supported for vectorization
                };
                return Ok(Some(result));
            }
        }
        
        // Fall back to string comparison
        let string_values = self.dataframe.get_column_string_values(col_name)?;
        if let LiteralValue::String(target) = literal {
            let result: Vec<bool> = match op {
                BinaryOp::Equal => string_values.iter().map(|v| v == target).collect(),
                BinaryOp::NotEqual => string_values.iter().map(|v| v != target).collect(),
                _ => return Ok(None), // String comparison only supports equality
            };
            return Ok(Some(result));
        }
        
        Ok(None)
    }
    
    /// Row-by-row evaluation as fallback
    fn evaluate_query_row_by_row(&self, expr: &Expr) -> Result<Vec<bool>> {
        let row_count = self.dataframe.row_count();
        let mut result = Vec::with_capacity(row_count);
        
        for row_idx in 0..row_count {
            let value = self.evaluate_expression_for_row(expr, row_idx)?;
            match value {
                LiteralValue::Boolean(b) => result.push(b),
                _ => return Err(Error::InvalidValue("Query expression must evaluate to boolean".to_string())),
            }
        }
        
        Ok(result)
    }
    
    /// Evaluate expression for a single row (same as regular evaluator but with caching)
    fn evaluate_expression_for_row(&self, expr: &Expr, row_idx: usize) -> Result<LiteralValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),
            
            Expr::Column(name) => {
                if !self.dataframe.contains_column(name) {
                    return Err(Error::ColumnNotFound(name.clone()));
                }
                
                // Use cached column data if available
                {
                    let cache = self.column_cache.borrow();
                    if let Some(cached_values) = cache.get(name) {
                        if row_idx < cached_values.len() {
                            return Ok(cached_values[row_idx].clone());
                        } else {
                            return Err(Error::IndexOutOfBounds { index: row_idx, size: cached_values.len() });
                        }
                    }
                }
                
                // Cache miss - load and parse column data
                let column_values = self.dataframe.get_column_string_values(name)?;
                let parsed_values: Vec<LiteralValue> = column_values.iter()
                    .map(|str_value| {
                        if let Ok(num) = str_value.parse::<f64>() {
                            LiteralValue::Number(num)
                        } else if let Ok(bool_val) = str_value.parse::<bool>() {
                            LiteralValue::Boolean(bool_val)
                        } else {
                            LiteralValue::String(str_value.clone())
                        }
                    })
                    .collect();
                
                {
                    let mut cache = self.column_cache.borrow_mut();
                    cache.insert(name.clone(), parsed_values.clone());
                }
                
                if row_idx < parsed_values.len() {
                    Ok(parsed_values[row_idx].clone())
                } else {
                    Err(Error::IndexOutOfBounds { index: row_idx, size: parsed_values.len() })
                }
            }
            
            Expr::Binary { left, op, right } => {
                // Always use short-circuiting in optimized evaluator
                match op {
                    BinaryOp::And => {
                        let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                        if let LiteralValue::Boolean(false) = left_val {
                            return Ok(LiteralValue::Boolean(false));
                        }
                        let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                        self.apply_binary_operation(&left_val, op, &right_val)
                    }
                    BinaryOp::Or => {
                        let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                        if let LiteralValue::Boolean(true) = left_val {
                            return Ok(LiteralValue::Boolean(true));
                        }
                        let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                        self.apply_binary_operation(&left_val, op, &right_val)
                    }
                    _ => {
                        let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                        let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                        self.apply_binary_operation(&left_val, op, &right_val)
                    }
                }
            }
            
            Expr::Unary { op, operand } => {
                let operand_val = self.evaluate_expression_for_row(operand, row_idx)?;
                self.apply_unary_operation(op, &operand_val)
            }
            
            Expr::Function { name, args } => {
                let arg_values: Result<Vec<f64>> = args.iter()
                    .map(|arg| {
                        let val = self.evaluate_expression_for_row(arg, row_idx)?;
                        match val {
                            LiteralValue::Number(n) => Ok(n),
                            _ => Err(Error::InvalidValue("Function arguments must be numeric".to_string())),
                        }
                    })
                    .collect();
                
                let arg_values = arg_values?;
                
                if let Some(func) = self.context.functions.get(name) {
                    let result = func(&arg_values);
                    Ok(LiteralValue::Number(result))
                } else {
                    Err(Error::InvalidValue(format!("Unknown function: {}", name)))
                }
            }
        }
    }
    
    /// Apply binary operation (shared with regular evaluator)
    fn apply_binary_operation(&self, left: &LiteralValue, op: &BinaryOp, right: &LiteralValue) -> Result<LiteralValue> {
        match (left, right, op) {
            // Numeric operations
            (LiteralValue::Number(l), LiteralValue::Number(r), op) => {
                match op {
                    BinaryOp::Add => Ok(LiteralValue::Number(l + r)),
                    BinaryOp::Subtract => Ok(LiteralValue::Number(l - r)),
                    BinaryOp::Multiply => Ok(LiteralValue::Number(l * r)),
                    BinaryOp::Divide => {
                        if *r == 0.0 {
                            Err(Error::InvalidValue("Division by zero".to_string()))
                        } else {
                            Ok(LiteralValue::Number(l / r))
                        }
                    }
                    BinaryOp::Modulo => Ok(LiteralValue::Number(l % r)),
                    BinaryOp::Power => Ok(LiteralValue::Number(l.powf(*r))),
                    BinaryOp::Equal => Ok(LiteralValue::Boolean((l - r).abs() < f64::EPSILON)),
                    BinaryOp::NotEqual => Ok(LiteralValue::Boolean((l - r).abs() >= f64::EPSILON)),
                    BinaryOp::LessThan => Ok(LiteralValue::Boolean(l < r)),
                    BinaryOp::LessThanOrEqual => Ok(LiteralValue::Boolean(l <= r)),
                    BinaryOp::GreaterThan => Ok(LiteralValue::Boolean(l > r)),
                    BinaryOp::GreaterThanOrEqual => Ok(LiteralValue::Boolean(l >= r)),
                    BinaryOp::And | BinaryOp::Or => Err(Error::InvalidValue("Logical operations require boolean operands".to_string())),
                }
            }
            
            // String operations
            (LiteralValue::String(l), LiteralValue::String(r), op) => {
                match op {
                    BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                    BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                    BinaryOp::Add => Ok(LiteralValue::String(format!("{}{}", l, r))),
                    _ => Err(Error::InvalidValue("Unsupported operation for strings".to_string())),
                }
            }
            
            // Boolean operations
            (LiteralValue::Boolean(l), LiteralValue::Boolean(r), op) => {
                match op {
                    BinaryOp::And => Ok(LiteralValue::Boolean(*l && *r)),
                    BinaryOp::Or => Ok(LiteralValue::Boolean(*l || *r)),
                    BinaryOp::Equal => Ok(LiteralValue::Boolean(l == r)),
                    BinaryOp::NotEqual => Ok(LiteralValue::Boolean(l != r)),
                    _ => Err(Error::InvalidValue("Unsupported operation for booleans".to_string())),
                }
            }
            
            // Mixed type comparisons
            (LiteralValue::Number(l), LiteralValue::String(r), op) => {
                if let Ok(r_num) = r.parse::<f64>() {
                    self.apply_binary_operation(&LiteralValue::Number(*l), op, &LiteralValue::Number(r_num))
                } else {
                    Err(Error::InvalidValue("Cannot compare number with non-numeric string".to_string()))
                }
            }
            
            (LiteralValue::String(l), LiteralValue::Number(r), op) => {
                if let Ok(l_num) = l.parse::<f64>() {
                    self.apply_binary_operation(&LiteralValue::Number(l_num), op, &LiteralValue::Number(*r))
                } else {
                    Err(Error::InvalidValue("Cannot compare non-numeric string with number".to_string()))
                }
            }
            
            _ => Err(Error::InvalidValue("Unsupported operand types for operation".to_string())),
        }
    }
    
    /// Apply unary operation
    fn apply_unary_operation(&self, op: &UnaryOp, operand: &LiteralValue) -> Result<LiteralValue> {
        match (op, operand) {
            (UnaryOp::Not, LiteralValue::Boolean(b)) => Ok(LiteralValue::Boolean(!b)),
            (UnaryOp::Negate, LiteralValue::Number(n)) => Ok(LiteralValue::Number(-n)),
            _ => Err(Error::InvalidValue("Unsupported unary operation".to_string())),
        }
    }
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
    
    /// Execute a query on a DataFrame
    pub fn query(&self, dataframe: &DataFrame, query_str: &str) -> Result<DataFrame> {
        // Tokenize the query string
        let input_str: &'static str = unsafe { std::mem::transmute(query_str) };
        let mut lexer = Lexer::new(input_str);
        let mut tokens = Vec::new();
        
        loop {
            let token = lexer.next_token()?;
            let is_eof = matches!(token, Token::Eof);
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
        let selected_indices: Vec<usize> = mask.iter()
            .enumerate()
            .filter_map(|(idx, &include)| if include { Some(idx) } else { None })
            .collect();
        
        // Create filtered columns
        for col_name in dataframe.column_names() {
            let column_values = dataframe.get_column_string_values(&col_name)?;
            let filtered_values: Vec<String> = selected_indices.iter()
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
            let is_eof = matches!(token, Token::Eof);
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

// Manual Clone implementation for QueryContext since functions can't be cloned
impl Clone for QueryContext {
    fn clone(&self) -> Self {
        let mut new_context = Self {
            variables: self.variables.clone(),
            functions: HashMap::new(),
            compiled_expressions: Arc::clone(&self.compiled_expressions),
            jit_stats: Arc::clone(&self.jit_stats),
            jit_threshold: self.jit_threshold,
            jit_enabled: self.jit_enabled,
        };
        
        // Re-add built-in functions
        new_context.add_builtin_functions();
        
        new_context
    }
}