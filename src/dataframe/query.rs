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

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

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

/// Query execution context
pub struct QueryContext {
    /// Variable bindings for substitution
    pub variables: HashMap<String, LiteralValue>,
    /// Available functions
    pub functions: HashMap<String, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
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

/// Expression evaluator
pub struct Evaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator
    pub fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self { dataframe, context }
    }
    
    /// Evaluate an expression and return a boolean mask for filtering
    pub fn evaluate_query(&self, expr: &Expr) -> Result<Vec<bool>> {
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
    
    /// Evaluate an expression for a specific row
    fn evaluate_expression_for_row(&self, expr: &Expr, row_idx: usize) -> Result<LiteralValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),
            
            Expr::Column(name) => {
                if !self.dataframe.contains_column(name) {
                    return Err(Error::ColumnNotFound(name.clone()));
                }
                
                let column_values = self.dataframe.get_column_string_values(name)?;
                if row_idx < column_values.len() {
                    let str_value = &column_values[row_idx];
                    
                    // Try to parse as number first, then boolean, then keep as string
                    if let Ok(num) = str_value.parse::<f64>() {
                        Ok(LiteralValue::Number(num))
                    } else if let Ok(bool_val) = str_value.parse::<bool>() {
                        Ok(LiteralValue::Boolean(bool_val))
                    } else {
                        Ok(LiteralValue::String(str_value.clone()))
                    }
                } else {
                    Err(Error::IndexOutOfBounds { index: row_idx, size: column_values.len() })
                }
            }
            
            Expr::Binary { left, op, right } => {
                let left_val = self.evaluate_expression_for_row(left, row_idx)?;
                let right_val = self.evaluate_expression_for_row(right, row_idx)?;
                self.apply_binary_operation(&left_val, op, &right_val)
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
        
        // Evaluate expression to get boolean mask
        let evaluator = Evaluator::new(dataframe, &self.context);
        let mask = evaluator.evaluate_query(&expr)?;
        
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
        };
        
        // Re-add built-in functions
        new_context.add_builtin_functions();
        
        new_context
    }
}