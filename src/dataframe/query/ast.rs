//! AST definitions for query expressions
//!
//! This module contains the token and abstract syntax tree (AST) definitions
//! used for parsing and representing query expressions.

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
    Unary { op: UnaryOp, operand: Box<Expr> },
    /// Function calls
    Function { name: String, args: Vec<Expr> },
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
