//! Query and evaluation engine for DataFrames
//!
//! This module provides pandas-like query functionality and expression evaluation:
//! - String-based query expressions (.query() method)
//! - Expression evaluation (.eval() method) 
//! - Boolean indexing with complex conditions
//! - Support for mathematical operations and comparisons
//! - Variable substitution and context evaluation
//!
//! The module is organized into:
//! - ast: Token and AST definitions (~150 lines)
//! - lexer_parser: Lexical analysis and parsing (~500 lines)  
//! - evaluator: Expression evaluation and JIT compilation (~1200 lines)
//! - engine: Query engine and DataFrame integration (~320 lines)

mod ast;
mod lexer_parser;
mod evaluator;
mod engine;

// Re-export all public APIs to maintain backward compatibility
pub use ast::{Token, Expr, LiteralValue, BinaryOp, UnaryOp};
pub use lexer_parser::{Lexer, Parser};
pub use evaluator::{QueryContext, Evaluator, JitEvaluator, OptimizedEvaluator, JitQueryStats};
pub use engine::{QueryEngine, QueryExt};