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
mod engine;
mod evaluator;
mod lexer_parser;

// Re-export all public APIs to maintain backward compatibility
pub use ast::{BinaryOp, Expr, LiteralValue, Token, UnaryOp};
pub use engine::{QueryEngine, QueryExt};
pub use evaluator::{Evaluator, JitEvaluator, JitQueryStats, OptimizedEvaluator, QueryContext};
pub use lexer_parser::{Lexer, Parser};
