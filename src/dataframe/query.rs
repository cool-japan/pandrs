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
//! - ast: Token and AST definitions
//! - lexer_parser: Lexical analysis and parsing
//! - ops: Element-level operator semantics shared by every evaluation path
//! - vectorized: Column-at-a-time expression evaluation
//! - evaluator: Row-by-row interpreter, tier selection and statistics
//! - filter: dtype-preserving row selection for query results
//! - engine: Query engine and DataFrame integration
//!
//! # Supported syntax
//!
//! - comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
//! - logic: `&&`/`&`/`and`, `||`/`|`/`or`, `!`/`~`/`not`
//! - arithmetic: `+`, `-`, `*`, `/`, `%`, `**`
//! - literals: numbers (`1`, `1.5`, `.5`, `1e-3`), strings (`'a'`, `"a"`),
//!   booleans (`true`/`True`, `false`/`False`)
//! - identifiers: bare names and backtick-quoted names (`` `my col` ``)
//! - context variables: `@name`, or a bare name that is not a column
//! - function calls: `sqrt(x)`, `abs(x)`, plus any function registered on the
//!   query context

mod ast;
mod engine;
mod evaluator;
mod filter;
mod lexer_parser;
mod ops;
mod vectorized;

// Re-export all public APIs to maintain backward compatibility
pub use ast::{BinaryOp, Expr, LiteralValue, Token, UnaryOp};
pub use engine::{QueryEngine, QueryExt};
pub use evaluator::{Evaluator, JitEvaluator, JitQueryStats, OptimizedEvaluator, QueryContext};
pub use lexer_parser::{Lexer, Parser};
