//! Query Plan AST for lazy evaluation engine
//!
//! This module defines the logical plan representation used by the query optimizer.
//! Plans are constructed as trees of `LogicalPlan` nodes, with expressions represented
//! by the `Expr` enum.

use std::fmt;
use std::sync::Arc;

use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::operations::JoinType;

/// Aggregation expression types
#[derive(Debug, Clone, PartialEq)]
pub enum AggExpr {
    /// Sum of values
    Sum(Box<Expr>),
    /// Arithmetic mean
    Mean(Box<Expr>),
    /// Minimum value
    Min(Box<Expr>),
    /// Maximum value
    Max(Box<Expr>),
    /// Count of non-null values
    Count(Box<Expr>),
    /// Standard deviation
    StdDev(Box<Expr>),
}

impl fmt::Display for AggExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggExpr::Sum(e) => write!(f, "SUM({})", e),
            AggExpr::Mean(e) => write!(f, "MEAN({})", e),
            AggExpr::Min(e) => write!(f, "MIN({})", e),
            AggExpr::Max(e) => write!(f, "MAX({})", e),
            AggExpr::Count(e) => write!(f, "COUNT({})", e),
            AggExpr::StdDev(e) => write!(f, "STDDEV({})", e),
        }
    }
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Equal
    Eq,
    /// Not equal
    NotEq,
    /// Less than
    Lt,
    /// Less than or equal
    LtEq,
    /// Greater than
    Gt,
    /// Greater than or equal
    GtEq,
    /// Logical AND
    And,
    /// Logical OR
    Or,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::NotEq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::LtEq => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::GtEq => write!(f, ">="),
            BinaryOp::And => write!(f, "AND"),
            BinaryOp::Or => write!(f, "OR"),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    /// Logical NOT
    Not,
    /// Numeric negation
    Neg,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Not => write!(f, "NOT"),
            UnaryOp::Neg => write!(f, "-"),
        }
    }
}

/// Literal scalar values
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    /// 64-bit integer
    Int64(i64),
    /// 64-bit float
    Float64(f64),
    /// UTF-8 string
    Utf8(String),
    /// Boolean
    Boolean(bool),
    /// NULL value
    Null,
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiteralValue::Int64(v) => write!(f, "{}", v),
            LiteralValue::Float64(v) => write!(f, "{}", v),
            LiteralValue::Utf8(v) => write!(f, "\"{}\"", v),
            LiteralValue::Boolean(v) => write!(f, "{}", v),
            LiteralValue::Null => write!(f, "NULL"),
        }
    }
}

/// Expression AST node
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Reference to a column by name
    Column(String),
    /// Literal scalar value
    Literal(LiteralValue),
    /// Binary operation between two expressions
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// Unary operation on a single expression
    UnaryOp { op: UnaryOp, expr: Box<Expr> },
    /// Aggregate expression (used in groupby contexts)
    Agg(Box<AggExpr>),
    /// Cast expression to a different type
    Cast {
        expr: Box<Expr>,
        data_type: CastType,
    },
    /// Is-null test
    IsNull(Box<Expr>),
    /// Is-not-null test
    IsNotNull(Box<Expr>),
    /// Conditional (IF-THEN-ELSE)
    If {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// Alias an expression with a name
    Alias { expr: Box<Expr>, name: String },
    /// Wildcard (select all columns)
    Wildcard,
}

/// Target data types for casting
#[derive(Debug, Clone, PartialEq)]
pub enum CastType {
    Int64,
    Float64,
    Utf8,
    Boolean,
}

impl fmt::Display for CastType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CastType::Int64 => write!(f, "Int64"),
            CastType::Float64 => write!(f, "Float64"),
            CastType::Utf8 => write!(f, "Utf8"),
            CastType::Boolean => write!(f, "Boolean"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "col(\"{}\")", name),
            Expr::Literal(lit) => write!(f, "lit({})", lit),
            Expr::BinaryOp { left, op, right } => write!(f, "({} {} {})", left, op, right),
            Expr::UnaryOp { op, expr } => write!(f, "({} {})", op, expr),
            Expr::Agg(agg) => write!(f, "{}", agg),
            Expr::Cast { expr, data_type } => write!(f, "CAST({} AS {})", expr, data_type),
            Expr::IsNull(expr) => write!(f, "IS_NULL({})", expr),
            Expr::IsNotNull(expr) => write!(f, "IS_NOT_NULL({})", expr),
            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => write!(f, "IF({}, {}, {})", condition, then_expr, else_expr),
            Expr::Alias { expr, name } => write!(f, "{}.alias(\"{}\")", expr, name),
            Expr::Wildcard => write!(f, "*"),
        }
    }
}

impl Expr {
    /// Create a column reference expression
    pub fn col(name: impl Into<String>) -> Self {
        Expr::Column(name.into())
    }

    /// Create a literal integer expression
    pub fn lit_int(v: i64) -> Self {
        Expr::Literal(LiteralValue::Int64(v))
    }

    /// Create a literal float expression
    pub fn lit_float(v: f64) -> Self {
        Expr::Literal(LiteralValue::Float64(v))
    }

    /// Create a literal string expression
    pub fn lit_str(v: impl Into<String>) -> Self {
        Expr::Literal(LiteralValue::Utf8(v.into()))
    }

    /// Create a literal boolean expression
    pub fn lit_bool(v: bool) -> Self {
        Expr::Literal(LiteralValue::Boolean(v))
    }

    /// Apply an alias to this expression
    pub fn alias(self, name: impl Into<String>) -> Self {
        Expr::Alias {
            expr: Box::new(self),
            name: name.into(),
        }
    }

    /// Build a binary operation
    pub fn binary_op(self, op: BinaryOp, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op,
            right: Box::new(other),
        }
    }

    /// Equality comparison
    pub fn eq(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::Eq, other)
    }

    /// Not-equal comparison
    pub fn neq(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::NotEq, other)
    }

    /// Greater-than comparison
    pub fn gt(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::Gt, other)
    }

    /// Greater-than-or-equal comparison
    pub fn gt_eq(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::GtEq, other)
    }

    /// Less-than comparison
    pub fn lt(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::Lt, other)
    }

    /// Less-than-or-equal comparison
    pub fn lt_eq(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::LtEq, other)
    }

    /// Logical AND
    pub fn and(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::And, other)
    }

    /// Logical OR
    pub fn or(self, other: Expr) -> Self {
        self.binary_op(BinaryOp::Or, other)
    }

    /// Is null test
    pub fn is_null(self) -> Self {
        Expr::IsNull(Box::new(self))
    }

    /// Is not null test
    pub fn is_not_null(self) -> Self {
        Expr::IsNotNull(Box::new(self))
    }

    /// Sum aggregation
    pub fn sum(self) -> Self {
        Expr::Agg(Box::new(AggExpr::Sum(Box::new(self))))
    }

    /// Mean aggregation
    pub fn mean(self) -> Self {
        Expr::Agg(Box::new(AggExpr::Mean(Box::new(self))))
    }

    /// Min aggregation
    pub fn min(self) -> Self {
        Expr::Agg(Box::new(AggExpr::Min(Box::new(self))))
    }

    /// Max aggregation
    pub fn max(self) -> Self {
        Expr::Agg(Box::new(AggExpr::Max(Box::new(self))))
    }

    /// Count aggregation
    pub fn count(self) -> Self {
        Expr::Agg(Box::new(AggExpr::Count(Box::new(self))))
    }

    /// Standard deviation aggregation
    pub fn std_dev(self) -> Self {
        Expr::Agg(Box::new(AggExpr::StdDev(Box::new(self))))
    }

    /// Collect all column names referenced by this expression
    pub fn referenced_columns(&self) -> Vec<String> {
        let mut cols = Vec::new();
        self.collect_columns(&mut cols);
        cols
    }

    fn collect_columns(&self, cols: &mut Vec<String>) {
        match self {
            Expr::Column(name) => cols.push(name.clone()),
            Expr::BinaryOp { left, right, .. } => {
                left.collect_columns(cols);
                right.collect_columns(cols);
            }
            Expr::UnaryOp { expr, .. } => expr.collect_columns(cols),
            Expr::Agg(agg) => match agg.as_ref() {
                AggExpr::Sum(e)
                | AggExpr::Mean(e)
                | AggExpr::Min(e)
                | AggExpr::Max(e)
                | AggExpr::Count(e)
                | AggExpr::StdDev(e) => e.collect_columns(cols),
            },
            Expr::Cast { expr, .. } => expr.collect_columns(cols),
            Expr::IsNull(expr) | Expr::IsNotNull(expr) => expr.collect_columns(cols),
            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.collect_columns(cols);
                then_expr.collect_columns(cols);
                else_expr.collect_columns(cols);
            }
            Expr::Alias { expr, .. } => expr.collect_columns(cols),
            Expr::Literal(_) | Expr::Wildcard => {}
        }
    }

    /// Returns the output name for this expression (the alias if set, else the column name)
    pub fn output_name(&self) -> Option<String> {
        match self {
            Expr::Column(name) => Some(name.clone()),
            Expr::Alias { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    /// Returns true if this expression is a pure constant (no column references)
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Literal(_) => true,
            Expr::BinaryOp { left, right, .. } => left.is_constant() && right.is_constant(),
            Expr::UnaryOp { expr, .. } => expr.is_constant(),
            Expr::Cast { expr, .. } => expr.is_constant(),
            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => condition.is_constant() && then_expr.is_constant() && else_expr.is_constant(),
            _ => false,
        }
    }
}

/// Logical plan node — the AST for query plans
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan data from an in-memory DataFrame
    Scan {
        /// Source data
        source: Arc<OptimizedDataFrame>,
        /// Optional projection — columns to read (None = all)
        projection: Option<Vec<String>>,
    },
    /// Filter rows using a predicate expression
    Filter {
        /// Predicate expression (must evaluate to Boolean)
        predicate: Expr,
        /// Input plan
        input: Box<LogicalPlan>,
    },
    /// Project (select/transform) columns
    Project {
        /// Projection expressions
        exprs: Vec<Expr>,
        /// Input plan
        input: Box<LogicalPlan>,
    },
    /// Aggregate with optional group keys
    Aggregate {
        /// Group-by expressions
        keys: Vec<Expr>,
        /// Aggregation expressions
        aggs: Vec<Expr>,
        /// Input plan
        input: Box<LogicalPlan>,
    },
    /// Join two plans
    Join {
        /// Left input
        left: Box<LogicalPlan>,
        /// Right input
        right: Box<LogicalPlan>,
        /// Left join key expression
        left_on: Expr,
        /// Right join key expression
        right_on: Expr,
        /// Join type
        join_type: JoinType,
    },
    /// Sort rows
    Sort {
        /// Sort key expressions
        by: Vec<Expr>,
        /// Ascending flags (one per sort key)
        ascending: Vec<bool>,
        /// Input plan
        input: Box<LogicalPlan>,
    },
    /// Limit the number of rows
    Limit {
        /// Maximum rows to return
        n: usize,
        /// Input plan
        input: Box<LogicalPlan>,
    },
    /// Union of two plans (UNION ALL)
    Union {
        /// Left input
        left: Box<LogicalPlan>,
        /// Right input
        right: Box<LogicalPlan>,
    },
}

impl LogicalPlan {
    /// Return a human-readable string representation of the plan tree
    pub fn display(&self) -> String {
        self.display_indent(0)
    }

    fn display_indent(&self, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match self {
            LogicalPlan::Scan { source, projection } => {
                let proj = match projection {
                    Some(cols) => format!("[{}]", cols.join(", ")),
                    None => "*".to_string(),
                };
                format!(
                    "{}Scan: {} rows, projection={}\n",
                    pad,
                    source.row_count(),
                    proj
                )
            }
            LogicalPlan::Filter { predicate, input } => {
                let mut s = format!("{}Filter: {}\n", pad, predicate);
                s.push_str(&input.display_indent(indent + 1));
                s
            }
            LogicalPlan::Project { exprs, input } => {
                let expr_str = exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut s = format!("{}Project: {}\n", pad, expr_str);
                s.push_str(&input.display_indent(indent + 1));
                s
            }
            LogicalPlan::Aggregate { keys, aggs, input } => {
                let key_str = keys
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let agg_str = aggs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut s = format!("{}Aggregate: keys=[{}], aggs=[{}]\n", pad, key_str, agg_str);
                s.push_str(&input.display_indent(indent + 1));
                s
            }
            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                join_type,
            } => {
                let mut s = format!(
                    "{}Join ({:?}): {} = {}\n",
                    pad, join_type, left_on, right_on
                );
                s.push_str(&left.display_indent(indent + 1));
                s.push_str(&right.display_indent(indent + 1));
                s
            }
            LogicalPlan::Sort {
                by,
                ascending,
                input,
            } => {
                let sort_str = by
                    .iter()
                    .zip(ascending.iter())
                    .map(|(e, asc)| format!("{} {}", e, if *asc { "ASC" } else { "DESC" }))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut s = format!("{}Sort: {}\n", pad, sort_str);
                s.push_str(&input.display_indent(indent + 1));
                s
            }
            LogicalPlan::Limit { n, input } => {
                let mut s = format!("{}Limit: {}\n", pad, n);
                s.push_str(&input.display_indent(indent + 1));
                s
            }
            LogicalPlan::Union { left, right } => {
                let mut s = format!("{}Union:\n", pad);
                s.push_str(&left.display_indent(indent + 1));
                s.push_str(&right.display_indent(indent + 1));
                s
            }
        }
    }

    /// Collect all column names referenced in the plan (not recursing into subplans)
    pub fn referenced_columns_shallow(&self) -> Vec<String> {
        match self {
            LogicalPlan::Filter { predicate, .. } => predicate.referenced_columns(),
            LogicalPlan::Project { exprs, .. } => {
                exprs.iter().flat_map(|e| e.referenced_columns()).collect()
            }
            LogicalPlan::Aggregate { keys, aggs, .. } => keys
                .iter()
                .chain(aggs.iter())
                .flat_map(|e| e.referenced_columns())
                .collect(),
            LogicalPlan::Sort { by, .. } => {
                by.iter().flat_map(|e| e.referenced_columns()).collect()
            }
            LogicalPlan::Join {
                left_on, right_on, ..
            } => {
                let mut cols = left_on.referenced_columns();
                cols.extend(right_on.referenced_columns());
                cols
            }
            LogicalPlan::Scan { projection, .. } => projection.clone().unwrap_or_default(),
            LogicalPlan::Limit { .. } | LogicalPlan::Union { .. } => vec![],
        }
    }

    /// Get the input plan (if there is a single input)
    pub fn input(&self) -> Option<&LogicalPlan> {
        match self {
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => Some(input),
            _ => None,
        }
    }
}
