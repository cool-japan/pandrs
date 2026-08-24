use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::column::Column;
use crate::error::Result;
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::operations::AggregateOp;
use crate::optimized::split_dataframe::group::AggregateOp as GroupAggregateOp;
use crate::optimized::split_dataframe::OptimizedDataFrame as SplitDataFrame;

/// Map the lazy-plan aggregation op onto the grouping implementation's op.
///
/// The two enums are variant-for-variant identical; they exist separately only
/// because the optimized and split DataFrame implementations were never merged.
fn to_group_aggregate_op(op: AggregateOp) -> GroupAggregateOp {
    match op {
        AggregateOp::Sum => GroupAggregateOp::Sum,
        AggregateOp::Mean => GroupAggregateOp::Mean,
        AggregateOp::Min => GroupAggregateOp::Min,
        AggregateOp::Max => GroupAggregateOp::Max,
        AggregateOp::Count => GroupAggregateOp::Count,
        AggregateOp::Std => GroupAggregateOp::Std,
        AggregateOp::Var => GroupAggregateOp::Var,
        AggregateOp::Median => GroupAggregateOp::Median,
        AggregateOp::First => GroupAggregateOp::First,
        AggregateOp::Last => GroupAggregateOp::Last,
        AggregateOp::Custom => GroupAggregateOp::Custom,
    }
}

/// DataFrame wrapper for lazy evaluation
#[derive(Clone)]
pub struct LazyFrame {
    // Original DataFrame
    source: Arc<OptimizedDataFrame>,
    // Queue of operations to apply
    operations: Vec<Operation>,
}

/// Operations for lazy evaluation
#[derive(Clone)]
pub enum Operation {
    /// Select columns
    Select(Vec<String>),
    /// Filtering
    Filter(String),
    /// Mapping function
    Map(Arc<dyn Fn(&Column) -> Result<Column> + Send + Sync>),
    /// Aggregation
    Aggregate {
        /// Columns to group by
        group_by: Vec<String>,
        /// Aggregation operations
        aggregations: Vec<(String, AggregateOp, String)>,
    },
    /// Join
    Join {
        /// Right DataFrame
        right: Arc<OptimizedDataFrame>,
        /// Left join key
        left_on: String,
        /// Right join key
        right_on: String,
        /// Join type
        join_type: crate::optimized::operations::JoinType,
    },
    /// Sort
    Sort {
        /// Column to sort by
        by: String,
        /// Whether to sort in ascending order
        ascending: bool,
    },
}

impl Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Select(columns) => {
                write!(f, "Select({})", columns.join(", "))
            }
            Operation::Filter(condition) => {
                write!(f, "Filter({})", condition)
            }
            Operation::Map(_) => {
                write!(f, "Map(...)")
            }
            Operation::Aggregate {
                group_by,
                aggregations,
            } => {
                write!(f, "Aggregate(group_by=[{}], aggs=[", group_by.join(", "))?;
                let aggs: Vec<String> = aggregations
                    .iter()
                    .map(|(col, op, alias)| format!("{} {:?} as {}", col, op, alias))
                    .collect();
                write!(f, "{}])", aggs.join(", "))
            }
            Operation::Join {
                right: _,
                left_on,
                right_on,
                join_type,
            } => {
                write!(f, "Join({:?}, {} = {})", join_type, left_on, right_on)
            }
            Operation::Sort { by, ascending } => {
                write!(
                    f,
                    "Sort({}, {})",
                    by,
                    if *ascending { "asc" } else { "desc" }
                )
            }
        }
    }
}

impl LazyFrame {
    /// Create a new LazyFrame
    pub fn new(df: OptimizedDataFrame) -> Self {
        Self {
            source: Arc::new(df),
            operations: Vec::new(),
        }
    }

    /// Select columns
    pub fn select(mut self, columns: &[&str]) -> Self {
        let columns = columns.iter().map(|&s| s.to_string()).collect();
        self.operations.push(Operation::Select(columns));
        self
    }

    /// Filter data
    pub fn filter(mut self, condition: &str) -> Self {
        self.operations
            .push(Operation::Filter(condition.to_string()));
        self
    }

    /// Apply mapping function
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(&Column) -> Result<Column> + Send + Sync + 'static,
    {
        self.operations.push(Operation::Map(Arc::new(f)));
        self
    }

    /// Perform aggregation
    pub fn aggregate<I, J>(mut self, group_by: I, aggregations: J) -> Self
    where
        I: IntoIterator<Item = String>,
        J: IntoIterator<Item = (String, AggregateOp, String)>,
    {
        self.operations.push(Operation::Aggregate {
            group_by: group_by.into_iter().collect(),
            aggregations: aggregations.into_iter().collect(),
        });
        self
    }

    /// Join operation
    pub fn join(
        mut self,
        right: OptimizedDataFrame,
        left_on: &str,
        right_on: &str,
        join_type: crate::optimized::operations::JoinType,
    ) -> Self {
        self.operations.push(Operation::Join {
            right: Arc::new(right),
            left_on: left_on.to_string(),
            right_on: right_on.to_string(),
            join_type,
        });
        self
    }

    /// Sort data
    pub fn sort(mut self, by: &str, ascending: bool) -> Self {
        self.operations.push(Operation::Sort {
            by: by.to_string(),
            ascending,
        });
        self
    }

    /// Execute computation graph and get results
    pub fn execute(self) -> Result<OptimizedDataFrame> {
        let mut df = (*self.source).clone();

        for op in self.operations {
            match op {
                Operation::Select(columns) => {
                    let columns_slice = columns.iter().map(|s| s.as_str()).collect::<Vec<_>>();
                    df = df.select(&columns_slice)?;
                }
                Operation::Filter(condition) => {
                    // Use parallel filtering
                    df = df.par_filter(&condition)?;
                }
                Operation::Map(f) => {
                    df = df.par_apply(|view| f(&view.clone().into_column()))?;
                }
                Operation::Aggregate {
                    group_by,
                    aggregations,
                } => {
                    // Delegate to the shared grouping implementation instead of
                    // re-deriving it here. The local copy stringified every key
                    // (so a missing key became the literal text "NULL" and
                    // merged with a genuine "NULL" cell), emitted its columns in
                    // hash-map order, and substituted 0.0 for undefined
                    // reductions.
                    let mut split_df = SplitDataFrame::new();
                    for name in df.column_names() {
                        let view = df.column(name.as_str())?;
                        split_df.add_column(name.clone(), view.column().clone())?;
                    }

                    let grouped = split_df.group_by_with_options(group_by.iter(), false)?;
                    let group_aggregations: Vec<(String, GroupAggregateOp, String)> = aggregations
                        .iter()
                        .map(|(col_name, op, alias)| {
                            (col_name.clone(), to_group_aggregate_op(*op), alias.clone())
                        })
                        .collect();
                    let aggregated = grouped.par_aggregate(group_aggregations)?;

                    let mut result = OptimizedDataFrame::new();
                    for name in aggregated.column_names() {
                        let view = aggregated.column(name.as_str())?;
                        result.add_column(name.clone(), view.into_column())?;
                    }

                    df = result;
                }
                Operation::Join {
                    right,
                    left_on,
                    right_on,
                    join_type,
                } => {
                    df = match join_type {
                        crate::optimized::operations::JoinType::Inner => {
                            df.inner_join(&right, left_on.as_str(), right_on.as_str())?
                        }
                        crate::optimized::operations::JoinType::Left => {
                            df.left_join(&right, left_on.as_str(), right_on.as_str())?
                        }
                        crate::optimized::operations::JoinType::Right => {
                            df.right_join(&right, left_on.as_str(), right_on.as_str())?
                        }
                        crate::optimized::operations::JoinType::Outer => {
                            df.outer_join(&right, left_on.as_str(), right_on.as_str())?
                        }
                    };
                }
                Operation::Sort { by, ascending } => {
                    // Delegate to the typed sort implementation. Stringifying
                    // the sort key made an Int64 column sort lexicographically
                    // (1, 10, 100, 2) and rebuilt every column with 0/""/false
                    // in place of NULL; `sort_by` compares the column's real
                    // type and preserves NULLs (placed last, as pandas does).
                    df = df.sort_by(&by, ascending)?;
                }
            }
        }

        Ok(df)
    }

    /// Display optimized computation graph
    pub fn explain(&self) -> String {
        let mut result = String::new();
        result.push_str("LazyFrame execution plan:\n");
        result.push_str("----------------------\n");
        result.push_str("SOURCE: OptimizedDataFrame\n");

        for (i, op) in self.operations.iter().enumerate() {
            result.push_str(&format!("{}: {:?}\n", i + 1, op));
        }

        result
    }
}
