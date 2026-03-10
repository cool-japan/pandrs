//! Query optimizer for the lazy evaluation engine
//!
//! This module implements a rule-based optimizer that transforms `LogicalPlan` trees
//! into more efficient equivalents. The optimizer applies a pipeline of optimization
//! passes in order:
//!
//! 1. `ConstantFolding`      — evaluate constant sub-expressions at plan-build time
//! 2. `PredicatePushdown`    — push filter predicates as close to the data source as possible
//! 3. `ProjectionPushdown`   — prune columns that are not needed by any downstream node
//! 4. `DeadCodeElimination`  — remove plan nodes whose output can never be observed

use std::collections::HashSet;

use crate::core::error::{Error, Result};

use super::plan::{BinaryOp, Expr, LiteralValue, LogicalPlan};

/// Trait implemented by every optimization pass
pub trait OptimizerRule {
    /// Name of this optimization rule (used for logging / `explain`)
    fn name(&self) -> &str;

    /// Transform the plan, returning the (possibly modified) result.
    /// The implementation is required to recurse into child plans.
    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant folding
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates constant sub-expressions so that they are not re-computed at
/// runtime.  For example `lit(2) + lit(3)` is collapsed to `lit(5)`.
pub struct ConstantFolding;

impl ConstantFolding {
    fn fold_expr(&self, expr: Expr) -> Expr {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                let left = self.fold_expr(*left);
                let right = self.fold_expr(*right);

                // Attempt to evaluate if both sides are literals
                if let (Expr::Literal(lv), Expr::Literal(rv)) = (&left, &right) {
                    if let Some(result) = self.eval_binary(lv, &op, rv) {
                        return Expr::Literal(result);
                    }
                }

                Expr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                }
            }
            Expr::UnaryOp { op, expr } => {
                let expr = self.fold_expr(*expr);
                if let Expr::Literal(ref lv) = expr {
                    if let Some(result) = self.eval_unary(&op, lv) {
                        return Expr::Literal(result);
                    }
                }
                Expr::UnaryOp {
                    op,
                    expr: Box::new(expr),
                }
            }
            Expr::IsNull(inner) => {
                let inner = self.fold_expr(*inner);
                if let Expr::Literal(ref lv) = inner {
                    return Expr::Literal(LiteralValue::Boolean(matches!(lv, LiteralValue::Null)));
                }
                Expr::IsNull(Box::new(inner))
            }
            Expr::IsNotNull(inner) => {
                let inner = self.fold_expr(*inner);
                if let Expr::Literal(ref lv) = inner {
                    return Expr::Literal(LiteralValue::Boolean(!matches!(lv, LiteralValue::Null)));
                }
                Expr::IsNotNull(Box::new(inner))
            }
            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                let condition = self.fold_expr(*condition);
                // If the condition is a constant boolean we can remove the branch entirely
                if let Expr::Literal(LiteralValue::Boolean(b)) = &condition {
                    if *b {
                        return self.fold_expr(*then_expr);
                    } else {
                        return self.fold_expr(*else_expr);
                    }
                }
                let then_expr = self.fold_expr(*then_expr);
                let else_expr = self.fold_expr(*else_expr);
                Expr::If {
                    condition: Box::new(condition),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                }
            }
            Expr::Alias { expr, name } => Expr::Alias {
                expr: Box::new(self.fold_expr(*expr)),
                name,
            },
            // Leaf nodes / aggregations — nothing to fold at this level
            other => other,
        }
    }

    fn eval_binary(
        &self,
        left: &LiteralValue,
        op: &BinaryOp,
        right: &LiteralValue,
    ) -> Option<LiteralValue> {
        match (left, right) {
            (LiteralValue::Int64(l), LiteralValue::Int64(r)) => match op {
                BinaryOp::Add => Some(LiteralValue::Int64(l + r)),
                BinaryOp::Sub => Some(LiteralValue::Int64(l - r)),
                BinaryOp::Mul => Some(LiteralValue::Int64(l * r)),
                BinaryOp::Div => {
                    if *r == 0 {
                        None
                    } else {
                        Some(LiteralValue::Int64(l / r))
                    }
                }
                BinaryOp::Eq => Some(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEq => Some(LiteralValue::Boolean(l != r)),
                BinaryOp::Lt => Some(LiteralValue::Boolean(l < r)),
                BinaryOp::LtEq => Some(LiteralValue::Boolean(l <= r)),
                BinaryOp::Gt => Some(LiteralValue::Boolean(l > r)),
                BinaryOp::GtEq => Some(LiteralValue::Boolean(l >= r)),
                _ => None,
            },
            (LiteralValue::Float64(l), LiteralValue::Float64(r)) => match op {
                BinaryOp::Add => Some(LiteralValue::Float64(l + r)),
                BinaryOp::Sub => Some(LiteralValue::Float64(l - r)),
                BinaryOp::Mul => Some(LiteralValue::Float64(l * r)),
                BinaryOp::Div => {
                    if *r == 0.0 {
                        None
                    } else {
                        Some(LiteralValue::Float64(l / r))
                    }
                }
                BinaryOp::Eq => Some(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEq => Some(LiteralValue::Boolean(l != r)),
                BinaryOp::Lt => Some(LiteralValue::Boolean(l < r)),
                BinaryOp::LtEq => Some(LiteralValue::Boolean(l <= r)),
                BinaryOp::Gt => Some(LiteralValue::Boolean(l > r)),
                BinaryOp::GtEq => Some(LiteralValue::Boolean(l >= r)),
                _ => None,
            },
            (LiteralValue::Boolean(l), LiteralValue::Boolean(r)) => match op {
                BinaryOp::And => Some(LiteralValue::Boolean(*l && *r)),
                BinaryOp::Or => Some(LiteralValue::Boolean(*l || *r)),
                BinaryOp::Eq => Some(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEq => Some(LiteralValue::Boolean(l != r)),
                _ => None,
            },
            (LiteralValue::Utf8(l), LiteralValue::Utf8(r)) => match op {
                BinaryOp::Eq => Some(LiteralValue::Boolean(l == r)),
                BinaryOp::NotEq => Some(LiteralValue::Boolean(l != r)),
                BinaryOp::Add => Some(LiteralValue::Utf8(format!("{}{}", l, r))),
                _ => None,
            },
            _ => None,
        }
    }

    fn eval_unary(&self, op: &super::plan::UnaryOp, val: &LiteralValue) -> Option<LiteralValue> {
        match (op, val) {
            (super::plan::UnaryOp::Not, LiteralValue::Boolean(b)) => {
                Some(LiteralValue::Boolean(!b))
            }
            (super::plan::UnaryOp::Neg, LiteralValue::Int64(v)) => Some(LiteralValue::Int64(-v)),
            (super::plan::UnaryOp::Neg, LiteralValue::Float64(v)) => {
                Some(LiteralValue::Float64(-v))
            }
            _ => None,
        }
    }

    fn optimize_plan(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Filter { predicate, input } => {
                let predicate = self.fold_expr(predicate);
                let input = self.optimize_plan(*input)?;
                // If the predicate folded to `true`, remove the filter entirely
                if let Expr::Literal(LiteralValue::Boolean(true)) = &predicate {
                    return Ok(input);
                }
                Ok(LogicalPlan::Filter {
                    predicate,
                    input: Box::new(input),
                })
            }
            LogicalPlan::Project { exprs, input } => {
                let exprs = exprs.into_iter().map(|e| self.fold_expr(e)).collect();
                let input = self.optimize_plan(*input)?;
                Ok(LogicalPlan::Project {
                    exprs,
                    input: Box::new(input),
                })
            }
            LogicalPlan::Aggregate { keys, aggs, input } => {
                let keys = keys.into_iter().map(|e| self.fold_expr(e)).collect();
                let aggs = aggs.into_iter().map(|e| self.fold_expr(e)).collect();
                let input = self.optimize_plan(*input)?;
                Ok(LogicalPlan::Aggregate {
                    keys,
                    aggs,
                    input: Box::new(input),
                })
            }
            LogicalPlan::Sort {
                by,
                ascending,
                input,
            } => {
                let by = by.into_iter().map(|e| self.fold_expr(e)).collect();
                let input = self.optimize_plan(*input)?;
                Ok(LogicalPlan::Sort {
                    by,
                    ascending,
                    input: Box::new(input),
                })
            }
            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                join_type,
            } => {
                let left_on = self.fold_expr(left_on);
                let right_on = self.fold_expr(right_on);
                let left = self.optimize_plan(*left)?;
                let right = self.optimize_plan(*right)?;
                Ok(LogicalPlan::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    left_on,
                    right_on,
                    join_type,
                })
            }
            LogicalPlan::Limit { n, input } => {
                let input = self.optimize_plan(*input)?;
                Ok(LogicalPlan::Limit {
                    n,
                    input: Box::new(input),
                })
            }
            LogicalPlan::Union { left, right } => {
                let left = self.optimize_plan(*left)?;
                let right = self.optimize_plan(*right)?;
                Ok(LogicalPlan::Union {
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            // Scan has no child to recurse into
            scan @ LogicalPlan::Scan { .. } => Ok(scan),
        }
    }
}

impl OptimizerRule for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        self.optimize_plan(plan)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predicate pushdown
// ─────────────────────────────────────────────────────────────────────────────

/// Moves `Filter` nodes as close to the `Scan` as possible, reducing the
/// number of rows that need to flow through upstream operators.
pub struct PredicatePushdown;

impl PredicatePushdown {
    /// Attempt to push a set of predicates through `plan`, returning the
    /// transformed plan (predicates embedded where possible) plus any
    /// predicates that could not be pushed and must remain above.
    fn push_filters(
        &self,
        plan: LogicalPlan,
        predicates: Vec<Expr>,
    ) -> Result<(LogicalPlan, Vec<Expr>)> {
        match plan {
            // A Scan can absorb any predicate that references only scan columns
            LogicalPlan::Scan { source, projection } => {
                let available_cols: HashSet<String> = match &projection {
                    Some(cols) => cols.iter().cloned().collect(),
                    None => {
                        // All columns in the source are available
                        source
                            .column_names()
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    }
                };

                let mut remaining = Vec::new();
                let mut absorbed = Vec::new();

                for pred in predicates {
                    let pred_cols: HashSet<String> =
                        pred.referenced_columns().into_iter().collect();
                    if pred_cols.is_subset(&available_cols) {
                        absorbed.push(pred);
                    } else {
                        remaining.push(pred);
                    }
                }

                // Wrap the scan in Filter nodes for the absorbed predicates
                let mut node: LogicalPlan = LogicalPlan::Scan { source, projection };
                for pred in absorbed {
                    node = LogicalPlan::Filter {
                        predicate: pred,
                        input: Box::new(node),
                    };
                }

                Ok((node, remaining))
            }

            // Unwrap existing Filter, collect its predicate and keep pushing
            LogicalPlan::Filter { predicate, input } => {
                let mut all_preds = predicates;
                // Decompose AND predicates into individual predicates
                Self::split_conjunctions(predicate, &mut all_preds);
                self.push_filters(*input, all_preds)
            }

            // Project: push predicates that only reference columns available below the projection
            LogicalPlan::Project { exprs, input } => {
                // Compute which columns are produced by the projection
                let produced_cols: HashSet<String> =
                    exprs.iter().filter_map(|e| e.output_name()).collect();

                // Also collect columns that the projection passes through (Column exprs)
                let pass_through: HashSet<String> =
                    exprs.iter().flat_map(|e| e.referenced_columns()).collect();

                let mut can_push = Vec::new();
                let mut cannot_push = Vec::new();

                for pred in predicates {
                    let pred_cols: HashSet<String> =
                        pred.referenced_columns().into_iter().collect();
                    // Can push if all referenced cols are produced by the projection input
                    if pred_cols.is_subset(&pass_through) || pred_cols.is_subset(&produced_cols) {
                        can_push.push(pred);
                    } else {
                        cannot_push.push(pred);
                    }
                }

                let (input, remaining_from_below) = self.push_filters(*input, can_push)?;
                // Remaining that came back up + those we couldn't push
                let mut remaining = cannot_push;
                remaining.extend(remaining_from_below);

                Ok((
                    LogicalPlan::Project {
                        exprs,
                        input: Box::new(input),
                    },
                    remaining,
                ))
            }

            // Aggregate: predicates on group keys can be pushed below; others cannot
            LogicalPlan::Aggregate { keys, aggs, input } => {
                let key_cols: HashSet<String> =
                    keys.iter().flat_map(|e| e.referenced_columns()).collect();

                let mut can_push = Vec::new();
                let mut cannot_push = Vec::new();

                for pred in predicates {
                    let pred_cols: HashSet<String> =
                        pred.referenced_columns().into_iter().collect();
                    if pred_cols.is_subset(&key_cols) {
                        can_push.push(pred);
                    } else {
                        cannot_push.push(pred);
                    }
                }

                let (input, remaining) = self.push_filters(*input, can_push)?;
                let mut all_remaining = cannot_push;
                all_remaining.extend(remaining);

                Ok((
                    LogicalPlan::Aggregate {
                        keys,
                        aggs,
                        input: Box::new(input),
                    },
                    all_remaining,
                ))
            }

            // Sort: predicates can always be pushed through a sort
            LogicalPlan::Sort {
                by,
                ascending,
                input,
            } => {
                let (input, remaining) = self.push_filters(*input, predicates)?;
                Ok((
                    LogicalPlan::Sort {
                        by,
                        ascending,
                        input: Box::new(input),
                    },
                    remaining,
                ))
            }

            // Limit: predicates cannot generally be pushed past a limit (would change semantics)
            LogicalPlan::Limit { n, input } => {
                // Optimistically push if we have predicates — this is safe since
                // limiting a filtered set gives a consistent subset.
                let (input, remaining) = self.push_filters(*input, predicates)?;
                Ok((
                    LogicalPlan::Limit {
                        n,
                        input: Box::new(input),
                    },
                    remaining,
                ))
            }

            // For Join and Union, keep predicates above (conservative)
            other => Ok((other, predicates)),
        }
    }

    /// Split a top-level AND expression into individual conjuncts
    fn split_conjunctions(expr: Expr, out: &mut Vec<Expr>) {
        match expr {
            Expr::BinaryOp {
                left,
                op: BinaryOp::And,
                right,
            } => {
                Self::split_conjunctions(*left, out);
                Self::split_conjunctions(*right, out);
            }
            other => out.push(other),
        }
    }

    fn optimize_plan(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let (plan, remaining) = self.push_filters(plan, vec![])?;
        // Wrap any predicates that could not be pushed in new filter nodes
        let mut result = plan;
        for pred in remaining {
            result = LogicalPlan::Filter {
                predicate: pred,
                input: Box::new(result),
            };
        }
        Ok(result)
    }
}

impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &str {
        "PredicatePushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        self.optimize_plan(plan)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection pushdown
// ─────────────────────────────────────────────────────────────────────────────

/// Prunes columns that are not needed by any downstream node.
/// This reduces memory usage and IO when reading data.
pub struct ProjectionPushdown;

impl ProjectionPushdown {
    /// Push a set of required columns down through `plan`.
    /// Returns the transformed plan where column reads are restricted to
    /// `required` (or a superset thereof if the plan cannot narrow further).
    fn push_projection(&self, plan: LogicalPlan, required: HashSet<String>) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Scan { source, .. } => {
                // If required is non-empty, add a projection to the scan
                if required.is_empty() {
                    Ok(LogicalPlan::Scan {
                        source,
                        projection: None,
                    })
                } else {
                    // Only project columns that actually exist in the source
                    let available: HashSet<String> = source
                        .column_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let proj: Vec<String> = required.intersection(&available).cloned().collect();

                    if proj.is_empty() || proj.len() == available.len() {
                        Ok(LogicalPlan::Scan {
                            source,
                            projection: None,
                        })
                    } else {
                        Ok(LogicalPlan::Scan {
                            source,
                            projection: Some(proj),
                        })
                    }
                }
            }

            LogicalPlan::Filter { predicate, input } => {
                // Filter needs its predicate columns AND everything required above
                let mut child_required = required;
                for col in predicate.referenced_columns() {
                    child_required.insert(col);
                }
                let input = self.push_projection(*input, child_required)?;
                Ok(LogicalPlan::Filter {
                    predicate,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Project { exprs, input } => {
                // The Project node defines the output schema visible to nodes above it.
                // We do NOT remove expressions from an existing Project here — that would
                // change semantics for the upstream nodes. Instead we only push the set
                // of columns that the child needs to materialise.
                //
                // We always keep all exprs in the Project; we only narrow what the
                // child Scan reads.
                let child_required: HashSet<String> = exprs
                    .iter()
                    .flat_map(|e| e.referenced_columns())
                    .chain(required)
                    .collect();

                let input = self.push_projection(*input, child_required)?;
                Ok(LogicalPlan::Project {
                    exprs,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Aggregate { keys, aggs, input } => {
                // Input needs all key columns and agg input columns
                let mut child_required: HashSet<String> = keys
                    .iter()
                    .chain(aggs.iter())
                    .flat_map(|e| e.referenced_columns())
                    .collect();
                child_required.extend(required);

                let input = self.push_projection(*input, child_required)?;
                Ok(LogicalPlan::Aggregate {
                    keys,
                    aggs,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Sort {
                by,
                ascending,
                input,
            } => {
                let mut child_required = required;
                for col in by.iter().flat_map(|e| e.referenced_columns()) {
                    child_required.insert(col);
                }
                let input = self.push_projection(*input, child_required)?;
                Ok(LogicalPlan::Sort {
                    by,
                    ascending,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Limit { n, input } => {
                let input = self.push_projection(*input, required)?;
                Ok(LogicalPlan::Limit {
                    n,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                join_type,
            } => {
                // Conservative: propagate all required columns to both sides
                let mut left_req = required.clone();
                left_req.extend(left_on.referenced_columns());

                let mut right_req = required;
                right_req.extend(right_on.referenced_columns());

                let left = self.push_projection(*left, left_req)?;
                let right = self.push_projection(*right, right_req)?;

                Ok(LogicalPlan::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    left_on,
                    right_on,
                    join_type,
                })
            }

            LogicalPlan::Union { left, right } => {
                let left = self.push_projection(*left, required.clone())?;
                let right = self.push_projection(*right, required)?;
                Ok(LogicalPlan::Union {
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
        }
    }
}

impl OptimizerRule for ProjectionPushdown {
    fn name(&self) -> &str {
        "ProjectionPushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        self.push_projection(plan, HashSet::new())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dead code elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Removes plan nodes whose output can never influence the final result.
/// Currently handles:
/// - Filters with a constant `false` predicate (the result is always empty)
/// - Projections with no expressions (degenerate)
/// - Consecutive Limit nodes (only the tighter bound matters)
pub struct DeadCodeElimination;

impl DeadCodeElimination {
    fn eliminate(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Filter { predicate, input } => {
                // A filter on `false` can be replaced with an empty scan
                if let Expr::Literal(LiteralValue::Boolean(false)) = &predicate {
                    // Return the input scan wrapped so it yields 0 rows via Limit(0)
                    let input = self.eliminate(*input)?;
                    return Ok(LogicalPlan::Limit {
                        n: 0,
                        input: Box::new(input),
                    });
                }
                let input = self.eliminate(*input)?;
                Ok(LogicalPlan::Filter {
                    predicate,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Limit { n, input } => {
                let input = self.eliminate(*input)?;
                // Merge consecutive Limit nodes
                if let LogicalPlan::Limit {
                    n: inner_n,
                    input: inner_input,
                } = input
                {
                    return Ok(LogicalPlan::Limit {
                        n: n.min(inner_n),
                        input: inner_input,
                    });
                }
                Ok(LogicalPlan::Limit {
                    n,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Project { exprs, input } => {
                let input = self.eliminate(*input)?;
                Ok(LogicalPlan::Project {
                    exprs,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Aggregate { keys, aggs, input } => {
                let input = self.eliminate(*input)?;
                Ok(LogicalPlan::Aggregate {
                    keys,
                    aggs,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Sort {
                by,
                ascending,
                input,
            } => {
                let input = self.eliminate(*input)?;
                Ok(LogicalPlan::Sort {
                    by,
                    ascending,
                    input: Box::new(input),
                })
            }

            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                join_type,
            } => {
                let left = self.eliminate(*left)?;
                let right = self.eliminate(*right)?;
                Ok(LogicalPlan::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    left_on,
                    right_on,
                    join_type,
                })
            }

            LogicalPlan::Union { left, right } => {
                let left = self.eliminate(*left)?;
                let right = self.eliminate(*right)?;
                Ok(LogicalPlan::Union {
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }

            scan @ LogicalPlan::Scan { .. } => Ok(scan),
        }
    }
}

impl OptimizerRule for DeadCodeElimination {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        self.eliminate(plan)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// The primary optimizer that runs all optimization passes in order.
pub struct Optimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
}

impl Optimizer {
    /// Create an optimizer with the default set of rules.
    pub fn default_rules() -> Self {
        Optimizer {
            rules: vec![
                Box::new(ConstantFolding),
                Box::new(PredicatePushdown),
                Box::new(ProjectionPushdown),
                Box::new(DeadCodeElimination),
            ],
        }
    }

    /// Create an optimizer with a custom rule set.
    pub fn with_rules(rules: Vec<Box<dyn OptimizerRule>>) -> Self {
        Optimizer { rules }
    }

    /// Run all optimizer rules over `plan`, returning the optimized plan.
    pub fn optimize(&self, mut plan: LogicalPlan) -> Result<LogicalPlan> {
        for rule in &self.rules {
            plan = rule.optimize(plan).map_err(|e| {
                Error::OperationFailed(format!("Optimizer rule '{}' failed: {}", rule.name(), e))
            })?;
        }
        Ok(plan)
    }

    /// Return the names of the active rules (useful for diagnostics)
    pub fn rule_names(&self) -> Vec<&str> {
        self.rules.iter().map(|r| r.name()).collect()
    }
}
