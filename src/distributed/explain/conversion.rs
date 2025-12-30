//! # Execution Plan Conversion
//!
//! This module provides functionality for converting execution plans to plan nodes.

use super::core::PlanNode;
use crate::distributed::execution::{ExecutionPlan, Operation};
use crate::distributed::window::WindowFunction;
use crate::error::Result;

/// Explains an execution plan
pub fn explain_plan(plan: &ExecutionPlan, options: &super::core::ExplainOptions) -> Result<String> {
    let plan_node = execution_plan_to_plan_node(plan)?;
    Ok(plan_node.explain(options))
}

/// Converts an execution plan to a plan node
pub fn execution_plan_to_plan_node(plan: &ExecutionPlan) -> Result<PlanNode> {
    let node = match plan.operation() {
        Operation::Select { columns } => {
            let input = dummy_scan_node(plan);

            PlanNode::Project {
                columns: columns.clone(),
                input: Box::new(input),
            }
        }
        Operation::Filter { predicate } => {
            let input = dummy_scan_node(plan);

            PlanNode::Filter {
                predicate: predicate.clone(),
                input: Box::new(input),
                selectivity: None,
            }
        }
        Operation::Join {
            right,
            join_type,
            left_keys,
            right_keys,
        } => {
            let left_input = dummy_scan_node(plan);
            let right_input = PlanNode::Scan {
                table: right.clone(),
                columns: vec![],
                statistics: None,
            };

            let join_type_str = match join_type {
                crate::distributed::execution::JoinType::Inner => "INNER",
                crate::distributed::execution::JoinType::Left => "LEFT",
                crate::distributed::execution::JoinType::Right => "RIGHT",
                crate::distributed::execution::JoinType::Full => "FULL OUTER",
                crate::distributed::execution::JoinType::Cross => "CROSS",
            };

            let keys = left_keys
                .iter()
                .zip(right_keys.iter())
                .map(|(l, r)| (l.clone(), r.clone()))
                .collect();

            PlanNode::Join {
                join_type: join_type_str.to_string(),
                left: Box::new(left_input),
                right: Box::new(right_input),
                keys,
            }
        }
        Operation::GroupBy { keys, aggregates } => {
            let input = dummy_scan_node(plan);
            let agg_exprs = aggregates
                .iter()
                .map(|agg| format!("{}({}) as {}", agg.function, agg.input, agg.output))
                .collect();

            PlanNode::Aggregate {
                keys: keys.clone(),
                aggregates: agg_exprs,
                input: Box::new(input),
            }
        }
        Operation::OrderBy { sort_exprs } => {
            let input = dummy_scan_node(plan);
            let sort_exprs_str = sort_exprs
                .iter()
                .map(|expr| {
                    let direction = if expr.ascending { "ASC" } else { "DESC" };
                    let nulls = if expr.nulls_first {
                        "NULLS FIRST"
                    } else {
                        "NULLS LAST"
                    };
                    format!("{} {} {}", expr.column, direction, nulls)
                })
                .collect();

            PlanNode::Sort {
                sort_exprs: sort_exprs_str,
                input: Box::new(input),
            }
        }
        Operation::Limit { limit } => {
            let input = dummy_scan_node(plan);

            PlanNode::Limit {
                limit: *limit,
                input: Box::new(input),
            }
        }
        Operation::Window { window_functions } => {
            let input = dummy_scan_node(plan);
            let window_exprs = window_functions.iter().map(|wf| wf.to_sql()).collect();

            PlanNode::Window {
                window_functions: window_exprs,
                input: Box::new(input),
            }
        }
        Operation::Custom { name, params } => {
            let input = dummy_scan_node(plan);

            PlanNode::Custom {
                name: name.clone(),
                params: params.clone(),
                input: Box::new(input),
            }
        }
    };

    Ok(node)
}

/// Creates a dummy scan node for an execution plan's input
pub fn dummy_scan_node(plan: &ExecutionPlan) -> PlanNode {
    if plan.inputs().is_empty() {
        // No inputs, use a dummy table
        PlanNode::Scan {
            table: "DUMMY".to_string(),
            columns: vec![],
            statistics: None,
        }
    } else {
        // Use the first input as the scan source
        PlanNode::Scan {
            table: plan.inputs()[0].clone(),
            columns: vec![],
            statistics: None,
        }
    }
}
