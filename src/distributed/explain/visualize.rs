//! # Query Plan Visualization
//!
//! This module provides functionality for visualizing query plans with DOT format.

use super::core::{ExplainOptions, PlanNode};

impl PlanNode {
    /// Explains the plan in DOT format (for visualization)
    pub(crate) fn explain_dot(&self, options: &ExplainOptions) -> String {
        let mut result = String::from("digraph plan {\n");
        result.push_str("  node [shape=box];\n");

        let mut node_counter = 0;
        self.explain_dot_recursive(&mut result, &mut node_counter, options);

        result.push_str("}\n");
        result
    }

    /// Helper for DOT format explanation
    pub(crate) fn explain_dot_recursive(
        &self,
        result: &mut String,
        counter: &mut usize,
        options: &ExplainOptions,
    ) -> usize {
        let current = *counter;
        *counter += 1;

        let node_label = match self {
            Self::Scan {
                table,
                columns,
                statistics,
            } => {
                let cols_str = if columns.is_empty() {
                    String::new()
                } else {
                    format!("\\nColumns: [{}]", columns.join(", "))
                };

                let stats_str = if options.with_statistics {
                    if let Some(stats) = statistics {
                        format!(
                            "\\nRows: {}\\nSize: {} bytes",
                            stats.row_count, stats.size_bytes
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                format!("Scan: {}{}{}", table, cols_str, stats_str)
            }
            Self::Project { columns, .. } => {
                format!("Project: [{}]", columns.join(", "))
            }
            Self::Filter {
                predicate,
                selectivity,
                ..
            } => {
                let selectivity_str = if options.with_statistics {
                    if let Some(sel) = selectivity {
                        format!("\\nSelectivity: {:.2}", sel)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                format!("Filter: {}{}", predicate, selectivity_str)
            }
            Self::Join {
                join_type, keys, ..
            } => {
                let keys_str = keys
                    .iter()
                    .map(|(l, r)| format!("{} = {}", l, r))
                    .collect::<Vec<_>>()
                    .join(" AND ");

                format!("Join: {}\\nON {}", join_type, keys_str)
            }
            Self::Aggregate {
                keys, aggregates, ..
            } => {
                format!(
                    "Aggregate:\\nGroup by: [{}]\\nAgg: [{}]",
                    keys.join(", "),
                    aggregates.join(", ")
                )
            }
            Self::Sort { sort_exprs, .. } => {
                format!("Sort: [{}]", sort_exprs.join(", "))
            }
            Self::Limit { limit, .. } => {
                format!("Limit: {}", limit)
            }
            Self::Window {
                window_functions, ..
            } => {
                format!("Window: [{}]", window_functions.join(", "))
            }
            Self::Custom { name, params, .. } => {
                let params_str = params
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("Custom: {}\\n[{}]", name, params_str)
            }
        };

        result.push_str(&format!(
            "  node{} [label=\"{}\"];\n",
            current,
            escape_dot(&node_label)
        ));

        match self {
            Self::Scan { .. } => {}
            Self::Project { input, .. }
            | Self::Filter { input, .. }
            | Self::Aggregate { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::Window { input, .. }
            | Self::Custom { input, .. } => {
                let child = input.explain_dot_recursive(result, counter, options);
                result.push_str(&format!("  node{} -> node{};\n", current, child));
            }
            Self::Join { left, right, .. } => {
                let left_child = left.explain_dot_recursive(result, counter, options);
                let right_child = right.explain_dot_recursive(result, counter, options);
                result.push_str(&format!("  node{} -> node{};\n", current, left_child));
                result.push_str(&format!("  node{} -> node{};\n", current, right_child));
            }
        }

        current
    }
}

/// Escapes a string for DOT
pub(crate) fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
