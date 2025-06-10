//! # Query Plan Explanation Formatting
//!
//! This module provides functionality for formatting query plans in different formats.

use super::core::{ExplainOptions, PlanNode};

impl PlanNode {
    /// Explains the plan in text format
    pub(crate) fn explain_text(&self, options: &ExplainOptions, indent: usize) -> String {
        let indent_str = " ".repeat(indent * 2);

        let node_str = match self {
            Self::Scan {
                table,
                columns,
                statistics,
            } => {
                let mut result = format!("{}Scan: {}", indent_str, table);

                if !columns.is_empty() {
                    result.push_str(&format!(" [{}]", columns.join(", ")));
                }

                if options.with_statistics {
                    if let Some(stats) = statistics {
                        result.push_str(&format!(
                            " (rows: {}, size: {} bytes)",
                            stats.row_count, stats.size_bytes
                        ));
                    }
                }

                result
            }
            Self::Project { columns, input } => {
                let mut result = format!("{}Project: [{}]", indent_str, columns.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
            Self::Filter {
                predicate,
                input,
                selectivity,
            } => {
                let mut result = format!("{}Filter: {}", indent_str, predicate);

                if options.with_statistics {
                    if let Some(sel) = selectivity {
                        result.push_str(&format!(" (selectivity: {:.2})", sel));
                    }
                }

                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
            Self::Join {
                join_type,
                left,
                right,
                keys,
            } => {
                let keys_str = keys
                    .iter()
                    .map(|(l, r)| format!("{} = {}", l, r))
                    .collect::<Vec<_>>()
                    .join(" AND ");

                let mut result = format!("{}Join: {} ON {}", indent_str, join_type, keys_str);
                result.push('\n');
                result.push_str(&left.explain_text(options, indent + 1));
                result.push('\n');
                result.push_str(&right.explain_text(options, indent + 1));
                result
            }
            Self::Aggregate {
                keys,
                aggregates,
                input,
            } => {
                let mut result = format!(
                    "{}Aggregate: group by [{}], agg [{}]",
                    indent_str,
                    keys.join(", "),
                    aggregates.join(", ")
                );
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
            Self::Sort { sort_exprs, input } => {
                let mut result = format!("{}Sort: [{}]", indent_str, sort_exprs.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
            Self::Limit { limit, input } => {
                let mut result = format!("{}Limit: {}", indent_str, limit);
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
            Self::Window {
                window_functions,
                input,
            } => {
                let mut result = format!("{}Window: [{}]", indent_str, window_functions.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
            Self::Custom {
                name,
                params,
                input,
            } => {
                let params_str = params
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");

                let mut result = format!("{}Custom: {} [{}]", indent_str, name, params_str);
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            }
        };

        node_str
    }

    /// Explains the plan in JSON format
    pub(crate) fn explain_json(&self, options: &ExplainOptions) -> String {
        // Simplified JSON serialization
        match self {
            Self::Scan {
                table,
                columns,
                statistics,
            } => {
                let stats_str = if options.with_statistics {
                    if let Some(stats) = statistics {
                        format!(
                            ", \"rows\": {}, \"size\": {}",
                            stats.row_count, stats.size_bytes
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                format!(
                    "{{\"type\": \"Scan\", \"table\": \"{}\", \"columns\": [{}]{}}}",
                    table,
                    columns
                        .iter()
                        .map(|c| format!("\"{}\"", c))
                        .collect::<Vec<_>>()
                        .join(", "),
                    stats_str
                )
            }
            Self::Project { columns, input } => {
                format!(
                    "{{\"type\": \"Project\", \"columns\": [{}], \"input\": {}}}",
                    columns
                        .iter()
                        .map(|c| format!("\"{}\"", c))
                        .collect::<Vec<_>>()
                        .join(", "),
                    input.explain_json(options)
                )
            }
            Self::Filter {
                predicate,
                input,
                selectivity,
            } => {
                let selectivity_str = if options.with_statistics {
                    if let Some(sel) = selectivity {
                        format!(", \"selectivity\": {:.2}", sel)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                format!(
                    "{{\"type\": \"Filter\", \"predicate\": \"{}\"{}, \"input\": {}}}",
                    escape_json(predicate),
                    selectivity_str,
                    input.explain_json(options)
                )
            }
            Self::Join {
                join_type,
                left,
                right,
                keys,
            } => {
                let keys_json = keys
                    .iter()
                    .map(|(l, r)| format!("{{\"left\": \"{}\", \"right\": \"{}\"}}", l, r))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(
                    "{{\"type\": \"Join\", \"join_type\": \"{}\", \"keys\": [{}], \"left\": {}, \"right\": {}}}",
                    join_type,
                    keys_json,
                    left.explain_json(options),
                    right.explain_json(options)
                )
            }
            Self::Aggregate {
                keys,
                aggregates,
                input,
            } => {
                format!(
                    "{{\"type\": \"Aggregate\", \"keys\": [{}], \"aggregates\": [{}], \"input\": {}}}",
                    keys.iter().map(|k| format!("\"{}\"", k)).collect::<Vec<_>>().join(", "),
                    aggregates.iter().map(|a| format!("\"{}\"", a)).collect::<Vec<_>>().join(", "),
                    input.explain_json(options)
                )
            }
            Self::Sort { sort_exprs, input } => {
                format!(
                    "{{\"type\": \"Sort\", \"sort_exprs\": [{}], \"input\": {}}}",
                    sort_exprs
                        .iter()
                        .map(|e| format!("\"{}\"", e))
                        .collect::<Vec<_>>()
                        .join(", "),
                    input.explain_json(options)
                )
            }
            Self::Limit { limit, input } => {
                format!(
                    "{{\"type\": \"Limit\", \"limit\": {}, \"input\": {}}}",
                    limit,
                    input.explain_json(options)
                )
            }
            Self::Window {
                window_functions,
                input,
            } => {
                format!(
                    "{{\"type\": \"Window\", \"window_functions\": [{}], \"input\": {}}}",
                    window_functions
                        .iter()
                        .map(|w| format!("\"{}\"", w))
                        .collect::<Vec<_>>()
                        .join(", "),
                    input.explain_json(options)
                )
            }
            Self::Custom {
                name,
                params,
                input,
            } => {
                let params_json = params
                    .iter()
                    .map(|(k, v)| format!("\"{}\"\"{}\"", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(
                    "{{\"type\": \"Custom\", \"name\": \"{}\", \"params\": {{{}}}, \"input\": {}}}",
                    name,
                    params_json,
                    input.explain_json(options)
                )
            }
        }
    }
}

/// Escapes a string for JSON
pub(crate) fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
