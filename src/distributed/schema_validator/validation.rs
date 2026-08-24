//! # Execution Plan Validation
//!
//! This module provides validation of execution plans against schemas,
//! ensuring type safety and preventing runtime errors.

use super::compatibility::are_join_compatible;
use super::core::SchemaValidator;
use crate::distributed::execution::{AggregateExpr, ExecutionPlan, Operation};
use crate::distributed::expr::{
    ColumnMeta, ColumnProjection, ExprDataType, ExprSchema, ExprValidator,
};
use crate::error::{Error, Result};

/// Builds the schema produced by a `SELECT`/projection of `columns` from
/// `schema` (columns not present in the source are dropped — a missing column
/// is caught by `validate_select` before this runs).
fn schema_of_columns(schema: &ExprSchema, columns: &[String]) -> ExprSchema {
    let mut out = ExprSchema::new();
    for col in columns {
        if let Some(meta) = schema.column(col) {
            out.add_column(meta.clone());
        }
    }
    out
}

/// Builds the schema produced by an aggregate: the group-by keys (with their
/// source types) plus one column per aggregate, named exactly as the SQL
/// generator names it (`alias` when provided, else `{func}_{column}`) and typed
/// by the aggregate function.
fn schema_of_aggregate(
    schema: &ExprSchema,
    keys: &[String],
    aggregates: &[AggregateExpr],
) -> ExprSchema {
    let mut out = ExprSchema::new();
    for key in keys {
        if let Some(meta) = schema.column(key) {
            out.add_column(meta.clone());
        }
    }
    for agg in aggregates {
        let out_name = if agg.alias.trim().is_empty() {
            format!("{}_{}", agg.function.trim().to_lowercase(), agg.column)
        } else {
            agg.alias.clone()
        };
        let data_type = match agg.function.trim().to_lowercase().as_str() {
            "count" => ExprDataType::Integer,
            "avg" | "mean" | "stddev" | "std" | "variance" | "var" | "median" => {
                ExprDataType::Float
            }
            // sum/min/max keep the source column's type when known.
            _ => schema
                .column(&agg.column)
                .map(|m| m.data_type.clone())
                .unwrap_or(ExprDataType::Float),
        };
        out.add_column(ColumnMeta::new(out_name, data_type, true, None));
    }
    out
}

impl SchemaValidator {
    /// Validates an execution plan against the schema of its input dataset,
    /// evolving the working schema through the pipeline so that **every**
    /// operation is checked against the columns actually visible to it (not
    /// just the first operation against the original schema).
    ///
    /// Validation is fail-safe: if the input schema is unknown, or an operation
    /// reshapes/combines the schema in a way we do not model precisely
    /// (join/window/projection/set-op/custom outputs), the remaining operations
    /// are deferred to DataFusion's own planning checks rather than risk
    /// rejecting a valid plan.
    pub fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        // A bare `SELECT * FROM t` (no operations) is always valid.
        if plan.operations().is_empty() {
            return Ok(());
        }

        // Resolve the primary input schema; if unknown, defer to the engine.
        let mut current = match self.schema(plan.input()) {
            Some(s) => s.clone(),
            None => return Ok(()),
        };

        for operation in plan.operations() {
            match operation {
                Operation::Select(columns) => {
                    self.validate_select(&current, columns)?;
                    current = schema_of_columns(&current, columns);
                }
                Operation::Filter(predicate) => {
                    self.validate_filter(&current, predicate)?;
                    // A filter does not change the schema.
                }
                Operation::Aggregate(keys, aggregates) => {
                    // Aggregate() is the form the public `.aggregate()` API
                    // emits; route it through the same check as GroupBy{} rather
                    // than letting the old catch-all silently approve it.
                    self.validate_groupby(&current, keys, aggregates)?;
                    current = schema_of_aggregate(&current, keys, aggregates);
                }
                Operation::GroupBy { keys, aggregates } => {
                    self.validate_groupby(&current, keys, aggregates)?;
                    current = schema_of_aggregate(&current, keys, aggregates);
                }
                Operation::OrderBy(sort_exprs) => {
                    self.validate_orderby(&current, sort_exprs)?;
                }
                Operation::Limit(_) | Operation::Distinct => {
                    // No column references, no schema change.
                }
                Operation::Join {
                    left_keys,
                    right_keys,
                    right,
                    ..
                } => {
                    // Join validation needs two schemas, but an ExecutionPlan
                    // structurally carries only one input(); resolve the right
                    // schema by name from the registered set.
                    match self.schema(right) {
                        Some(right_schema) => {
                            self.validate_join(&current, right_schema, left_keys, right_keys)?;
                        }
                        None => return Ok(()), // right schema unknown; defer
                    }
                    // The joined output introduces qualified/duplicate names we
                    // do not model; stop precise tracking here.
                    return Ok(());
                }
                Operation::Window(window_functions) => {
                    self.validate_window(&current, window_functions)?;
                    // Window adds computed columns we cannot name precisely.
                    return Ok(());
                }
                Operation::Custom { name, params } => {
                    self.validate_custom(&current, name, params)?;
                    // select_expr / with_column reshape the schema; stop here.
                    return Ok(());
                }
                Operation::Project(_)
                | Operation::Union(_)
                | Operation::Intersect(_)
                | Operation::Except(_) => {
                    // These reshape or combine schemas in ways we do not model;
                    // defer the remainder to DataFusion.
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Validates a `Custom` operation (`select_expr` / `with_column` /
    /// `create_udf`) against the current schema.
    fn validate_custom(
        &self,
        schema: &ExprSchema,
        name: &str,
        params: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        match name {
            "select_expr" => {
                let projections_json = params.get("projections").ok_or_else(|| {
                    Error::InvalidOperation(
                        "select_expr operation requires projections parameter".to_string(),
                    )
                })?;
                let projections: Vec<ColumnProjection> = serde_json::from_str(projections_json)
                    .map_err(|e| {
                        Error::DistributedProcessing(format!("Failed to parse projections: {}", e))
                    })?;
                self.validate_select_expr(schema, &projections)
            }
            "with_column" => {
                let column_name = params.get("column_name").ok_or_else(|| {
                    Error::InvalidOperation(
                        "with_column operation requires column_name parameter".to_string(),
                    )
                })?;
                let projection_json = params.get("projection").ok_or_else(|| {
                    Error::InvalidOperation(
                        "with_column operation requires projection parameter".to_string(),
                    )
                })?;
                let projection: ColumnProjection =
                    serde_json::from_str(projection_json).map_err(|e| {
                        Error::DistributedProcessing(format!("Failed to parse projection: {}", e))
                    })?;
                self.validate_with_column(schema, column_name, &projection)
            }
            // UDF creation does not reference the schema.
            "create_udf" => Ok(()),
            _ => Err(Error::NotImplemented(format!(
                "Schema validation for custom operation '{}' is not implemented",
                name
            ))),
        }
    }

    /// Validates a SELECT operation
    fn validate_select(&self, schema: &ExprSchema, columns: &[String]) -> Result<()> {
        for column in columns {
            if !schema.has_column(column) {
                return Err(Error::InvalidOperation(format!(
                    "Column not found in schema: {}",
                    column
                )));
            }
        }
        Ok(())
    }

    /// Validates a SELECT_EXPR operation
    fn validate_select_expr(
        &self,
        schema: &ExprSchema,
        projections: &[ColumnProjection],
    ) -> Result<()> {
        let validator = ExprValidator::new(schema);
        validator.validate_projections(projections)?;
        Ok(())
    }

    /// Validates a WITH_COLUMN operation
    fn validate_with_column(
        &self,
        schema: &ExprSchema,
        _column_name: &str,
        projection: &ColumnProjection,
    ) -> Result<()> {
        let validator = ExprValidator::new(schema);
        validator.validate_expr(&projection.expr)?;
        Ok(())
    }

    /// Validates a FILTER operation
    fn validate_filter(&self, _schema: &ExprSchema, predicate: &str) -> Result<()> {
        // For a simple implementation, we'll just check if it's a valid SQL predicate
        // In a more advanced implementation, we'd parse the predicate into an Expr
        // and validate it against the schema

        // Placeholder for SQL predicate validation
        // This is simplified, but can be enhanced with a proper SQL parser
        if predicate.is_empty() {
            return Err(Error::InvalidOperation(
                "Empty predicate in filter operation".to_string(),
            ));
        }

        // Basic check for balanced parentheses
        let mut paren_count = 0;
        for c in predicate.chars() {
            if c == '(' {
                paren_count += 1;
            } else if c == ')' {
                paren_count -= 1;
                if paren_count < 0 {
                    return Err(Error::InvalidOperation(format!(
                        "Unbalanced parentheses in predicate: {}",
                        predicate
                    )));
                }
            }
        }

        if paren_count != 0 {
            return Err(Error::InvalidOperation(format!(
                "Unbalanced parentheses in predicate: {}",
                predicate
            )));
        }

        Ok(())
    }

    /// Validates a JOIN operation
    fn validate_join(
        &self,
        left_schema: &ExprSchema,
        right_schema: &ExprSchema,
        left_keys: &[String],
        right_keys: &[String],
    ) -> Result<()> {
        if left_keys.len() != right_keys.len() {
            return Err(Error::InvalidOperation(format!(
                "Number of left keys ({}) does not match number of right keys ({})",
                left_keys.len(),
                right_keys.len()
            )));
        }

        for (left_key, right_key) in left_keys.iter().zip(right_keys.iter()) {
            // Check that keys exist in schemas
            let left_col = left_schema.column(left_key).ok_or_else(|| {
                Error::InvalidOperation(format!("Left join key not found in schema: {}", left_key))
            })?;

            let right_col = right_schema.column(right_key).ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Right join key not found in schema: {}",
                    right_key
                ))
            })?;

            // Check that keys have compatible types
            if !are_join_compatible(&left_col.data_type, &right_col.data_type) {
                return Err(Error::InvalidOperation(format!(
                    "Incompatible join key types: {:?} and {:?}",
                    left_col.data_type, right_col.data_type
                )));
            }
        }

        Ok(())
    }

    /// Validates a GROUP BY operation
    fn validate_groupby(
        &self,
        schema: &ExprSchema,
        keys: &[String],
        aggregates: &[crate::distributed::execution::AggregateExpr],
    ) -> Result<()> {
        // Check that keys exist in schema
        for key in keys {
            if !schema.has_column(key) {
                return Err(Error::InvalidOperation(format!(
                    "Grouping key not found in schema: {}",
                    key
                )));
            }
        }

        // Check that aggregated columns exist in schema
        for agg in aggregates {
            let func = agg.function.trim().to_lowercase();

            // `COUNT(*)` references no named column.
            let is_count_star = func == "count" && agg.column.trim() == "*";

            if !is_count_star && !schema.has_column(&agg.column) {
                return Err(Error::InvalidOperation(format!(
                    "Aggregated column not found in schema: {}",
                    agg.column
                )));
            }

            // The accepted function set is kept in lock-step with the SQL
            // generator's `sanitize_agg_function` allow-list so validation never
            // rejects a function the generator would emit and DataFusion would
            // execute (e.g. `sum`/`count` over an integer column, `mean`,
            // `stddev`, `median`).
            match func.as_str() {
                // count works on any column; min/max on any orderable type.
                "count" | "min" | "max" => {}
                // These require a numeric column.
                "sum" | "avg" | "mean" | "stddev" | "std" | "variance" | "var" | "median" => {
                    if is_count_star {
                        // unreachable (count handled above), kept for clarity
                    } else if let Some(col) = schema.column(&agg.column) {
                        match col.data_type {
                            ExprDataType::Integer | ExprDataType::Float => {}
                            _ => {
                                return Err(Error::InvalidOperation(format!(
                                    "Aggregation function '{}' requires a numeric column but '{}' has type {:?}",
                                    agg.function, agg.column, col.data_type
                                )));
                            }
                        }
                    }
                }
                _ => {
                    return Err(Error::InvalidOperation(format!(
                        "Unknown aggregation function: {}",
                        agg.function
                    )));
                }
            }
        }

        Ok(())
    }

    /// Validates an ORDER BY operation
    fn validate_orderby(
        &self,
        schema: &ExprSchema,
        sort_exprs: &[crate::distributed::execution::SortExpr],
    ) -> Result<()> {
        // Check that sort columns exist in schema
        for sort_expr in sort_exprs {
            if !schema.has_column(&sort_expr.column) {
                return Err(Error::InvalidOperation(format!(
                    "Sort column not found in schema: {}",
                    sort_expr.column
                )));
            }
        }

        Ok(())
    }

    /// Validates a WINDOW operation
    pub(crate) fn validate_window(
        &self,
        schema: &ExprSchema,
        window_functions: &[String],
    ) -> Result<()> {
        use crate::distributed::expr::ExprDataType;

        if window_functions.is_empty() {
            return Err(Error::InvalidOperation(
                "Window operation requires at least one window function".to_string(),
            ));
        }

        for sql in window_functions {
            let spec = parse_window_sql(sql)?;

            // Check input columns exist in schema
            for col in &spec.input_columns {
                if !schema.has_column(col) {
                    return Err(Error::InvalidOperation(format!(
                        "Window function input column '{}' not found in schema",
                        col
                    )));
                }
            }

            // Check PARTITION BY columns exist in schema
            for col in &spec.partition_by {
                if !schema.has_column(col) {
                    return Err(Error::InvalidOperation(format!(
                        "Window function PARTITION BY column '{}' not found in schema",
                        col
                    )));
                }
            }

            // Check ORDER BY columns exist in schema
            for col in &spec.order_by {
                if !schema.has_column(col) {
                    return Err(Error::InvalidOperation(format!(
                        "Window function ORDER BY column '{}' not found in schema",
                        col
                    )));
                }
            }

            // Numeric type check for SUM, AVG, STDDEV, VARIANCE
            match spec.func_name.as_str() {
                "SUM" | "AVG" | "STDDEV" | "VARIANCE" => {
                    for col in &spec.input_columns {
                        let col_meta = schema.column(col).ok_or_else(|| {
                            Error::InvalidOperation(format!(
                                "Window function input column '{}' not found in schema",
                                col
                            ))
                        })?;
                        match col_meta.data_type {
                            ExprDataType::Integer | ExprDataType::Float => {
                                // Valid numeric types
                            }
                            _ => {
                                return Err(Error::InvalidOperation(format!(
                                    "Window function '{}' requires a numeric column but '{}' has type {:?}",
                                    spec.func_name, col, col_meta.data_type
                                )));
                            }
                        }
                    }
                }
                _ => {}
            }

            // Sortability check: ORDER BY columns cannot be Boolean
            for col in &spec.order_by {
                let col_meta = schema.column(col).ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "Window function ORDER BY column '{}' not found in schema",
                        col
                    ))
                })?;
                if col_meta.data_type == ExprDataType::Boolean {
                    return Err(Error::InvalidOperation(format!(
                        "Window function ORDER BY column '{}' has unsortable type Boolean",
                        col
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Parsed representation of a SQL window function expression
struct ParsedWindowSpec {
    func_name: String,
    input_columns: Vec<String>,
    partition_by: Vec<String>,
    order_by: Vec<String>,
}

/// Case-insensitive substring search that returns a byte offset **into
/// `haystack`** (the original string), so the result is always safe to slice
/// with.
///
/// `needle` is expected to be an ASCII keyword (e.g. `" OVER "`, `"ORDER BY"`).
/// Unlike `haystack.to_uppercase().find(needle)`, this never mismatches offsets
/// when `haystack` contains non-ASCII characters whose upper-casing changes
/// byte length (e.g. `'ß'` → `"SS"`).
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    let (hl, nl) = (hb.len(), nb.len());
    if nl == 0 || nl > hl {
        return None;
    }
    for i in 0..=(hl - nl) {
        // Only match at char boundaries so the returned offset is sliceable.
        if !haystack.is_char_boundary(i) {
            continue;
        }
        if (0..nl).all(|j| hb[i + j].eq_ignore_ascii_case(&nb[j])) {
            return Some(i);
        }
    }
    None
}

/// Parses a SQL window function expression string into a `ParsedWindowSpec`.
///
/// Accepts expressions of the form:
/// - `"SUM(amount) OVER (PARTITION BY dept ORDER BY date ASC) AS dept_sum"`
/// - `"ROW_NUMBER(*) OVER (PARTITION BY region ORDER BY sales DESC) AS rn"`
/// - `"AVG(price) OVER () AS overall_avg"`
fn parse_window_sql(sql: &str) -> Result<ParsedWindowSpec> {
    // Find the function name (everything before first '(')
    let first_paren = sql
        .find('(')
        .ok_or_else(|| Error::InvalidOperation(format!("Invalid window function SQL: {}", sql)))?;
    let func_name = sql[..first_paren].trim().to_uppercase();

    const KNOWN_WINDOW_FUNCTIONS: &[&str] = &[
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "LAG",
        "LEAD",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "COUNT",
        "NTILE",
        "PERCENT_RANK",
        "CUME_DIST",
        "FIRST_VALUE",
        "LAST_VALUE",
        "NTH_VALUE",
        "STDDEV",
        "VARIANCE",
    ];
    if !KNOWN_WINDOW_FUNCTIONS.contains(&func_name.as_str()) {
        return Err(Error::InvalidOperation(format!(
            "Unknown window function '{}'; known functions are: {}",
            func_name,
            KNOWN_WINDOW_FUNCTIONS.join(", ")
        )));
    }

    // Find OVER keyword to split func args from window spec. Use a
    // byte-offset-preserving case-insensitive search: `sql.to_uppercase()` can
    // change byte length for non-ASCII column names (e.g. 'ß' -> "SS"), so an
    // offset found in the uppercased copy must NOT be used to slice the
    // original — that panics on a non-char-boundary or corrupts the slice.
    let over_pos = find_ci(sql, " OVER ").ok_or_else(|| {
        Error::InvalidOperation(format!("Missing OVER clause in window function: {}", sql))
    })?;

    // Extract function arguments (between first '(' and the ')' before OVER)
    let func_args_region = &sql[first_paren + 1..over_pos];
    // The func_args_region ends with the closing ')' of the function call
    let close_paren = func_args_region.rfind(')').ok_or_else(|| {
        Error::InvalidOperation(format!("Malformed window function SQL: {}", sql))
    })?;
    let func_args = &func_args_region[..close_paren];

    let input_columns: Vec<String> = func_args
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "*")
        .collect();

    // Extract OVER clause content (inside the outer parens after OVER)
    let after_over = &sql[over_pos + 6..]; // skip " OVER "
    let over_open = after_over
        .find('(')
        .ok_or_else(|| Error::InvalidOperation(format!("Missing '(' after OVER in: {}", sql)))?;
    let over_close = after_over.rfind(')').ok_or_else(|| {
        Error::InvalidOperation(format!("Missing ')' to close OVER clause in: {}", sql))
    })?;
    let over_content = &after_over[over_open + 1..over_close];

    // Parse PARTITION BY and ORDER BY from over_content, again using
    // byte-offset-preserving case-insensitive search so offsets stay valid on
    // the original (possibly non-ASCII) string.
    let (partition_by, order_by) = if let Some(pb_pos) = find_ci(over_content, "PARTITION BY") {
        let after_pb = &over_content[pb_pos + 12..]; // skip "PARTITION BY"

        let (pb_part, ob_part) = if let Some(ob_pos) = find_ci(after_pb, "ORDER BY") {
            (&after_pb[..ob_pos], &after_pb[ob_pos + 8..])
        } else {
            (after_pb, "")
        };

        let pb_cols: Vec<String> = pb_part
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let ob_cols: Vec<String> = ob_part
            .split(',')
            .map(|s| {
                let trimmed = s.trim();
                let upper = trimmed.to_uppercase();
                if upper.ends_with(" ASC") {
                    trimmed[..trimmed.len() - 4].trim().to_string()
                } else if upper.ends_with(" DESC") {
                    trimmed[..trimmed.len() - 5].trim().to_string()
                } else {
                    trimmed.to_string()
                }
            })
            .filter(|s| !s.is_empty())
            .collect();

        (pb_cols, ob_cols)
    } else if let Some(ob_pos) = find_ci(over_content, "ORDER BY") {
        let after_ob = &over_content[ob_pos + 8..];
        let ob_cols: Vec<String> = after_ob
            .split(',')
            .map(|s| {
                let trimmed = s.trim();
                let upper = trimmed.to_uppercase();
                if upper.ends_with(" ASC") {
                    trimmed[..trimmed.len() - 4].trim().to_string()
                } else if upper.ends_with(" DESC") {
                    trimmed[..trimmed.len() - 5].trim().to_string()
                } else {
                    trimmed.to_string()
                }
            })
            .filter(|s| !s.is_empty())
            .collect();
        (vec![], ob_cols)
    } else {
        (vec![], vec![])
    };

    Ok(ParsedWindowSpec {
        func_name,
        input_columns,
        partition_by,
        order_by,
    })
}

#[cfg(test)]
mod tests {
    use crate::distributed::expr::{ColumnMeta, ExprDataType, ExprSchema};
    use crate::distributed::schema_validator::core::SchemaValidator;

    fn make_schema() -> ExprSchema {
        let mut schema = ExprSchema::new();
        schema.add_column(ColumnMeta::new("amount", ExprDataType::Float, false, None));
        schema.add_column(ColumnMeta::new("dept", ExprDataType::String, false, None));
        schema.add_column(ColumnMeta::new("date", ExprDataType::Date, false, None));
        schema.add_column(ColumnMeta::new("region", ExprDataType::String, false, None));
        schema.add_column(ColumnMeta::new(
            "active",
            ExprDataType::Boolean,
            false,
            None,
        ));
        schema
    }

    fn make_validator(schema: ExprSchema) -> SchemaValidator {
        let mut v = SchemaValidator::new();
        v.register_schema("test", schema);
        v
    }

    #[test]
    fn test_validate_window_valid() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf =
            vec!["SUM(amount) OVER (PARTITION BY dept ORDER BY date ASC) AS total".to_string()];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
    }

    #[test]
    fn test_validate_window_missing_column() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf =
            vec!["SUM(salary) OVER (PARTITION BY dept ORDER BY date ASC) AS total".to_string()];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("salary"),
            "Error should mention 'salary': {}",
            msg
        );
    }

    #[test]
    fn test_validate_window_nonnumeric_sum() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf = vec!["SUM(dept) OVER (PARTITION BY region ORDER BY date ASC) AS bad".to_string()];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("numeric") || msg.contains("dept"),
            "Error should mention numeric requirement or column: {}",
            msg
        );
    }

    #[test]
    fn test_validate_window_missing_partition_col() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf = vec![
            "ROW_NUMBER(*) OVER (PARTITION BY nonexistent ORDER BY date ASC) AS rn".to_string(),
        ];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent"),
            "Error should mention 'nonexistent': {}",
            msg
        );
    }

    #[test]
    fn test_validate_window_boolean_order_by() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf =
            vec!["ROW_NUMBER(*) OVER (PARTITION BY dept ORDER BY active ASC) AS rn".to_string()];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Boolean") || msg.contains("active"),
            "Error should mention Boolean or active: {}",
            msg
        );
    }

    #[test]
    fn test_validate_window_empty_over() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf = vec!["AVG(amount) OVER () AS overall_avg".to_string()];
        let result = validator.validate_window(&schema, &wf);
        assert!(
            result.is_ok(),
            "Expected Ok for empty OVER clause, got: {:?}",
            result
        );
    }

    #[test]
    fn test_validate_window_empty_list() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf: Vec<String> = vec![];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("least one") || msg.contains("empty"),
            "Error should mention empty list: {}",
            msg
        );
    }

    #[test]
    fn test_validate_window_unknown_function() {
        let schema = make_schema();
        let validator = make_validator(schema.clone());
        let wf = vec!["FOOBAR(amount) OVER ()".to_string()];
        let result = validator.validate_window(&schema, &wf);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("FOOBAR") || msg.contains("unknown") || msg.contains("Unknown"),
            "Error should mention unknown function: {}",
            msg
        );
    }
}
