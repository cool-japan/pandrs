//! Multi-index column results for hierarchical GroupBy operations
//!
//! This module provides advanced multi-index column structures for presenting
//! hierarchical aggregation results in a pandas-like format with proper
//! column hierarchies and intuitive navigation.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::hierarchical_groupby::HierarchicalAgg;
use crate::series::base::Series;

/// Render a DataFrame column as owned strings, tolerating a `&'static
/// str`-backed column (e.g. built from `Series::new(vec!["a", "b"], ..)`, as
/// this crate's own examples and tests commonly do) that
/// [`DataFrame::get_column_string_values`] does not special-case. Mirrors
/// the identical helper in `crate::dataframe::transform`; kept local rather
/// than shared so the two modules' fixes stay independently self-contained.
fn column_as_strings(df: &DataFrame, col_name: &str) -> Result<Vec<String>> {
    if let Ok(series) = df.get_column::<&'static str>(col_name) {
        return Ok(series.values().iter().map(|s| s.to_string()).collect());
    }
    df.get_column_string_values(col_name)
}

/// Copy `source`'s `src_name` column into `target` under `dst_name`,
/// preserving the column's exact element type when it is one of this
/// crate's common scalar types, and otherwise bridging through
/// [`column_as_strings`] rather than silently dropping the column. This
/// replaces the fabrication it fixes: code that only ever tried
/// `Series<String>` and silently omitted every column of another type
/// (numeric group keys included) from its output.
fn copy_column_as(
    source: &DataFrame,
    src_name: &str,
    dst_name: &str,
    target: &mut DataFrame,
) -> Result<()> {
    macro_rules! try_copy {
        ($ty:ty) => {
            if let Ok(series) = source.get_column::<$ty>(src_name) {
                target.add_column(
                    dst_name.to_string(),
                    series.clone().with_name(dst_name.to_string()),
                )?;
                return Ok(());
            }
        };
    }
    try_copy!(String);
    try_copy!(i64);
    try_copy!(f64);
    try_copy!(i32);
    try_copy!(f32);
    try_copy!(bool);
    try_copy!(chrono::NaiveDate);
    try_copy!(chrono::NaiveDateTime);

    let values = column_as_strings(source, src_name)?;
    target.add_column(
        dst_name.to_string(),
        Series::new(values, Some(dst_name.to_string()))?,
    )?;
    Ok(())
}

/// Multi-level column index for hierarchical results
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MultiIndexColumn {
    /// Column hierarchy levels (e.g., ["Sales", "Q1", "mean"] for Sales.Q1.mean)
    pub levels: Vec<String>,
    /// Level names for each hierarchy level
    pub level_names: Vec<String>,
    /// Data type of the column
    pub dtype: String,
}

impl MultiIndexColumn {
    /// Create a new multi-index column
    pub fn new(levels: Vec<String>, level_names: Vec<String>, dtype: String) -> Self {
        Self {
            levels,
            level_names,
            dtype,
        }
    }

    /// Get the display name for this column (flattened representation)
    pub fn display_name(&self) -> String {
        self.levels.join(".")
    }

    /// Get the leaf column name (last level)
    pub fn leaf_name(&self) -> &str {
        self.levels.last().map(|s| s.as_str()).unwrap_or("")
    }

    /// Get parent levels (all but the last)
    pub fn parent_levels(&self) -> &[String] {
        if self.levels.len() > 1 {
            &self.levels[..self.levels.len() - 1]
        } else {
            &[]
        }
    }

    /// Get the depth of the column hierarchy
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Check if this column belongs to a specific parent hierarchy
    pub fn belongs_to_parent(&self, parent_levels: &[String]) -> bool {
        if parent_levels.len() >= self.levels.len() {
            return false;
        }

        self.levels[..parent_levels.len()] == *parent_levels
    }

    /// Create a new column with an additional level
    pub fn with_additional_level(&self, level: String) -> Self {
        let mut new_levels = self.levels.clone();
        new_levels.push(level);

        let mut new_level_names = self.level_names.clone();
        new_level_names.push(format!("level_{}", new_levels.len() - 1));

        Self::new(new_levels, new_level_names, self.dtype.clone())
    }
}

impl fmt::Display for MultiIndexColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Multi-index DataFrame for hierarchical results
#[derive(Debug, Clone)]
pub struct MultiIndexDataFrame {
    /// Underlying DataFrame with flattened column names
    data: DataFrame,
    /// Multi-index column specifications
    column_index: Vec<MultiIndexColumn>,
    /// Index hierarchy (row multi-index)
    index_hierarchy: Vec<String>,
    /// Column hierarchy level names
    #[allow(dead_code)] // retained: hierarchical column-level metadata for navigation
    column_level_names: Vec<String>,
    /// Metadata about the hierarchical structure
    metadata: MultiIndexMetadata,
}

/// Metadata about the multi-index structure
#[derive(Debug, Clone)]
pub struct MultiIndexMetadata {
    /// Maximum column hierarchy depth
    pub max_column_depth: usize,
    /// Number of index levels
    pub index_levels: usize,
    /// Total number of columns
    pub total_columns: usize,
    /// Aggregation functions used
    pub aggregation_functions: Vec<String>,
    /// Original grouping columns
    pub grouping_columns: Vec<String>,
}

impl MultiIndexDataFrame {
    /// Create a new multi-index DataFrame
    pub fn new(
        data: DataFrame,
        column_index: Vec<MultiIndexColumn>,
        index_hierarchy: Vec<String>,
    ) -> Self {
        let max_column_depth = column_index.iter().map(|c| c.depth()).max().unwrap_or(0);
        let index_levels = index_hierarchy.len();
        let total_columns = column_index.len();

        let column_level_names: Vec<String> = (0..max_column_depth)
            .map(|i| format!("level_{}", i))
            .collect();

        let metadata = MultiIndexMetadata {
            max_column_depth,
            index_levels,
            total_columns,
            aggregation_functions: Vec::new(),
            grouping_columns: index_hierarchy.clone(),
        };

        Self {
            data,
            column_index,
            index_hierarchy,
            column_level_names,
            metadata,
        }
    }

    /// Get the underlying DataFrame
    pub fn data(&self) -> &DataFrame {
        &self.data
    }

    /// Get the column index structure
    pub fn column_index(&self) -> &[MultiIndexColumn] {
        &self.column_index
    }

    /// Get the index hierarchy
    pub fn index_hierarchy(&self) -> &[String] {
        &self.index_hierarchy
    }

    /// Get metadata about the multi-index structure
    pub fn metadata(&self) -> &MultiIndexMetadata {
        &self.metadata
    }

    /// Select columns by level values
    pub fn select_columns_by_level(
        &self,
        level: usize,
        values: &[String],
    ) -> Result<MultiIndexDataFrame> {
        let selected_columns: Vec<_> = self
            .column_index
            .iter()
            .enumerate()
            .filter(|(_, col)| col.levels.get(level).map_or(false, |v| values.contains(v)))
            .collect();

        if selected_columns.is_empty() {
            return Err(Error::InvalidValue(
                "No columns match the selection criteria".to_string(),
            ));
        }

        // Create new DataFrame with selected columns
        let mut new_data = DataFrame::new();

        // Copy index columns, preserving whatever concrete type each one
        // actually has (numeric grouping keys included) instead of only
        // ever trying `Series<String>` and silently dropping the rest.
        for index_col in &self.index_hierarchy {
            copy_column_as(&self.data, index_col, index_col, &mut new_data)?;
        }

        // Copy selected data columns
        let mut new_column_index = Vec::new();
        for (_col_idx, multi_col) in selected_columns {
            let flat_name = multi_col.display_name();
            copy_column_as(&self.data, &flat_name, &flat_name, &mut new_data)?;
            new_column_index.push(multi_col.clone());
        }

        Ok(MultiIndexDataFrame::new(
            new_data,
            new_column_index,
            self.index_hierarchy.clone(),
        ))
    }

    /// Get columns that belong to a specific parent hierarchy
    pub fn get_columns_under_parent(&self, parent_levels: &[String]) -> Vec<&MultiIndexColumn> {
        self.column_index
            .iter()
            .filter(|col| col.belongs_to_parent(parent_levels))
            .collect()
    }

    /// Group columns by their parent levels up to a specific depth
    pub fn group_columns_by_parent(
        &self,
        depth: usize,
    ) -> BTreeMap<Vec<String>, Vec<&MultiIndexColumn>> {
        let mut groups: BTreeMap<Vec<String>, Vec<&MultiIndexColumn>> = BTreeMap::new();

        for column in &self.column_index {
            let parent_key = if depth < column.levels.len() {
                column.levels[..depth].to_vec()
            } else {
                column.levels.clone()
            };

            groups.entry(parent_key).or_default().push(column);
        }

        groups
    }

    /// Flatten the multi-index to a regular DataFrame with hierarchical column names
    pub fn flatten(&self) -> DataFrame {
        self.data.clone()
    }

    /// Get a hierarchical column summary
    pub fn column_summary(&self) -> ColumnHierarchySummary {
        let mut summary = ColumnHierarchySummary {
            total_columns: self.column_index.len(),
            max_depth: self.metadata.max_column_depth,
            levels_summary: BTreeMap::new(),
        };

        // Group by depth and analyze
        for column in &self.column_index {
            let depth = column.depth();
            let level_summary =
                summary
                    .levels_summary
                    .entry(depth)
                    .or_insert_with(|| LevelSummary {
                        count: 0,
                        unique_parents: std::collections::HashSet::new(),
                        sample_columns: Vec::new(),
                    });

            level_summary.count += 1;

            if column.depth() > 1 {
                level_summary
                    .unique_parents
                    .insert(column.parent_levels().join("."));
            }

            if level_summary.sample_columns.len() < 3 {
                level_summary.sample_columns.push(column.display_name());
            }
        }

        summary
    }

    /// Reshape to a different column hierarchy.
    ///
    /// Reordering a column's levels changes its flattened
    /// [`MultiIndexColumn::display_name`], which is also the name of its
    /// backing column in the underlying `DataFrame` -- so the data itself is
    /// copied to the new flattened name here, not just the `column_index`
    /// metadata. Two columns that the reorder maps onto the same flattened
    /// name (e.g. because a dropped level was the only thing distinguishing
    /// them) are rejected with an error instead of one silently overwriting
    /// the other in the result.
    pub fn pivot_columns(&self, new_level_order: &[usize]) -> Result<MultiIndexDataFrame> {
        if new_level_order.len() != self.metadata.max_column_depth {
            return Err(Error::InvalidValue(
                "New level order must match column depth".to_string(),
            ));
        }

        let mut new_data = DataFrame::new();

        // Index columns are untouched by a column-level reorder.
        for index_col in &self.index_hierarchy {
            copy_column_as(&self.data, index_col, index_col, &mut new_data)?;
        }

        // Reorder each column's levels and copy its underlying data to the
        // resulting flattened name.
        let mut new_column_index = Vec::new();
        let mut seen_targets: std::collections::HashSet<String> = std::collections::HashSet::new();
        for column in &self.column_index {
            let mut new_levels = Vec::new();
            let mut new_level_names = Vec::new();

            for &level_idx in new_level_order {
                if level_idx < column.levels.len() {
                    new_levels.push(column.levels[level_idx].clone());
                    if level_idx < column.level_names.len() {
                        new_level_names.push(column.level_names[level_idx].clone());
                    }
                }
            }

            let new_column =
                MultiIndexColumn::new(new_levels, new_level_names, column.dtype.clone());
            let old_name = column.display_name();
            let new_name = new_column.display_name();

            if !seen_targets.insert(new_name.clone()) {
                return Err(Error::DuplicateColumnName(format!(
                    "pivot_columns: level reorder maps two or more columns onto the same name \
                     '{}'; choose a level order that keeps every column's flattened name unique",
                    new_name
                )));
            }

            copy_column_as(&self.data, &old_name, &new_name, &mut new_data)?;
            new_column_index.push(new_column);
        }

        Ok(MultiIndexDataFrame::new(
            new_data,
            new_column_index,
            self.index_hierarchy.clone(),
        ))
    }

    /// Convert to a human-readable hierarchical table format
    pub fn to_hierarchical_string(&self, max_rows: Option<usize>) -> String {
        let mut output = String::new();

        // Header information
        output.push_str(&format!(
            "MultiIndex DataFrame ({}x{})\n",
            self.data.row_count(),
            self.column_index.len()
        ));
        output.push_str(&format!(
            "Index levels: {}\n",
            self.index_hierarchy.join(" | ")
        ));
        output.push_str(&format!(
            "Column hierarchy depth: {}\n",
            self.metadata.max_column_depth
        ));
        output.push_str("\n");

        // Column hierarchy display
        output.push_str("Column Hierarchy:\n");
        for (depth, columns) in self.group_columns_by_parent(1) {
            output.push_str(&format!("  {}: {}\n", depth.join("."), columns.len()));
        }
        output.push_str("\n");

        // Sample data
        let display_rows = max_rows.unwrap_or(10).min(self.data.row_count());
        output.push_str("Sample Data:\n");

        // Create column headers with hierarchy
        let mut header_lines: Vec<Vec<String>> = vec![Vec::new(); self.metadata.max_column_depth];

        // Add index headers
        for level in 0..self.metadata.max_column_depth {
            for index_col in &self.index_hierarchy {
                if level == 0 {
                    header_lines[level].push(index_col.clone());
                } else {
                    header_lines[level].push("".to_string());
                }
            }
        }

        // Add data column headers
        for column in &self.column_index {
            for level in 0..self.metadata.max_column_depth {
                if level < column.levels.len() {
                    header_lines[level].push(column.levels[level].clone());
                } else {
                    header_lines[level].push("".to_string());
                }
            }
        }

        // Print headers
        for (i, header_line) in header_lines.iter().enumerate() {
            output.push_str(&format!("Level {}: {}\n", i, header_line.join(" | ")));
        }
        output.push_str("\n");

        // Print data rows
        for row_idx in 0..display_rows {
            let mut row_data = Vec::new();

            // Add index data
            for index_col in &self.index_hierarchy {
                if let Ok(values) = self.data.get_column_string_values(index_col) {
                    if row_idx < values.len() {
                        row_data.push(values[row_idx].clone());
                    } else {
                        row_data.push("NULL".to_string());
                    }
                } else {
                    row_data.push("ERROR".to_string());
                }
            }

            // Add data columns
            for column in &self.column_index {
                let flat_name = column.display_name();
                if let Ok(values) = self.data.get_column_string_values(&flat_name) {
                    if row_idx < values.len() {
                        row_data.push(values[row_idx].clone());
                    } else {
                        row_data.push("NULL".to_string());
                    }
                } else {
                    row_data.push("ERROR".to_string());
                }
            }

            output.push_str(&format!("Row {}: {}\n", row_idx, row_data.join(" | ")));
        }

        if self.data.row_count() > display_rows {
            output.push_str(&format!(
                "... and {} more rows\n",
                self.data.row_count() - display_rows
            ));
        }

        output
    }
}

/// Summary of column hierarchy structure
#[derive(Debug)]
pub struct ColumnHierarchySummary {
    pub total_columns: usize,
    pub max_depth: usize,
    pub levels_summary: BTreeMap<usize, LevelSummary>,
}

/// Summary for a specific level in the hierarchy
#[derive(Debug)]
pub struct LevelSummary {
    pub count: usize,
    pub unique_parents: std::collections::HashSet<String>,
    pub sample_columns: Vec<String>,
}

/// Builder for creating multi-index DataFrames from hierarchical aggregations
#[derive(Debug)]
pub struct MultiIndexDataFrameBuilder {
    index_columns: Vec<String>,
    index_values: Vec<Vec<String>>,
    data_columns: HashMap<String, Series<String>>,
    column_hierarchy: Vec<MultiIndexColumn>,
    aggregation_metadata: Vec<String>,
}

impl MultiIndexDataFrameBuilder {
    /// Create a new builder
    pub fn new(index_columns: Vec<String>) -> Self {
        Self {
            index_columns,
            index_values: Vec::new(),
            data_columns: HashMap::new(),
            column_hierarchy: Vec::new(),
            aggregation_metadata: Vec::new(),
        }
    }

    /// Provide the concrete index (row-key) values for this multi-index
    /// DataFrame: one `Vec<String>` per index column declared in [`Self::new`],
    /// each of length equal to the eventual row count (the number of rows in
    /// the data columns added via [`Self::add_column`]).
    ///
    /// Must be called before [`Self::build`] whenever `index_columns` is
    /// non-empty: nothing about the columns added so far can tell the
    /// builder what a row's real key actually is, so without this call
    /// `build()` returns an error instead of fabricating placeholder text in
    /// the index columns.
    pub fn set_index_values(&mut self, index_values: Vec<Vec<String>>) -> Result<()> {
        if index_values.len() != self.index_columns.len() {
            return Err(Error::InvalidValue(format!(
                "set_index_values: expected {} column(s) of index values (one per index column \
                 declared in `new`), got {}",
                self.index_columns.len(),
                index_values.len()
            )));
        }
        self.index_values = index_values;
        Ok(())
    }

    /// Add a data column with multi-index specification
    pub fn add_column(
        &mut self,
        column_spec: MultiIndexColumn,
        data: Series<String>,
    ) -> Result<()> {
        let flat_name = column_spec.display_name();
        self.data_columns.insert(flat_name, data);
        self.column_hierarchy.push(column_spec);
        Ok(())
    }

    /// Add multiple columns for a hierarchical aggregation
    pub fn add_hierarchical_aggregation(
        &mut self,
        base_column: &str,
        agg_func: &str,
        level_results: Vec<(usize, Series<String>)>,
    ) -> Result<()> {
        for (level, data) in level_results {
            let column_spec = MultiIndexColumn::new(
                vec![
                    base_column.to_string(),
                    format!("level_{}", level),
                    agg_func.to_string(),
                ],
                vec![
                    "column".to_string(),
                    "level".to_string(),
                    "function".to_string(),
                ],
                "f64".to_string(),
            );

            self.add_column(column_spec, data)?;
        }

        self.aggregation_metadata
            .push(format!("{}_{}", base_column, agg_func));
        Ok(())
    }

    /// Build the multi-index DataFrame.
    ///
    /// Returns an error, rather than fabricating placeholder index values,
    /// when `index_columns` is non-empty but [`Self::set_index_values`] was
    /// never called (or supplied rows of the wrong length): there is no
    /// honest value to put in the index columns without it.
    pub fn build(self) -> Result<MultiIndexDataFrame> {
        let mut data = DataFrame::new();

        // Determine row count from data columns, falling back to the index
        // values when there are no data columns yet.
        let row_count = if let Some(first_series) = self.data_columns.values().next() {
            first_series.len()
        } else if let Some(first_index) = self.index_values.first() {
            first_index.len()
        } else {
            0
        };

        if !self.index_columns.is_empty() && self.index_values.len() != self.index_columns.len() {
            return Err(Error::InvalidValue(format!(
                "MultiIndexDataFrameBuilder::build: {} index column(s) declared but no \
                 index values were supplied; call set_index_values(..) before build() instead \
                 of relying on fabricated placeholder text",
                self.index_columns.len()
            )));
        }

        // Add index columns using the real values supplied via
        // `set_index_values`.
        for (index_col, values) in self.index_columns.iter().zip(self.index_values.iter()) {
            if values.len() != row_count {
                return Err(Error::InconsistentRowCount {
                    expected: row_count,
                    found: values.len(),
                });
            }
            let index_series: Series<String> =
                Series::new(values.clone(), Some(index_col.clone()))?;
            data.add_column(index_col.clone(), index_series)?;
        }

        // Add data columns
        for (flat_name, series) in self.data_columns {
            data.add_column(flat_name, series)?;
        }

        Ok(MultiIndexDataFrame::new(
            data,
            self.column_hierarchy,
            self.index_columns,
        ))
    }
}

/// Extension trait for converting hierarchical results to multi-index format
pub trait ToMultiIndex {
    /// Convert hierarchical aggregation results to multi-index DataFrame
    fn to_multi_index(&self, agg_specs: &[HierarchicalAgg]) -> Result<MultiIndexDataFrame>;
}

impl ToMultiIndex for DataFrame {
    fn to_multi_index(&self, agg_specs: &[HierarchicalAgg]) -> Result<MultiIndexDataFrame> {
        let mut column_index = Vec::new();

        // Identify index columns (grouping columns)
        let mut index_columns = Vec::new();
        let column_names = self.column_names();

        // Find grouping columns (typically the first few columns)
        for col_name in column_names {
            if !agg_specs.iter().any(|spec| {
                spec.level_functions
                    .iter()
                    .any(|(_, _, alias)| alias == col_name)
            }) {
                index_columns.push(col_name.to_string());
            }
        }

        // Create multi-index columns for aggregation results
        for agg_spec in agg_specs {
            for (level, func, _alias) in &agg_spec.level_functions {
                let column_spec = MultiIndexColumn::new(
                    vec![
                        agg_spec.column.clone(),
                        format!("level_{}", level),
                        func.as_str().to_string(),
                    ],
                    vec![
                        "column".to_string(),
                        "level".to_string(),
                        "function".to_string(),
                    ],
                    "f64".to_string(),
                );

                column_index.push(column_spec);
            }
        }

        Ok(MultiIndexDataFrame::new(
            self.clone(),
            column_index,
            index_columns,
        ))
    }
}

/// Utility functions for multi-index operations
pub mod utils {
    use super::*;

    /// Create a simple multi-index column specification
    pub fn simple_multi_index_column(levels: Vec<&str>, dtype: &str) -> MultiIndexColumn {
        let level_names = (0..levels.len()).map(|i| format!("level_{}", i)).collect();

        MultiIndexColumn::new(
            levels.into_iter().map(|s| s.to_string()).collect(),
            level_names,
            dtype.to_string(),
        )
    }

    /// Create column hierarchy for common aggregation patterns
    pub fn create_aggregation_hierarchy(
        columns: &[&str],
        functions: &[&str],
    ) -> Vec<MultiIndexColumn> {
        let mut result = Vec::new();

        for &column in columns {
            for &function in functions {
                result.push(simple_multi_index_column(vec![column, function], "f64"));
            }
        }

        result
    }

    /// Merge multiple multi-index DataFrames that share the same index
    /// hierarchy into one, combining their data columns side by side.
    ///
    /// All input frames must declare the identical `index_hierarchy` (same
    /// column names, in the same order) and must each have unique row keys
    /// covering exactly the same set of keys as every other frame -- the
    /// realistic case for merging aggregation results computed over the
    /// same grouping with different metrics. Frames that disagree on their
    /// index hierarchy or on the set of keys they cover are rejected
    /// explicitly rather than silently NA-padding or dropping rows, and a
    /// data column whose flattened name collides across frames is rejected
    /// rather than one frame's values silently overwriting another's.
    pub fn merge_multi_index_dataframes(
        dataframes: Vec<MultiIndexDataFrame>,
    ) -> Result<MultiIndexDataFrame> {
        if dataframes.is_empty() {
            return Err(Error::InvalidValue(
                "Cannot merge empty list of DataFrames".to_string(),
            ));
        }

        if dataframes.len() == 1 {
            return dataframes.into_iter().next().ok_or_else(|| {
                Error::InsufficientData("dataframes vec should have one element".to_string())
            });
        }

        let index_hierarchy: Vec<String> = dataframes[0].index_hierarchy().to_vec();
        for (i, mdf) in dataframes.iter().enumerate().skip(1) {
            if mdf.index_hierarchy().to_vec() != index_hierarchy {
                return Err(Error::InvalidValue(format!(
                    "merge_multi_index_dataframes: frame {} has index hierarchy {:?}, expected \
                     {:?} (every frame being merged must share identical index columns, in the \
                     same order)",
                    i,
                    mdf.index_hierarchy(),
                    index_hierarchy
                )));
            }
        }

        if index_hierarchy.is_empty() && dataframes.iter().any(|mdf| mdf.data().row_count() > 1) {
            return Err(Error::InvalidValue(
                "merge_multi_index_dataframes: frames with no index columns can only be merged \
                 when each has at most one row (with no index column, row identity is \
                 ambiguous)"
                    .to_string(),
            ));
        }

        // Row key -> row position, per input frame.
        let mut key_orders: Vec<Vec<Vec<String>>> = Vec::with_capacity(dataframes.len());
        let mut key_lookup: Vec<HashMap<Vec<String>, usize>> = Vec::with_capacity(dataframes.len());

        for (frame_idx, mdf) in dataframes.iter().enumerate() {
            let n = mdf.data().row_count();
            let mut per_level: Vec<Vec<String>> = Vec::with_capacity(index_hierarchy.len());
            for level in &index_hierarchy {
                per_level.push(column_as_strings(mdf.data(), level)?);
            }
            let mut keys: Vec<Vec<String>> = Vec::with_capacity(n);
            let mut lookup: HashMap<Vec<String>, usize> = HashMap::with_capacity(n);
            for row in 0..n {
                let key: Vec<String> = per_level.iter().map(|col| col[row].clone()).collect();
                lookup.insert(key.clone(), row);
                keys.push(key);
            }
            if lookup.len() != n {
                return Err(Error::InvalidValue(format!(
                    "merge_multi_index_dataframes: frame {} has duplicate index keys; each \
                     row's index values must be unique before merging",
                    frame_idx
                )));
            }
            key_orders.push(keys);
            key_lookup.push(lookup);
        }

        let first_key_set: BTreeSet<Vec<String>> = key_orders[0].iter().cloned().collect();
        for (i, keys) in key_orders.iter().enumerate().skip(1) {
            let key_set: BTreeSet<Vec<String>> = keys.iter().cloned().collect();
            if key_set != first_key_set {
                return Err(Error::InvalidValue(format!(
                    "merge_multi_index_dataframes: frame {} covers a different set of index \
                     keys than frame 0; merging requires every frame to cover exactly the same \
                     rows",
                    i
                )));
            }
        }

        let sorted_keys: Vec<Vec<String>> = first_key_set.into_iter().collect();

        let mut merged_data = DataFrame::new();
        for (level_idx, level_name) in index_hierarchy.iter().enumerate() {
            let values: Vec<String> = sorted_keys.iter().map(|k| k[level_idx].clone()).collect();
            merged_data.add_column(
                level_name.clone(),
                Series::new(values, Some(level_name.clone()))?,
            )?;
        }

        let mut merged_column_index: Vec<MultiIndexColumn> = Vec::new();
        for (frame_idx, mdf) in dataframes.iter().enumerate() {
            for column in mdf.column_index() {
                let flat_name = column.display_name();
                if merged_data.contains_column(&flat_name) {
                    return Err(Error::DuplicateColumnName(format!(
                        "merge_multi_index_dataframes: column '{}' appears in more than one \
                         input frame (frame {} is the second occurrence); rename it in one of \
                         the frames before merging",
                        flat_name, frame_idx
                    )));
                }
                let source_values = column_as_strings(mdf.data(), &flat_name)?;
                let lookup = &key_lookup[frame_idx];
                let mut values: Vec<String> = Vec::with_capacity(sorted_keys.len());
                for key in &sorted_keys {
                    let row = *lookup.get(key).ok_or_else(|| {
                        Error::InvalidValue(format!(
                            "merge_multi_index_dataframes: internal key lookup failure for \
                             column '{}'",
                            flat_name
                        ))
                    })?;
                    values.push(source_values[row].clone());
                }
                merged_data.add_column(
                    flat_name.clone(),
                    Series::new(values, Some(flat_name.clone()))?,
                )?;
                merged_column_index.push(column.clone());
            }
        }

        Ok(MultiIndexDataFrame::new(
            merged_data,
            merged_column_index,
            index_hierarchy,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_index_column() {
        let column = MultiIndexColumn::new(
            vec!["Sales".to_string(), "Q1".to_string(), "mean".to_string()],
            vec![
                "metric".to_string(),
                "period".to_string(),
                "function".to_string(),
            ],
            "f64".to_string(),
        );

        assert_eq!(column.display_name(), "Sales.Q1.mean");
        assert_eq!(column.leaf_name(), "mean");
        assert_eq!(column.depth(), 3);
        assert_eq!(column.parent_levels(), &["Sales", "Q1"]);

        assert!(column.belongs_to_parent(&["Sales".to_string()]));
        assert!(column.belongs_to_parent(&["Sales".to_string(), "Q1".to_string()]));
        assert!(!column.belongs_to_parent(&["Revenue".to_string()]));
    }

    #[test]
    fn test_multi_index_dataframe_builder() {
        let mut builder = MultiIndexDataFrameBuilder::new(vec!["region".to_string()]);

        let column_spec = utils::simple_multi_index_column(vec!["sales", "mean"], "f64");
        let data = Series::new(vec!["100.0".to_string()], Some("sales.mean".to_string()))
            .expect("operation should succeed");

        builder
            .add_column(column_spec, data)
            .expect("operation should succeed");

        // Real per-row index key values must be supplied explicitly:
        // nothing about the data columns alone tells the builder what a
        // row's actual key is.
        builder
            .set_index_values(vec![vec!["east".to_string()]])
            .expect("operation should succeed");

        let multi_df = builder.build().expect("operation should succeed");
        assert_eq!(multi_df.column_index().len(), 1);
        assert_eq!(multi_df.index_hierarchy(), &["region"]);

        // The index column carries the real key that was supplied, not the
        // old fabricated "placeholder" string.
        let region_col = multi_df
            .data()
            .get_column::<String>("region")
            .expect("operation should succeed");
        assert_eq!(region_col.values(), &["east".to_string()]);
    }

    #[test]
    fn test_multi_index_dataframe_builder_requires_index_values() {
        let mut builder = MultiIndexDataFrameBuilder::new(vec!["region".to_string()]);

        let column_spec = utils::simple_multi_index_column(vec!["sales", "mean"], "f64");
        let data = Series::new(vec!["100.0".to_string()], Some("sales.mean".to_string()))
            .expect("operation should succeed");
        builder
            .add_column(column_spec, data)
            .expect("operation should succeed");

        // No `set_index_values` call: building must fail loudly instead of
        // fabricating a "placeholder" index column.
        assert!(builder.build().is_err());
    }

    #[test]
    fn test_column_hierarchy_summary() {
        let columns = vec![
            utils::simple_multi_index_column(vec!["sales", "mean"], "f64"),
            utils::simple_multi_index_column(vec!["sales", "sum"], "f64"),
            utils::simple_multi_index_column(vec!["quantity", "count"], "i64"),
        ];

        let data = DataFrame::new();
        let multi_df = MultiIndexDataFrame::new(data, columns, vec!["region".to_string()]);

        let summary = multi_df.column_summary();
        assert_eq!(summary.total_columns, 3);
        assert_eq!(summary.max_depth, 2);
        assert!(summary.levels_summary.contains_key(&2));
    }
}
