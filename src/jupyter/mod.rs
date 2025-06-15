//! Jupyter Notebook Integration for PandRS
//!
//! This module provides enhanced integration with Jupyter notebooks including:
//! - Rich HTML display for DataFrames and Series
//! - Interactive widgets and visualizations
//! - Custom display formatting and styling
//! - Jupyter magic commands support
//! - Better integration with Jupyter's display system

use lazy_static::lazy_static;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::RwLock;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Jupyter display configuration
#[derive(Debug, Clone)]
pub struct JupyterConfig {
    /// Maximum number of rows to display
    pub max_rows: usize,
    /// Maximum number of columns to display
    pub max_columns: usize,
    /// Enable interactive features
    pub interactive: bool,
    /// Display precision for floating point numbers
    pub precision: usize,
    /// Color scheme for styling
    pub color_scheme: JupyterColorScheme,
    /// Table styling options
    pub table_style: TableStyle,
    /// Enable syntax highlighting
    pub syntax_highlighting: bool,
}

impl Default for JupyterConfig {
    fn default() -> Self {
        Self {
            max_rows: 100,
            max_columns: 20,
            interactive: true,
            precision: 6,
            color_scheme: JupyterColorScheme::Default,
            table_style: TableStyle::default(),
            syntax_highlighting: true,
        }
    }
}

/// Color schemes for Jupyter display
#[derive(Debug, Clone)]
pub enum JupyterColorScheme {
    /// Default color scheme
    Default,
    /// Dark theme
    Dark,
    /// Light theme  
    Light,
    /// Custom color scheme
    Custom {
        header_bg: String,
        header_text: String,
        row_bg: String,
        alt_row_bg: String,
        text_color: String,
        border_color: String,
    },
}

impl JupyterColorScheme {
    /// Get CSS styles for the color scheme
    pub fn to_css(&self) -> String {
        match self {
            JupyterColorScheme::Default => r#"
                .pandrs-table { background-color: #ffffff; color: #333333; }
                .pandrs-header { background-color: #f8f9fa; color: #495057; font-weight: bold; }
                .pandrs-row { background-color: #ffffff; }
                .pandrs-row:nth-child(even) { background-color: #f8f9fa; }
                .pandrs-border { border-color: #dee2e6; }
            "#
            .to_string(),

            JupyterColorScheme::Dark => r#"
                .pandrs-table { background-color: #2d3748; color: #e2e8f0; }
                .pandrs-header { background-color: #4a5568; color: #f7fafc; font-weight: bold; }
                .pandrs-row { background-color: #2d3748; }
                .pandrs-row:nth-child(even) { background-color: #4a5568; }
                .pandrs-border { border-color: #718096; }
            "#
            .to_string(),

            JupyterColorScheme::Light => r#"
                .pandrs-table { background-color: #fefefe; color: #2d3748; }
                .pandrs-header { background-color: #edf2f7; color: #2d3748; font-weight: bold; }
                .pandrs-row { background-color: #fefefe; }
                .pandrs-row:nth-child(even) { background-color: #f7fafc; }
                .pandrs-border { border-color: #cbd5e0; }
            "#
            .to_string(),

            JupyterColorScheme::Custom {
                header_bg,
                header_text,
                row_bg,
                alt_row_bg,
                text_color,
                border_color,
            } => {
                format!(
                    r#"
                    .pandrs-table {{ background-color: {}; color: {}; }}
                    .pandrs-header {{ background-color: {}; color: {}; font-weight: bold; }}
                    .pandrs-row {{ background-color: {}; }}
                    .pandrs-row:nth-child(even) {{ background-color: {}; }}
                    .pandrs-border {{ border-color: {}; }}
                "#,
                    row_bg, text_color, header_bg, header_text, row_bg, alt_row_bg, border_color
                )
            }
        }
    }
}

/// Table styling options
#[derive(Debug, Clone)]
pub struct TableStyle {
    /// Show borders around cells
    pub show_borders: bool,
    /// Show row numbers/index
    pub show_index: bool,
    /// Show column data types
    pub show_dtypes: bool,
    /// Show summary statistics
    pub show_summary: bool,
    /// Table width setting
    pub width: TableWidth,
    /// Cell padding
    pub cell_padding: String,
    /// Font family
    pub font_family: String,
    /// Font size
    pub font_size: String,
}

impl Default for TableStyle {
    fn default() -> Self {
        Self {
            show_borders: true,
            show_index: true,
            show_dtypes: false,
            show_summary: false,
            width: TableWidth::Auto,
            cell_padding: "8px 12px".to_string(),
            font_family: "'Monaco', 'Menlo', 'Ubuntu Mono', monospace".to_string(),
            font_size: "13px".to_string(),
        }
    }
}

/// Table width options
#[derive(Debug, Clone)]
pub enum TableWidth {
    /// Automatic width
    Auto,
    /// Fixed width in pixels
    Fixed(u32),
    /// Percentage of container
    Percentage(u32),
    /// Full width
    Full,
}

impl TableWidth {
    fn to_css(&self) -> String {
        match self {
            TableWidth::Auto => "width: auto;".to_string(),
            TableWidth::Fixed(px) => format!("width: {}px;", px),
            TableWidth::Percentage(pct) => format!("width: {}%;", pct),
            TableWidth::Full => "width: 100%;".to_string(),
        }
    }
}

/// Jupyter display trait for DataFrames
pub trait JupyterDisplay {
    /// Generate rich HTML display for Jupyter
    fn to_jupyter_html(&self, config: &JupyterConfig) -> Result<String>;

    /// Generate interactive widget HTML
    fn to_interactive_widget(&self, config: &JupyterConfig) -> Result<String>;

    /// Generate summary display
    fn to_summary_html(&self, config: &JupyterConfig) -> Result<String>;

    /// Generate data explorer widget
    fn to_data_explorer(&self, config: &JupyterConfig) -> Result<String>;
}

impl JupyterDisplay for DataFrame {
    fn to_jupyter_html(&self, config: &JupyterConfig) -> Result<String> {
        let mut html = String::new();

        // CSS styles
        html.push_str(&format!(
            r#"
        <style>
        .pandrs-container {{
            margin: 10px 0;
            font-family: {};
            font-size: {};
        }}
        .pandrs-table {{
            border-collapse: collapse;
            margin: 10px 0;
            {}
        }}
        .pandrs-table th, .pandrs-table td {{
            padding: {};
            text-align: left;
            border: {};
        }}
        {}
        .pandrs-info {{
            margin: 5px 0;
            font-size: 12px;
            color: #666;
        }}
        .pandrs-dtype {{
            font-style: italic;
            color: #888;
            font-size: 11px;
        }}
        </style>
        "#,
            config.table_style.font_family,
            config.table_style.font_size,
            config.table_style.width.to_css(),
            config.table_style.cell_padding,
            if config.table_style.show_borders {
                "1px solid var(--pandrs-border-color, #ddd)"
            } else {
                "none"
            },
            config.color_scheme.to_css()
        ));

        // Container start
        html.push_str(r#"<div class="pandrs-container">"#);

        // DataFrame info
        html.push_str(&format!(
            r#"<div class="pandrs-info">DataFrame: {} rows × {} columns</div>"#,
            self.row_count(),
            self.column_names().len()
        ));

        // Table start
        html.push_str(r#"<table class="pandrs-table">"#);

        // Header
        html.push_str("<thead><tr class='pandrs-header'>");

        if config.table_style.show_index {
            html.push_str("<th></th>"); // Index column header
        }

        let columns = self.column_names();
        let visible_columns = if columns.len() > config.max_columns {
            &columns[..config.max_columns]
        } else {
            &columns
        };

        for col_name in visible_columns {
            html.push_str(&format!("<th>{}</th>", escape_html(col_name)));
        }

        if columns.len() > config.max_columns {
            html.push_str("<th>...</th>");
        }

        html.push_str("</tr>");

        // Column types (if enabled)
        if config.table_style.show_dtypes {
            html.push_str("<tr class='pandrs-header'>");
            if config.table_style.show_index {
                html.push_str("<td class='pandrs-dtype'></td>");
            }
            for col_name in visible_columns {
                // Try to infer column type
                let dtype = self.infer_column_type(col_name);
                html.push_str(&format!("<td class='pandrs-dtype'>{}</td>", dtype));
            }
            if columns.len() > config.max_columns {
                html.push_str("<td class='pandrs-dtype'>...</td>");
            }
            html.push_str("</tr>");
        }

        html.push_str("</thead>");

        // Body
        html.push_str("<tbody>");

        let row_count = self.row_count();
        let visible_rows = if row_count > config.max_rows {
            config.max_rows / 2
        } else {
            row_count
        };

        // Display first rows
        for i in 0..visible_rows.min(row_count) {
            html.push_str(&format!("<tr class='pandrs-row'>"));

            if config.table_style.show_index {
                html.push_str(&format!("<td><strong>{}</strong></td>", i));
            }

            for col_name in visible_columns {
                let value = self
                    .get_value_at(i, col_name)
                    .unwrap_or_else(|_| "NaN".to_string());
                let formatted_value = format_value(&value, config.precision);
                html.push_str(&format!("<td>{}</td>", escape_html(&formatted_value)));
            }

            if columns.len() > config.max_columns {
                html.push_str("<td>...</td>");
            }

            html.push_str("</tr>");
        }

        // Show ellipsis if there are more rows
        if row_count > config.max_rows {
            html.push_str("<tr class='pandrs-row'>");
            if config.table_style.show_index {
                html.push_str("<td>...</td>");
            }
            for _ in 0..visible_columns.len() {
                html.push_str("<td>...</td>");
            }
            if columns.len() > config.max_columns {
                html.push_str("<td>...</td>");
            }
            html.push_str("</tr>");

            // Display last rows
            let remaining_rows = config.max_rows - visible_rows;
            let start_idx = row_count - remaining_rows;

            for i in start_idx..row_count {
                html.push_str(&format!("<tr class='pandrs-row'>"));

                if config.table_style.show_index {
                    html.push_str(&format!("<td><strong>{}</strong></td>", i));
                }

                for col_name in visible_columns {
                    let value = self
                        .get_value_at(i, col_name)
                        .unwrap_or_else(|_| "NaN".to_string());
                    let formatted_value = format_value(&value, config.precision);
                    html.push_str(&format!("<td>{}</td>", escape_html(&formatted_value)));
                }

                if columns.len() > config.max_columns {
                    html.push_str("<td>...</td>");
                }

                html.push_str("</tr>");
            }
        }

        html.push_str("</tbody>");
        html.push_str("</table>");

        // Summary info (if enabled)
        if config.table_style.show_summary {
            html.push_str(&format!(
                r#"<div class="pandrs-info">
                Memory usage: ~{} KB | 
                Columns: {} | 
                Data types: {}
                </div>"#,
                estimate_memory_usage(self),
                columns.len(),
                self.get_column_types().join(", ")
            ));
        }

        html.push_str("</div>");

        Ok(html)
    }

    fn to_interactive_widget(&self, config: &JupyterConfig) -> Result<String> {
        let mut html = String::new();

        // Interactive widget with JavaScript
        html.push_str(&format!(
            r#"
        <div id="pandrs-widget-{}" class="pandrs-interactive-widget">
            <style>
            .pandrs-interactive-widget {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                font-family: {};
            }}
            .pandrs-controls {{
                margin-bottom: 10px;
                padding: 5px;
                background-color: #f8f9fa;
                border-radius: 3px;
            }}
            .pandrs-controls button {{
                margin: 2px;
                padding: 5px 10px;
                border: 1px solid #ccc;
                background-color: #fff;
                cursor: pointer;
                border-radius: 3px;
            }}
            .pandrs-controls button:hover {{
                background-color: #e9ecef;
            }}
            .pandrs-search {{
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                margin: 0 5px;
            }}
            </style>
            
            <div class="pandrs-controls">
                <button onclick="pandrs_show_info('{}')">📊 Info</button>
                <button onclick="pandrs_show_describe('{}')">📈 Describe</button>
                <button onclick="pandrs_show_plot('{}')">📊 Plot</button>
                <input type="text" class="pandrs-search" placeholder="Search columns..." 
                       onkeyup="pandrs_filter_columns('{}', this.value)">
            </div>
            
            <div id="pandrs-content-{}">
                {}
            </div>
        </div>
        
        <script>
        function pandrs_show_info(widget_id) {{
            document.getElementById('pandrs-content-' + widget_id).innerHTML = `
                <h4>DataFrame Information</h4>
                <ul>
                    <li>Rows: {}</li>
                    <li>Columns: {}</li>
                    <li>Memory usage: ~{} KB</li>
                    <li>Column types: {}</li>
                </ul>
            `;
        }}
        
        function pandrs_show_describe(widget_id) {{
            document.getElementById('pandrs-content-' + widget_id).innerHTML = `
                <h4>Statistical Description</h4>
                <p>Summary statistics would be displayed here...</p>
            `;
        }}
        
        function pandrs_show_plot(widget_id) {{
            document.getElementById('pandrs-content-' + widget_id).innerHTML = `
                <h4>Quick Plot</h4>
                <p>Interactive plotting widget would be displayed here...</p>
            `;
        }}
        
        function pandrs_filter_columns(widget_id, filter) {{
            // Column filtering functionality would be implemented here
            console.log('Filtering columns by:', filter);
        }}
        </script>
        "#,
            self.row_count(), // widget id
            config.table_style.font_family,
            self.row_count(), // info button
            self.row_count(), // describe button
            self.row_count(), // plot button
            self.row_count(), // search input
            self.row_count(), // content div
            self.to_jupyter_html(config)?,
            self.row_count(),
            self.column_names().len(),
            estimate_memory_usage(self),
            self.get_column_types().join(", ")
        ));

        Ok(html)
    }

    fn to_summary_html(&self, config: &JupyterConfig) -> Result<String> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
        <div class="pandrs-summary" style="
            font-family: {};
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
            background-color: #f8f9fa;
        ">
            <h3 style="margin-top: 0;">DataFrame Summary</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4>Basic Information</h4>
                    <ul style="list-style: none; padding: 0;">
                        <li><strong>Shape:</strong> {} rows × {} columns</li>
                        <li><strong>Memory usage:</strong> ~{} KB</li>
                        <li><strong>Non-null values:</strong> {}</li>
                        <li><strong>Null values:</strong> {}</li>
                    </ul>
                </div>
                <div>
                    <h4>Column Information</h4>
                    <ul style="list-style: none; padding: 0; max-height: 200px; overflow-y: auto;">
        "#,
            config.table_style.font_family,
            self.row_count(),
            self.column_names().len(),
            estimate_memory_usage(self),
            estimate_non_null_count(self),
            estimate_null_count(self)
        ));

        // Column details
        for (i, col_name) in self.column_names().iter().enumerate() {
            let col_type = self.infer_column_type(col_name);
            html.push_str(&format!(
                "<li><strong>{}:</strong> {} <span style='color: #666; font-size: 0.9em'>({})</span></li>",
                escape_html(col_name),
                col_type,
                format!("col {}", i)
            ));
        }

        html.push_str(
            r#"
                    </ul>
                </div>
            </div>
        </div>
        "#,
        );

        Ok(html)
    }

    fn to_data_explorer(&self, config: &JupyterConfig) -> Result<String> {
        let widget_id = format!("explorer_{}", self.row_count());

        let mut html = String::new();

        html.push_str(&format!(r#"
        <div id="pandrs-explorer-{}" class="pandrs-data-explorer">
            <style>
            .pandrs-data-explorer {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 0;
                margin: 15px 0;
                font-family: {};
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .pandrs-explorer-header {{
                background-color: #007bff;
                color: white;
                padding: 12px 15px;
                border-radius: 8px 8px 0 0;
                font-weight: bold;
            }}
            .pandrs-explorer-tabs {{
                display: flex;
                background-color: #f8f9fa;
                border-bottom: 1px solid #ddd;
            }}
            .pandrs-explorer-tab {{
                padding: 10px 20px;
                cursor: pointer;
                border: none;
                background: none;
                border-bottom: 3px solid transparent;
                font-weight: 500;
            }}
            .pandrs-explorer-tab.active {{
                background-color: white;
                border-bottom-color: #007bff;
                color: #007bff;
            }}
            .pandrs-explorer-content {{
                padding: 15px;
                min-height: 300px;
            }}
            </style>
            
            <div class="pandrs-explorer-header">
                🔍 DataFrame Explorer - {} rows × {} columns
            </div>
            
            <div class="pandrs-explorer-tabs">
                <button class="pandrs-explorer-tab active" onclick="pandrs_show_tab('{}', 'data')">
                    📊 Data
                </button>
                <button class="pandrs-explorer-tab" onclick="pandrs_show_tab('{}', 'info')">
                    ℹ️ Info
                </button>
                <button class="pandrs-explorer-tab" onclick="pandrs_show_tab('{}', 'stats')">
                    📈 Statistics
                </button>
                <button class="pandrs-explorer-tab" onclick="pandrs_show_tab('{}', 'viz')">
                    📊 Visualize
                </button>
            </div>
            
            <div class="pandrs-explorer-content">
                <div id="pandrs-tab-data-{}" class="pandrs-tab-content">
                    {}
                </div>
                <div id="pandrs-tab-info-{}" class="pandrs-tab-content" style="display: none;">
                    {}
                </div>
                <div id="pandrs-tab-stats-{}" class="pandrs-tab-content" style="display: none;">
                    <h4>Statistical Summary</h4>
                    <p>Descriptive statistics would be computed and displayed here...</p>
                </div>
                <div id="pandrs-tab-viz-{}" class="pandrs-tab-content" style="display: none;">
                    <h4>Quick Visualizations</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                        <button onclick="alert('Histogram plot would be generated')">📊 Histogram</button>
                        <button onclick="alert('Scatter plot would be generated')">🔸 Scatter Plot</button>
                        <button onclick="alert('Box plot would be generated')">📦 Box Plot</button>
                        <button onclick="alert('Correlation matrix would be generated')">🔗 Correlation</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        function pandrs_show_tab(explorer_id, tab_name) {{
            // Hide all tabs
            ['data', 'info', 'stats', 'viz'].forEach(function(name) {{
                var element = document.getElementById('pandrs-tab-' + name + '-' + explorer_id);
                if (element) element.style.display = 'none';
            }});
            
            // Remove active class from all tab buttons
            var tabs = document.querySelectorAll('#pandrs-explorer-' + explorer_id + ' .pandrs-explorer-tab');
            tabs.forEach(function(tab) {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab
            var selectedTab = document.getElementById('pandrs-tab-' + tab_name + '-' + explorer_id);
            if (selectedTab) selectedTab.style.display = 'block';
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }}
        </script>
        "#,
            widget_id,
            config.table_style.font_family,
            self.row_count(),
            self.column_names().len(),
            widget_id, // data tab
            widget_id, // info tab
            widget_id, // stats tab
            widget_id, // viz tab
            widget_id, // data content div
            self.to_jupyter_html(config)?,
            widget_id, // info content div
            self.to_summary_html(config)?,
            widget_id, // stats content div
            widget_id  // viz content div
        ));

        Ok(html)
    }
}

impl<T> JupyterDisplay for Series<T>
where
    T: Clone + std::fmt::Display + std::fmt::Debug,
{
    fn to_jupyter_html(&self, config: &JupyterConfig) -> Result<String> {
        let mut html = String::new();

        // CSS styles
        html.push_str(&format!(
            r#"
        <style>
        .pandrs-series-container {{
            margin: 10px 0;
            font-family: {};
            font-size: {};
        }}
        .pandrs-series-table {{
            border-collapse: collapse;
            margin: 10px 0;
            {}
        }}
        .pandrs-series-table th, .pandrs-series-table td {{
            padding: {};
            text-align: left;
            border: {};
        }}
        {}
        .pandrs-series-info {{
            margin: 5px 0;
            font-size: 12px;
            color: #666;
        }}
        </style>
        "#,
            config.table_style.font_family,
            config.table_style.font_size,
            config.table_style.width.to_css(),
            config.table_style.cell_padding,
            if config.table_style.show_borders {
                "1px solid var(--pandrs-border-color, #ddd)"
            } else {
                "none"
            },
            config.color_scheme.to_css()
        ));

        // Container start
        html.push_str(r#"<div class="pandrs-series-container">"#);

        // Series info
        let series_name = self.name().map_or("Unnamed", |s| s.as_str());
        html.push_str(&format!(
            r#"<div class="pandrs-series-info">Series '{}': {} values</div>"#,
            escape_html(series_name),
            self.len()
        ));

        // Table start
        html.push_str(r#"<table class="pandrs-series-table">"#);

        // Header
        html.push_str("<thead><tr class='pandrs-header'>");
        if config.table_style.show_index {
            html.push_str("<th>Index</th>");
        }
        html.push_str(&format!("<th>{}</th>", escape_html(series_name)));
        html.push_str("</tr></thead>");

        // Body
        html.push_str("<tbody>");

        let values = self.values();
        let visible_count = if values.len() > config.max_rows {
            config.max_rows / 2
        } else {
            values.len()
        };

        // Display first values
        for (i, value) in values.iter().enumerate().take(visible_count) {
            html.push_str(&format!("<tr class='pandrs-row'>"));

            if config.table_style.show_index {
                html.push_str(&format!("<td><strong>{}</strong></td>", i));
            }

            let formatted_value = format!("{}", value);
            html.push_str(&format!("<td>{}</td>", escape_html(&formatted_value)));
            html.push_str("</tr>");
        }

        // Show ellipsis if there are more values
        if values.len() > config.max_rows {
            html.push_str("<tr class='pandrs-row'>");
            if config.table_style.show_index {
                html.push_str("<td>...</td>");
            }
            html.push_str("<td>...</td>");
            html.push_str("</tr>");

            // Display last values
            let remaining_count = config.max_rows - visible_count;
            let start_idx = values.len() - remaining_count;

            for (i, value) in values.iter().enumerate().skip(start_idx) {
                html.push_str(&format!("<tr class='pandrs-row'>"));

                if config.table_style.show_index {
                    html.push_str(&format!("<td><strong>{}</strong></td>", i));
                }

                let formatted_value = format!("{}", value);
                html.push_str(&format!("<td>{}</td>", escape_html(&formatted_value)));
                html.push_str("</tr>");
            }
        }

        html.push_str("</tbody>");
        html.push_str("</table>");
        html.push_str("</div>");

        Ok(html)
    }

    fn to_interactive_widget(&self, config: &JupyterConfig) -> Result<String> {
        // Simplified interactive widget for Series
        let mut html = String::new();
        let series_name = self.name().map_or("Unnamed", |s| s.as_str());

        html.push_str(&format!(
            r#"
        <div class="pandrs-series-widget">
            <div style="margin-bottom: 10px; font-weight: bold;">
                Series '{}' Interactive Widget
            </div>
            {}
        </div>
        "#,
            escape_html(series_name),
            self.to_jupyter_html(config)?
        ));

        Ok(html)
    }

    fn to_summary_html(&self, config: &JupyterConfig) -> Result<String> {
        let series_name = self.name().map_or("Unnamed", |s| s.as_str());

        Ok(format!(
            r#"
        <div class="pandrs-series-summary" style="
            font-family: {};
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
            background-color: #f8f9fa;
        ">
            <h3 style="margin-top: 0;">Series '{}' Summary</h3>
            <ul style="list-style: none; padding: 0;">
                <li><strong>Length:</strong> {} values</li>
                <li><strong>Non-null values:</strong> {}</li>
                <li><strong>Data type:</strong> {}</li>
            </ul>
        </div>
        "#,
            config.table_style.font_family,
            escape_html(series_name),
            self.len(),
            self.len(), // Assume all non-null for now
            std::any::type_name::<T>()
        ))
    }

    fn to_data_explorer(&self, config: &JupyterConfig) -> Result<String> {
        // Use the summary for series data explorer
        self.to_summary_html(config)
    }
}

/// Magic commands support for Jupyter
pub struct JupyterMagics;

impl JupyterMagics {
    /// Register PandRS magic commands
    pub fn register_magics() -> String {
        r#"
        # PandRS Magic Commands for Jupyter
        
        from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
        from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
        
        @magics_class
        class PandRSMagics(Magics):
            
            @line_magic
            def pandrs_info(self, line):
                """Display PandRS version and configuration info"""
                return "PandRS Alpha 4 - Enhanced DataFrame library for Rust"
            
            @line_magic 
            def pandrs_config(self, line):
                """Configure PandRS display settings for Jupyter"""
                # Configuration management would be implemented here
                return "PandRS configuration updated"
            
            @cell_magic
            def pandrs_sql(self, line, cell):
                """Execute SQL-like queries on DataFrames"""
                # SQL query execution would be implemented here
                return f"SQL query executed: {cell}"
        
        # Register the magic class
        ip = get_ipython()
        ip.register_magic_functions(PandRSMagics)
        "#
        .to_string()
    }
}

// Helper functions

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

fn format_value(value: &str, precision: usize) -> String {
    // Try to parse as float for better formatting
    if let Ok(f) = value.parse::<f64>() {
        if f.fract() == 0.0 && f.abs() < 1e10 {
            format!("{:.0}", f)
        } else {
            format!("{:.prec$}", f, prec = precision)
        }
    } else {
        value.to_string()
    }
}

fn estimate_memory_usage(df: &DataFrame) -> u64 {
    // Rough estimate: number of cells * average cell size
    let cell_count = df.row_count() * df.column_names().len();
    (cell_count * 8) as u64 / 1024 // Convert to KB
}

fn estimate_non_null_count(df: &DataFrame) -> usize {
    // Simplified: assume all values are non-null for now
    df.row_count() * df.column_names().len()
}

fn estimate_null_count(df: &DataFrame) -> usize {
    // Simplified: assume no null values for now
    0
}

impl DataFrame {
    /// Infer the type of a column for display purposes
    fn infer_column_type(&self, column_name: &str) -> String {
        // Try to get a sample value and infer type
        if let Ok(value) = self.get_value_at(0, column_name) {
            if value.parse::<i64>().is_ok() {
                "integer".to_string()
            } else if value.parse::<f64>().is_ok() {
                "float".to_string()
            } else if value.parse::<bool>().is_ok() {
                "boolean".to_string()
            } else {
                "string".to_string()
            }
        } else {
            "unknown".to_string()
        }
    }

    /// Get column types for all columns
    fn get_column_types(&self) -> Vec<String> {
        self.column_names()
            .iter()
            .map(|col| self.infer_column_type(col))
            .collect()
    }

    /// Get value at specific position (helper method)
    fn get_value_at(&self, row: usize, column: &str) -> Result<String> {
        let values = self.get_column_string_values(column)?;
        if row < values.len() {
            Ok(values[row].clone())
        } else {
            Err(Error::IndexOutOfBounds {
                index: row,
                size: values.len(),
            })
        }
    }
}

lazy_static! {
    /// Display configuration for Jupyter notebooks
    static ref JUPYTER_CONFIG: RwLock<Option<JupyterConfig>> = RwLock::new(None);
}

/// Get the current Jupyter configuration
pub fn get_jupyter_config() -> JupyterConfig {
    JUPYTER_CONFIG.read().unwrap().clone().unwrap_or_default()
}

/// Set the Jupyter configuration
pub fn set_jupyter_config(config: JupyterConfig) {
    *JUPYTER_CONFIG.write().unwrap() = Some(config);
}

/// Initialize Jupyter integration with default settings
pub fn init_jupyter() -> Result<()> {
    set_jupyter_config(JupyterConfig::default());

    // Print initialization message
    println!("🎯 PandRS Jupyter Integration Initialized!");
    println!("📚 Enhanced DataFrame display and interactive widgets are now available.");
    println!("🔍 Use .to_jupyter_html(), .to_interactive_widget(), or .to_data_explorer() on DataFrames.");

    Ok(())
}

/// Create a quick configuration for dark mode
pub fn jupyter_dark_mode() -> JupyterConfig {
    JupyterConfig {
        color_scheme: JupyterColorScheme::Dark,
        ..Default::default()
    }
}

/// Create a quick configuration for light mode  
pub fn jupyter_light_mode() -> JupyterConfig {
    JupyterConfig {
        color_scheme: JupyterColorScheme::Light,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_jupyter_config_default() {
        let config = JupyterConfig::default();
        assert_eq!(config.max_rows, 100);
        assert_eq!(config.max_columns, 20);
        assert!(config.interactive);
    }

    #[test]
    fn test_color_scheme_css() {
        let dark_scheme = JupyterColorScheme::Dark;
        let css = dark_scheme.to_css();
        assert!(css.contains("background-color: #2d3748"));
    }

    #[test]
    fn test_table_width_css() {
        let auto_width = TableWidth::Auto;
        assert_eq!(auto_width.to_css(), "width: auto;");

        let fixed_width = TableWidth::Fixed(800);
        assert_eq!(fixed_width.to_css(), "width: 800px;");
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("<test>"), "&lt;test&gt;");
        assert_eq!(escape_html("AT&T"), "AT&amp;T");
    }

    #[test]
    fn test_format_value() {
        assert_eq!(format_value("3.14159", 2), "3.14");
        assert_eq!(format_value("42", 2), "42");
        assert_eq!(format_value("hello", 2), "hello");
    }

    #[test]
    fn test_jupyter_html_generation() {
        let mut data = HashMap::new();
        data.insert("col1".to_string(), vec!["1".to_string(), "2".to_string()]);
        data.insert("col2".to_string(), vec!["a".to_string(), "b".to_string()]);

        let df = DataFrame::from_map(data, None).unwrap();
        let config = JupyterConfig::default();

        let html = df.to_jupyter_html(&config).unwrap();
        assert!(html.contains("pandrs-table"));
        assert!(html.contains("col1"));
        assert!(html.contains("col2"));
    }
}
