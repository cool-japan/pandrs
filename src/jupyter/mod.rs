//! Jupyter-friendly HTML rendering for PandRS
//!
//! This module generates HTML fragments intended for display in Jupyter
//! notebooks, including:
//! - Rich HTML tables for DataFrames and Series (real per-column type
//!   inference, real null counts, and CSS-injection-safe styling)
//! - An interactive widget and a data-explorer view backed by genuinely
//!   computed statistics ([`DataFrame::describe_to_json`],
//!   `describe_table_html`) and real charts (via
//!   [`crate::vis::svg::dataframe_ext::SvgVisualize`])
//! - An illustrative Python magic-command template ([`JupyterMagics`])
//!
//! What this module does **not** do: it has no MIME-bundle/kernel-level
//! integration and cannot execute Python-side magics — PandRS is a Rust
//! library with no bridge back into a running IPython kernel. Producing
//! the HTML above and getting it in front of a notebook cell (e.g. via a
//! `_repr_html_`-style hook) is the caller's responsibility, typically a
//! separate Python binding layer.

use lazy_static::lazy_static;
use std::sync::RwLock;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;
use crate::{read_lock_safe, write_lock_safe};

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
                // These are user-supplied strings interpolated unescaped
                // into a raw <style> block, so they are sanitized the
                // same way as JupyterConfig's other CSS-value fields
                // (see `sanitize_css_value`) to prevent a value like
                // `red</style><script>...` from breaking out of the
                // style element.
                let header_bg = sanitize_css_value(header_bg);
                let header_text = sanitize_css_value(header_text);
                let row_bg = sanitize_css_value(row_bg);
                let alt_row_bg = sanitize_css_value(alt_row_bg);
                let text_color = sanitize_css_value(text_color);
                let border_color = sanitize_css_value(border_color);
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
            sanitize_css_value(&config.table_style.font_family),
            sanitize_css_value(&config.table_style.font_size),
            config.table_style.width.to_css(),
            sanitize_css_value(&config.table_style.cell_padding),
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
                Memory usage: ~{} |
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

        // A process-wide counter, not row_count(), so two widgets built
        // from same-shaped DataFrames never collide on DOM id (see
        // `next_widget_id`).
        let widget_id = next_widget_id();

        // Every panel is rendered with real data up front; the buttons
        // only toggle which pre-rendered panel is visible (the same
        // pattern `to_data_explorer` already uses), instead of asking
        // client-side JS — which has no access to the DataFrame — to
        // fabricate "Describe"/"Plot" content on click.
        let data_panel = self.to_jupyter_html(config)?;
        let info_panel = format!(
            "<h4>DataFrame Information</h4><ul><li>Rows: {}</li><li>Columns: {}</li>\
             <li>Memory usage: ~{}</li><li>Column types: {}</li></ul>",
            self.row_count(),
            self.column_names().len(),
            estimate_memory_usage(self),
            escape_html(&self.get_column_types().join(", ")),
        );
        let describe_panel = describe_table_html(self);
        let plot_panel = quick_viz_html(self);

        html.push_str(&format!(
            r#"
        <div id="pandrs-widget-{widget_id}" class="pandrs-interactive-widget">
            <style>
            .pandrs-interactive-widget {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                font-family: {font_family};
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
                <button onclick="pandrs_show_panel({widget_id}, 'data')">📊 Data</button>
                <button onclick="pandrs_show_panel({widget_id}, 'info')">ℹ️ Info</button>
                <button onclick="pandrs_show_panel({widget_id}, 'describe')">📈 Describe</button>
                <button onclick="pandrs_show_panel({widget_id}, 'plot')">📊 Plot</button>
                <input type="text" class="pandrs-search" placeholder="Search columns..."
                       onkeyup="pandrs_filter_columns({widget_id}, this.value)">
            </div>

            <div id="pandrs-panel-data-{widget_id}" class="pandrs-widget-panel">{data_panel}</div>
            <div id="pandrs-panel-info-{widget_id}" class="pandrs-widget-panel" style="display:none;">{info_panel}</div>
            <div id="pandrs-panel-describe-{widget_id}" class="pandrs-widget-panel" style="display:none;">{describe_panel}</div>
            <div id="pandrs-panel-plot-{widget_id}" class="pandrs-widget-panel" style="display:none;">{plot_panel}</div>
        </div>

        <script>
        function pandrs_show_panel(widget_id, panel) {{
            ['data', 'info', 'describe', 'plot'].forEach(function(name) {{
                var el = document.getElementById('pandrs-panel-' + name + '-' + widget_id);
                if (el) {{
                    el.style.display = (name === panel) ? 'block' : 'none';
                }}
            }});
        }}

        function pandrs_filter_columns(widget_id, filter) {{
            // Column filtering functionality would be implemented here
            console.log('Filtering columns by:', filter);
        }}
        </script>
        "#,
            widget_id = widget_id,
            font_family = sanitize_css_value(&config.table_style.font_family),
            data_panel = data_panel,
            info_panel = info_panel,
            describe_panel = describe_panel,
            plot_panel = plot_panel,
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
                        <li><strong>Memory usage:</strong> ~{}</li>
                        <li><strong>Non-null values:</strong> {}</li>
                        <li><strong>Null values:</strong> {}</li>
                    </ul>
                </div>
                <div>
                    <h4>Column Information</h4>
                    <ul style="list-style: none; padding: 0; max-height: 200px; overflow-y: auto;">
        "#,
            sanitize_css_value(&config.table_style.font_family),
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
                "<li><strong>{}:</strong> {} <span style='color: #666; font-size: 0.9em'>(col {})</span></li>",
                escape_html(col_name),
                col_type,
                i
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
        // A process-wide counter, not row_count(), so two explorers built
        // from same-shaped DataFrames never collide on DOM id (see
        // `next_widget_id`).
        let widget_id = next_widget_id();

        let mut html = String::new();
        let data_tab = self.to_jupyter_html(config)?;
        let info_tab = self.to_summary_html(config)?;
        let stats_tab = describe_table_html(self);
        let viz_tab = quick_viz_html(self);

        html.push_str(&format!(r#"
        <div id="pandrs-explorer-{widget_id}" class="pandrs-data-explorer">
            <style>
            .pandrs-data-explorer {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 0;
                margin: 15px 0;
                font-family: {font_family};
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
                🔍 DataFrame Explorer - {rows} rows × {cols} columns
            </div>

            <div class="pandrs-explorer-tabs">
                <button class="pandrs-explorer-tab active" onclick="pandrs_show_tab(event, {widget_id}, 'data')">
                    📊 Data
                </button>
                <button class="pandrs-explorer-tab" onclick="pandrs_show_tab(event, {widget_id}, 'info')">
                    ℹ️ Info
                </button>
                <button class="pandrs-explorer-tab" onclick="pandrs_show_tab(event, {widget_id}, 'stats')">
                    📈 Statistics
                </button>
                <button class="pandrs-explorer-tab" onclick="pandrs_show_tab(event, {widget_id}, 'viz')">
                    📊 Visualize
                </button>
            </div>

            <div class="pandrs-explorer-content">
                <div id="pandrs-tab-data-{widget_id}" class="pandrs-tab-content">
                    {data_tab}
                </div>
                <div id="pandrs-tab-info-{widget_id}" class="pandrs-tab-content" style="display: none;">
                    {info_tab}
                </div>
                <div id="pandrs-tab-stats-{widget_id}" class="pandrs-tab-content" style="display: none;">
                    <h4>Statistical Summary</h4>
                    {stats_tab}
                </div>
                <div id="pandrs-tab-viz-{widget_id}" class="pandrs-tab-content" style="display: none;">
                    <h4>Quick Visualizations</h4>
                    {viz_tab}
                </div>
            </div>
        </div>

        <script>
        function pandrs_show_tab(evt, explorer_id, tab_name) {{
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

            // Add active class to the clicked button. The triggering
            // event is passed explicitly (`evt`) rather than read from
            // the global `window.event`, which is a deprecated,
            // non-standard, Chrome/IE-only fallback that recent browsers
            // are dropping.
            if (evt && evt.target) {{
                evt.target.classList.add('active');
            }}
        }}
        </script>
        "#,
            widget_id = widget_id,
            font_family = sanitize_css_value(&config.table_style.font_family),
            rows = self.row_count(),
            cols = self.column_names().len(),
            data_tab = data_tab,
            info_tab = info_tab,
            stats_tab = stats_tab,
            viz_tab = viz_tab,
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
            sanitize_css_value(&config.table_style.font_family),
            sanitize_css_value(&config.table_style.font_size),
            config.table_style.width.to_css(),
            sanitize_css_value(&config.table_style.cell_padding),
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
            sanitize_css_value(&config.table_style.font_family),
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

/// Generator for an illustrative Jupyter/IPython magic-command template.
///
/// PandRS is a Rust library with no Python interpreter embedded and no
/// live bridge back into a running kernel's namespace, so nothing in
/// this crate can genuinely *execute* Python-side IPython magics — that
/// would require a separate binding layer (e.g. PyO3) wiring these
/// method bodies to a real PandRS instance. [`Self::register_magics`] emits
/// Python source text meant as a starting-point template for such a
/// binding to adapt, not a working integration by itself.
pub struct JupyterMagics;

impl JupyterMagics {
    /// Emit an illustrative Python magic-command template.
    ///
    /// The generated `pandrs_sql`/`pandrs_config` bodies raise
    /// `NotImplementedError` rather than claiming to have executed a
    /// query or applied a configuration change: printing back the input
    /// cell as `"SQL query executed: {cell}"` without running anything
    /// against real data would tell a user their query succeeded when
    /// nothing happened. `pandrs_info` is left returning a real,
    /// static string, since that much requires no live PandRS state.
    pub fn register_magics() -> String {
        r#"
        # PandRS Magic Commands for Jupyter — illustrative template.
        #
        # This is NOT a working integration: PandRS is a Rust library and
        # has no bridge back into a live IPython kernel. Adapt the method
        # bodies below to call into your own PandRS binding before use.

        from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
        from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

        @magics_class
        class PandRSMagics(Magics):

            @line_magic
            def pandrs_info(self, line):
                """Display PandRS version and configuration info"""
                return "PandRS 0.4.1 - High-performance DataFrame library for Rust"

            @line_magic
            def pandrs_config(self, line):
                """Template only: not wired to a live PandRS configuration."""
                raise NotImplementedError(
                    "pandrs_config is an illustrative template — wire this "
                    "method to your own PandRS configuration object before use."
                )

            @cell_magic
            def pandrs_sql(self, line, cell):
                """Template only: does not execute SQL against any DataFrame."""
                raise NotImplementedError(
                    "pandrs_sql is an illustrative template — it does not run "
                    "the cell against any DataFrame. Wire this method to your "
                    "own PandRS query engine before use. Received:\n" + cell
                )

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

/// Sanitize a value before it is interpolated, unescaped, into a
/// `<style>...</style>` text block (e.g. `JupyterConfig`'s font family,
/// cell padding, font size, or a `Custom` color scheme's color strings).
///
/// These are plain-text `format!` substitutions into the middle of raw
/// HTML, not attribute values, so the one character sequence that
/// matters is `<`: legitimate CSS values (font names, "8px 12px",
/// "#rrggbb", "rgba(...)") never need it, while an attacker- or
/// mistake-supplied value containing it could close the `<style>` tag
/// early (e.g. via `</style><script>...`) and inject arbitrary markup.
/// Stripping `<`/`>` entirely closes that off without rejecting any
/// value a real CSS property would use.
fn sanitize_css_value(input: &str) -> String {
    input.chars().filter(|&c| c != '<' && c != '>').collect()
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

/// Format a byte count as a human-scaled "~N KB" string.
///
/// Plain `bytes / 1024` integer division reads as "~0 KB" for every
/// DataFrame under 1 KiB (i.e. most small/example frames), which looks
/// exactly like a broken/unmeasured estimate. Use float division so
/// small-but-nonzero sizes are visible, and switch to MB once the size
/// warrants it.
fn format_memory_kb(bytes: u64) -> String {
    let kb = bytes as f64 / 1024.0;
    if kb >= 1024.0 {
        format!("{:.2} MB", kb / 1024.0)
    } else if kb >= 1.0 {
        format!("{:.1} KB", kb)
    } else {
        format!("{:.2} KB", kb)
    }
}

fn estimate_memory_usage(df: &DataFrame) -> String {
    // Rough estimate: number of cells * average cell size (8 bytes/cell,
    // a reasonable stand-in given columns are stored as plain Vec<T> of
    // heterogeneous element types rather than a single fixed width).
    let cell_count = df.row_count() * df.column_names().len();
    format_memory_kb((cell_count * 8) as u64)
}

/// Count values that carry no information in a column.
///
/// `DataFrame`'s columns here are plain `Vec<T>` with no separate null
/// bitmap, so "missing" only has meaning per-type:
/// - Floating point columns use `NaN` as the missing sentinel, the same
///   convention `DataFrame`'s own outer-join fill logic uses.
/// - Every other column type (ints, bools, strings, ...) has no
///   sentinel available at this layer; an empty string is treated as
///   missing, matching how such columns display in `to_jupyter_html`.
fn count_column_nulls(df: &DataFrame, column_name: &str) -> usize {
    if df.is_numeric_column(column_name) {
        return df
            .get_column_numeric_values(column_name)
            .map(|values| values.iter().filter(|v| v.is_nan()).count())
            .unwrap_or(0);
    }
    df.get_column_string_values(column_name)
        .map(|values| values.iter().filter(|v| v.is_empty()).count())
        .unwrap_or(0)
}

/// Process-wide counter backing [`next_widget_id`].
static WIDGET_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generate a DOM-id suffix that is unique across every widget rendered
/// in this process.
///
/// The previous scheme used `DataFrame::row_count()` directly as the id,
/// so any two DataFrames with the same number of rows — an extremely
/// common coincidence, e.g. two 100-row frames — produced colliding
/// `id="pandrs-widget-N"` elements. With duplicate ids, clicking a
/// button in one widget can look up and mutate the *other* widget's DOM
/// nodes instead. A monotonic counter has no such collisions.
fn next_widget_id() -> u64 {
    WIDGET_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Render a real (not placeholder) descriptive-statistics table for
/// every numeric column, backing both the interactive widget's
/// "Describe" panel and the data explorer's "Statistics" tab.
fn describe_table_html(df: &DataFrame) -> String {
    let numeric_cols: Vec<String> = df
        .column_names()
        .iter()
        .filter(|c| df.is_numeric_column(c))
        .cloned()
        .collect();
    if numeric_cols.is_empty() {
        return "<p>No numeric columns to describe.</p>".to_string();
    }

    let mut rows = Vec::new();
    for col in &numeric_cols {
        let Ok(values) = df.get_column_numeric_values(col) else {
            continue;
        };
        let finite: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
        if let Ok(stats) = crate::stats::descriptive::describe(&finite) {
            rows.push((col.clone(), stats));
        }
    }
    if rows.is_empty() {
        return "<p>No numeric data available to describe.</p>".to_string();
    }

    let mut html = String::from(
        "<table class=\"pandrs-describe-table\"><thead><tr>\
         <th></th><th>count</th><th>mean</th><th>std</th><th>min</th>\
         <th>25%</th><th>50%</th><th>75%</th><th>max</th></tr></thead><tbody>",
    );
    for (name, s) in &rows {
        html.push_str(&format!(
            "<tr><td><strong>{}</strong></td><td>{}</td><td>{:.4}</td><td>{:.4}</td>\
             <td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>",
            escape_html(name),
            s.count,
            s.mean,
            s.std,
            s.min,
            s.quartiles.q1,
            s.quartiles.q2,
            s.quartiles.q3,
            s.max,
        ));
    }
    html.push_str("</tbody></table>");
    html
}

/// Render real (not placeholder) quick-look charts — a histogram, and,
/// with at least two numeric columns, a scatter plot and a correlation
/// heatmap — using [`crate::vis::svg::dataframe_ext::SvgVisualize`],
/// backing both the interactive widget's "Plot" panel and the data
/// explorer's "Visualize" tab.
fn quick_viz_html(df: &DataFrame) -> String {
    use crate::vis::svg::dataframe_ext::SvgVisualize;

    let numeric_cols: Vec<String> = df
        .column_names()
        .iter()
        .filter(|c| df.is_numeric_column(c))
        .cloned()
        .collect();
    if numeric_cols.is_empty() {
        return "<p>No numeric columns available to visualize.</p>".to_string();
    }

    let mut html = String::new();
    match df.plot_histogram_svg(&numeric_cols[0], 10, None) {
        Ok(svg) => html.push_str(&format!(
            "<h4>Histogram: {}</h4>{}",
            escape_html(&numeric_cols[0]),
            svg
        )),
        Err(e) => html.push_str(&format!(
            "<p>Histogram unavailable: {}</p>",
            escape_html(&e.to_string())
        )),
    }

    if numeric_cols.len() >= 2 {
        if let Ok(svg) = df.plot_scatter_svg(&numeric_cols[0], &numeric_cols[1], None) {
            html.push_str(&format!(
                "<h4>Scatter: {} vs {}</h4>{}",
                escape_html(&numeric_cols[1]),
                escape_html(&numeric_cols[0]),
                svg
            ));
        }
        let col_refs: Vec<&str> = numeric_cols.iter().map(String::as_str).collect();
        if let Ok(corr_df) = df.corr_matrix(&col_refs) {
            if let Ok(svg) = corr_df.plot_heatmap_svg(None) {
                html.push_str(&format!("<h4>Correlation</h4>{}", svg));
            }
        }
    }
    html
}

fn estimate_null_count(df: &DataFrame) -> usize {
    df.column_names()
        .iter()
        .map(|col| count_column_nulls(df, col))
        .sum()
}

fn estimate_non_null_count(df: &DataFrame) -> usize {
    let total = df.row_count() * df.column_names().len();
    total.saturating_sub(estimate_null_count(df))
}

impl DataFrame {
    /// Infer the type of a column for display purposes.
    ///
    /// Samples the whole column rather than just row 0: a column whose
    /// first row happens to be empty/missing (or, for a numeric column
    /// stored as text, an integer-looking value like "0") previously
    /// reported a type that didn't reflect the rest of the data.
    fn infer_column_type(&self, column_name: &str) -> String {
        let values = match self.get_column_string_values(column_name) {
            Ok(v) => v,
            Err(_) => return "unknown".to_string(),
        };
        let sample: Vec<&str> = values
            .iter()
            .map(String::as_str)
            .filter(|v| !v.is_empty())
            .collect();
        if sample.is_empty() {
            return "unknown".to_string();
        }
        if sample.iter().all(|v| v.parse::<i64>().is_ok()) {
            "integer".to_string()
        } else if sample.iter().all(|v| v.parse::<f64>().is_ok()) {
            "float".to_string()
        } else if sample.iter().all(|v| v.parse::<bool>().is_ok()) {
            "boolean".to_string()
        } else {
            "string".to_string()
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

    /// Compute descriptive statistics (count, mean, std, min, quartiles,
    /// max, skewness, kurtosis, ...) for every numeric column and
    /// serialize them as a JSON object keyed by column name.
    ///
    /// This backs the "Describe" panel in [`JupyterDisplay::to_interactive_widget`]
    /// and the "Statistics" tab in [`JupyterDisplay::to_data_explorer`],
    /// which previously showed the static placeholder text "Summary
    /// statistics would be displayed here...".
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails. Columns with no
    /// finite values are omitted rather than causing the whole call to
    /// fail.
    pub fn describe_to_json(&self) -> Result<String> {
        let numeric_cols: Vec<String> = self
            .column_names()
            .iter()
            .filter(|c| self.is_numeric_column(c))
            .cloned()
            .collect();

        let mut summaries: std::collections::BTreeMap<
            String,
            crate::stats::descriptive::StatisticalSummary,
        > = std::collections::BTreeMap::new();
        for col in &numeric_cols {
            let Ok(values) = self.get_column_numeric_values(col) else {
                continue;
            };
            let finite: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
            if let Ok(stats) = crate::stats::descriptive::describe(&finite) {
                summaries.insert(col.clone(), stats);
            }
        }

        serde_json::to_string_pretty(&summaries).map_err(Error::Json)
    }
}

lazy_static! {
    /// Display configuration for Jupyter notebooks
    static ref JUPYTER_CONFIG: RwLock<Option<JupyterConfig>> = RwLock::new(None);
}

/// Get the current Jupyter configuration
pub fn get_jupyter_config() -> JupyterConfig {
    read_lock_safe!(JUPYTER_CONFIG, "jupyter config read")
        .ok()
        .and_then(|c| c.clone())
        .unwrap_or_default()
}

/// Set the Jupyter configuration
pub fn set_jupyter_config(config: JupyterConfig) {
    if let Ok(mut cfg) = write_lock_safe!(JUPYTER_CONFIG, "jupyter config write") {
        *cfg = Some(config);
    }
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

        let df = DataFrame::from_map(data, None).expect("operation should succeed");
        let config = JupyterConfig::default();

        let html = df
            .to_jupyter_html(&config)
            .expect("operation should succeed");
        assert!(html.contains("pandrs-table"));
        assert!(html.contains("col1"));
        assert!(html.contains("col2"));
    }
}
