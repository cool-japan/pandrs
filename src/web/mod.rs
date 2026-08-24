//! WebAssembly support for interactive browser visualization
//!
//! This module provides WebAssembly (wasm) integration for PandRS, allowing for
//! interactive data visualization in web browsers.

use plotters::drawing::IntoDrawingArea;
use plotters_canvas::CanvasBackend;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::error::{Error, Result};
use crate::vis::plotters_ext::{PlotKind, PlotSettings};
use crate::DataFrame;

// Type alias for web-compatible results
type WebResult<T> = std::result::Result<T, JsValue>;

// Helper to convert our errors to JsValue
fn to_js_error<E: std::fmt::Display>(err: E) -> JsValue {
    JsValue::from_str(&err.to_string())
}

/// Possible color themes for web visualizations
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ColorTheme {
    Default,
    Dark,
    Light,
    Pastel,
    Vibrant,
}

/// Types of interactive visualizations
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum VisualizationType {
    Line,
    Bar,
    Scatter,
    Area,
    Pie,
    Histogram,
    BoxPlot,
    HeatMap,
}

/// Configuration for web visualizations
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebVisualizationConfig {
    /// Canvas element ID
    canvas_id: String,
    /// Chart title
    title: String,
    /// Visualization type
    viz_type: VisualizationType,
    /// Color theme
    theme: ColorTheme,
    /// Width of the visualization
    width: u32,
    /// Height of the visualization
    height: u32,
    /// Show interactive legend
    show_legend: bool,
    /// Show interactive tooltips
    show_tooltips: bool,
    /// Enable animation
    animate: bool,
}

#[wasm_bindgen]
impl WebVisualizationConfig {
    /// Create a new web visualization configuration
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Self {
        WebVisualizationConfig {
            canvas_id: canvas_id.to_string(),
            title: "Chart".to_string(),
            viz_type: VisualizationType::Line,
            theme: ColorTheme::Default,
            width: 800,
            height: 600,
            show_legend: true,
            show_tooltips: true,
            animate: true,
        }
    }

    /// Set the title
    #[wasm_bindgen]
    pub fn set_title(&mut self, title: &str) -> Self {
        self.title = title.to_string();
        self.clone()
    }

    /// Set the visualization type
    #[wasm_bindgen]
    pub fn set_type(&mut self, viz_type: VisualizationType) -> Self {
        self.viz_type = viz_type;
        self.clone()
    }

    /// Set the color theme
    #[wasm_bindgen]
    pub fn set_theme(&mut self, theme: ColorTheme) -> Self {
        self.theme = theme;
        self.clone()
    }

    /// Set the dimensions
    #[wasm_bindgen]
    pub fn set_dimensions(&mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self.clone()
    }

    /// Set whether to show legend
    #[wasm_bindgen]
    pub fn show_legend(&mut self, show: bool) -> Self {
        self.show_legend = show;
        self.clone()
    }

    /// Set whether to show tooltips
    #[wasm_bindgen]
    pub fn show_tooltips(&mut self, show: bool) -> Self {
        self.show_tooltips = show;
        self.clone()
    }

    /// Set whether to animate
    #[wasm_bindgen]
    pub fn animate(&mut self, animate: bool) -> Self {
        self.animate = animate;
        self.clone()
    }
}

/// Interactive visualization for the web
#[wasm_bindgen]
pub struct WebVisualization {
    config: WebVisualizationConfig,
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
    /// A sibling canvas stacked exactly on top of `canvas`, used only for
    /// drawing the tooltip. Tooltip drawing must never touch `context`
    /// (the chart canvas): the previous implementation called
    /// `context.clear_rect` on every `mousemove`, which erased the whole
    /// chart and left only a floating "Tooltip" box behind. Since the
    /// overlay starts fully transparent and pointer-events are disabled
    /// on it, clearing *it* on each move just removes the previous
    /// tooltip without ever touching the chart underneath.
    overlay_canvas: HtmlCanvasElement,
    overlay_context: CanvasRenderingContext2d,
    data: Option<Rc<RefCell<DataFrame>>>,
    event_listeners: Vec<(String, Closure<dyn FnMut(web_sys::MouseEvent)>)>,
}

#[wasm_bindgen]
impl WebVisualization {
    /// Create a new web visualization
    #[wasm_bindgen(constructor)]
    pub fn new(config: WebVisualizationConfig) -> WebResult<WebVisualization> {
        // Get the canvas element
        let window = web_sys::window().ok_or_else(|| to_js_error("No window object available"))?;

        let document = window
            .document()
            .ok_or_else(|| to_js_error("No document object available"))?;

        let canvas = document
            .get_element_by_id(&config.canvas_id)
            .ok_or_else(|| {
                to_js_error(format!(
                    "Canvas element with ID '{}' not found",
                    config.canvas_id
                ))
            })?;

        let canvas = canvas
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| to_js_error("Element is not a canvas"))?;

        // Set canvas dimensions
        canvas.set_width(config.width);
        canvas.set_height(config.height);

        // Get drawing context
        let context = canvas
            .get_context("2d")
            .map_err(|_| to_js_error("Failed to get canvas context"))?
            .ok_or_else(|| to_js_error("Canvas context is null"))?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| to_js_error("Failed to convert to CanvasRenderingContext2d"))?;

        // A same-sized sibling canvas, absolutely positioned directly on
        // top of the chart canvas, used only for tooltip drawing (see
        // the `overlay_canvas`/`overlay_context` doc comment on
        // `WebVisualization`). `pointer-events: none` keeps it from
        // intercepting the mouse events the chart canvas needs.
        let overlay_canvas = document
            .create_element("canvas")
            .map_err(|_| to_js_error("Failed to create tooltip overlay canvas"))?
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| to_js_error("Failed to cast overlay element to canvas"))?;
        overlay_canvas.set_width(config.width);
        overlay_canvas.set_height(config.height);
        {
            let style = overlay_canvas.style();
            let _ = style.set_property("position", "absolute");
            let _ = style.set_property("pointer-events", "none");
            let _ = style.set_property("left", &format!("{}px", canvas.offset_left()));
            let _ = style.set_property("top", &format!("{}px", canvas.offset_top()));
        }
        if let Some(parent) = canvas.parent_node() {
            let _ = parent.append_child(&overlay_canvas);
        }
        let overlay_context = overlay_canvas
            .get_context("2d")
            .map_err(|_| to_js_error("Failed to get overlay canvas context"))?
            .ok_or_else(|| to_js_error("Overlay canvas context is null"))?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| to_js_error("Failed to convert overlay context"))?;

        Ok(WebVisualization {
            config,
            canvas,
            context,
            overlay_canvas,
            overlay_context,
            data: None,
            event_listeners: Vec::new(),
        })
    }

    /// Set the DataFrame to visualize
    #[wasm_bindgen]
    pub fn set_data(&mut self, data_json: &str) -> WebResult<()> {
        // Parse JSON to DataFrame
        let df = DataFrame::from_json(data_json).map_err(|e| to_js_error(e))?;
        self.data = Some(Rc::new(RefCell::new(df)));
        Ok(())
    }

    /// Re-align the tooltip overlay canvas with the chart canvas.
    ///
    /// The overlay is positioned with `left`/`top` computed once in
    /// [`WebVisualization::new`]; if the page reflows the chart canvas
    /// afterward (a resize, a layout change elsewhere on the page, ...)
    /// the overlay would drift out of alignment and tooltips would be
    /// drawn in the wrong place relative to the chart. Called at the
    /// start of every [`WebVisualization::render`] so the overlay tracks
    /// the chart canvas's current position.
    fn sync_overlay_position(&self) {
        let style = self.overlay_canvas.style();
        let _ = style.set_property("left", &format!("{}px", self.canvas.offset_left()));
        let _ = style.set_property("top", &format!("{}px", self.canvas.offset_top()));
    }

    /// Render the visualization
    #[wasm_bindgen]
    pub fn render(&mut self) -> WebResult<()> {
        self.sync_overlay_position();
        {
            let df = match &self.data {
                Some(df) => df.borrow(),
                None => return Err(to_js_error("No data available to visualize")),
            };

            // Clear canvas
            self.context.clear_rect(
                0.0,
                0.0,
                self.config.width as f64,
                self.config.height as f64,
            );

            // Get column names
            let columns = df.column_names();
            if columns.is_empty() {
                return Err(to_js_error("DataFrame has no columns"));
            }

            // Create a plotters backend with the canvas
            let backend = CanvasBackend::with_canvas_object(self.canvas.clone())
                .ok_or_else(|| to_js_error("Failed to create canvas backend"))?;

            // Convert WebVisualizationConfig to PlotSettings
            let settings = self.create_plot_settings();

            // Render based on visualization type
            match self.config.viz_type {
                VisualizationType::Line => self.render_line_chart(&df, backend, &settings)?,
                VisualizationType::Bar => self.render_bar_chart(&df, backend, &settings)?,
                VisualizationType::Scatter => self.render_scatter_chart(&df, backend, &settings)?,
                VisualizationType::Area => self.render_area_chart(&df, backend, &settings)?,
                VisualizationType::Histogram => self.render_histogram(&df, backend, &settings)?,
                VisualizationType::BoxPlot => self.render_boxplot(&df, backend, &settings)?,
                VisualizationType::Pie => self.render_pie_chart(&df, backend, &settings)?,
                VisualizationType::HeatMap => self.render_heatmap(&df, backend, &settings)?,
            }
        }

        // Set up event listeners for interactivity if tooltips are enabled
        if self.config.show_tooltips {
            self.setup_tooltip_listeners()?;
        }

        Ok(())
    }

    // Convert WebVisualizationConfig to PlotSettings
    fn create_plot_settings(&self) -> PlotSettings {
        let mut settings = PlotSettings::default();

        settings.title = self.config.title.clone();
        settings.width = self.config.width;
        settings.height = self.config.height;
        settings.show_legend = self.config.show_legend;

        // Convert visualization type
        settings.plot_kind = match self.config.viz_type {
            VisualizationType::Line => PlotKind::Line,
            VisualizationType::Bar => PlotKind::Bar,
            VisualizationType::Scatter => PlotKind::Scatter,
            VisualizationType::Area => PlotKind::Area,
            VisualizationType::Histogram => PlotKind::Histogram,
            VisualizationType::BoxPlot => PlotKind::BoxPlot,
            // Default to Line for types not supported in PlotKind
            _ => PlotKind::Line,
        };

        // Set color palette based on theme
        settings.color_palette = match self.config.theme {
            ColorTheme::Default => vec![
                (0, 123, 255),  // Blue
                (255, 99, 71),  // Red
                (46, 204, 113), // Green
                (255, 193, 7),  // Yellow
                (142, 68, 173), // Purple
            ],
            ColorTheme::Dark => vec![
                (41, 98, 255), // Blue
                (221, 65, 36), // Red
                (11, 156, 49), // Green
                (255, 147, 0), // Orange
                (116, 0, 184), // Purple
            ],
            ColorTheme::Light => vec![
                (99, 179, 237),  // Light blue
                (255, 161, 145), // Light red
                (134, 226, 173), // Light green
                (255, 230, 140), // Light yellow
                (198, 163, 229), // Light purple
            ],
            ColorTheme::Pastel => vec![
                (174, 198, 242), // Pastel blue
                (255, 179, 186), // Pastel red
                (186, 241, 191), // Pastel green
                (255, 239, 186), // Pastel yellow
                (220, 198, 239), // Pastel purple
            ],
            ColorTheme::Vibrant => vec![
                (0, 116, 217),  // Vibrant blue
                (255, 65, 54),  // Vibrant red
                (46, 204, 64),  // Vibrant green
                (255, 220, 0),  // Vibrant yellow
                (177, 13, 201), // Vibrant purple
            ],
        };

        settings
    }

    // Helper methods for rendering different chart types
    fn render_line_chart(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Get numeric columns
        let mut numeric_columns: Vec<String> = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(col_name.as_str()) {
                numeric_columns.push(col_name.to_string());
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::InvalidInput(
                "No numeric columns available for line chart".to_string(),
            ));
        }

        // Select columns to plot (up to 5)
        let columns_to_plot = if numeric_columns.len() > 5 {
            numeric_columns[0..5].to_vec()
        } else {
            numeric_columns
        };

        // Convert to &[&str] for plotters
        let columns_str: Vec<&str> = columns_to_plot.iter().map(|s| s.as_str()).collect();

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Line;

        // Use existing plotting functionality from plotters_ext module
        plotters_ext_web::plot_multi_series_for_web(
            df,
            &columns_str,
            drawing_area,
            &plot_settings,
        )?;

        Ok(())
    }

    fn render_bar_chart(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Similar to line chart but with bar plot
        let mut numeric_columns: Vec<String> = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(col_name.as_str()) {
                numeric_columns.push(col_name.to_string());
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::InvalidInput(
                "No numeric columns available for bar chart".to_string(),
            ));
        }

        // For bar chart, let's just use the first numeric column
        let column = &numeric_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Bar;

        // Use existing plotting functionality
        plotters_ext_web::plot_column_for_web(df, column, drawing_area, &plot_settings)?;

        Ok(())
    }

    fn render_scatter_chart(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Need at least two numeric columns for scatter plot
        let mut numeric_columns: Vec<String> = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(col_name.as_str()) {
                numeric_columns.push(col_name.to_string());
            }
        }

        if numeric_columns.len() < 2 {
            return Err(Error::InvalidInput(
                "Need at least two numeric columns for scatter plot".to_string(),
            ));
        }

        let x_column = &numeric_columns[0];
        let y_column = &numeric_columns[1];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Scatter;

        // Use existing scatter plot functionality
        plotters_ext_web::plot_scatter_for_web(
            df,
            x_column,
            y_column,
            drawing_area,
            &plot_settings,
        )?;

        Ok(())
    }

    fn render_area_chart(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Similar to line chart but with area plot
        let mut numeric_columns: Vec<String> = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(col_name.as_str()) {
                numeric_columns.push(col_name.to_string());
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::InvalidInput(
                "No numeric columns available for area chart".to_string(),
            ));
        }

        // For area chart, just use the first numeric column
        let column = &numeric_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Area;

        // Use existing plotting functionality
        plotters_ext_web::plot_column_for_web(df, column, drawing_area, &plot_settings)?;

        Ok(())
    }

    fn render_histogram(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Find a numeric column for histogram
        let mut numeric_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
                break; // Only need one numeric column
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::InvalidInput(
                "No numeric columns available for histogram".to_string(),
            ));
        }

        let column = &numeric_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Histogram;

        // Use existing histogram functionality
        plotters_ext_web::plot_histogram_for_web(
            df,
            column,
            10, // Default 10 bins
            drawing_area,
            &plot_settings,
        )?;

        Ok(())
    }

    fn render_boxplot(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Need a numeric column and a categorical column
        let mut numeric_columns = Vec::new();
        let mut categorical_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            } else if df.is_categorical(&col_name) || !df.is_numeric_column(&col_name) {
                categorical_columns.push(col_name);
            }
        }

        if numeric_columns.is_empty() || categorical_columns.is_empty() {
            return Err(Error::InvalidInput(
                "Need at least one numeric and one categorical column for boxplot".to_string(),
            ));
        }

        let value_column = &numeric_columns[0];
        let category_column = &categorical_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::BoxPlot;

        // Use existing boxplot functionality
        plotters_ext_web::plot_boxplot_for_web(
            df,
            category_column,
            value_column,
            drawing_area,
            &plot_settings,
        )?;

        Ok(())
    }

    fn render_pie_chart(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Need a numeric column and a categorical column for pie chart
        let mut numeric_columns = Vec::new();
        let mut categorical_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            } else if df.is_categorical(&col_name) || !df.is_numeric_column(&col_name) {
                categorical_columns.push(col_name);
            }
        }

        if numeric_columns.is_empty() || categorical_columns.is_empty() {
            return Err(Error::InvalidInput(
                "Need at least one numeric and one categorical column for pie chart".to_string(),
            ));
        }

        let value_column = &numeric_columns[0];
        let category_column = &categorical_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let plot_settings = settings.clone();

        // Use custom pie chart implementation for web
        self.draw_pie_chart(df, category_column, value_column, &plot_settings)?;

        Ok(())
    }

    fn render_heatmap(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Need at least two numeric columns for heatmap
        let mut numeric_columns: Vec<String> = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(col_name.as_str()) {
                numeric_columns.push(col_name.to_string());
            }
        }

        if numeric_columns.len() < 2 {
            return Err(Error::InvalidInput(
                "Need at least two numeric columns for heatmap".to_string(),
            ));
        }

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::Visualization(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let plot_settings = settings.clone();

        // Use custom heatmap implementation for web
        self.draw_heatmap(df, &numeric_columns, &plot_settings)?;

        Ok(())
    }

    // Custom implementations for pie charts and heatmaps
    fn draw_pie_chart(
        &self,
        df: &DataFrame,
        category_col: &str,
        value_col: &str,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Get values and labels
        let categories = df.get_column_string_values(category_col)?;
        let values = df.get_column_numeric_values(value_col)?;

        if categories.len() != values.len() {
            return Err(Error::DimensionMismatch(
                "Category and value columns must have the same length".to_string(),
            ));
        }

        // Aggregate values by category
        let mut category_values = std::collections::HashMap::new();
        for (cat, val) in categories.iter().zip(values.iter()) {
            *category_values.entry(cat.clone()).or_insert(0.0) += *val as f64;
        }

        // Calculate total
        let total: f64 = category_values.values().sum();
        if total <= 0.0 {
            // A zero/negative total makes `value / total` NaN or
            // sign-flipped for every slice (all category_values.sum()
            // to <= 0, e.g. every value is 0), which cannot be drawn as
            // a meaningful pie; fail honestly instead of feeding NaN
            // angles into the canvas arc calls. Mirrors the equivalent
            // guard in `vis::svg::charts::PieChart::render`.
            return Err(Error::InvalidInput(
                "PieChart: total must be positive".to_string(),
            ));
        }

        // Draw pie chart
        let cx = (self.config.width as f64) / 2.0;
        let cy = (self.config.height as f64) / 2.0;
        let radius = (self.config.width.min(self.config.height) as f64) * 0.4;

        // Draw title. A failed text draw is not worth aborting the whole
        // visualization over, so failures are ignored rather than
        // `.expect()`-ed (which would panic and abort the WASM module).
        self.context.set_font("16px sans-serif");
        self.context.set_text_align("center");
        self.context.set_fill_style_str("black");
        let _ = self.context.fill_text(&settings.title, cx, 30.0);

        // Draw pie slices
        let mut start_angle = 0.0;
        let mut i = 0;
        let colors = settings.color_palette.clone();
        // An empty `color_palette` would otherwise make `i % colors.len()`
        // a division-by-zero panic; fall back to a single reasonable
        // default color in that case instead.
        let palette_len = colors.len().max(1);
        let color_at = |idx: usize| -> (u8, u8, u8) {
            colors
                .get(idx % palette_len)
                .copied()
                .unwrap_or((70, 130, 180))
        };

        for (_category, value) in &category_values {
            let slice_angle = 2.0 * std::f64::consts::PI * (value / total);

            let (r, g, b) = color_at(i);
            let color = format!("rgb({}, {}, {})", r, g, b);

            // Draw slice
            self.context.begin_path();
            self.context.move_to(cx, cy);
            let _ = self
                .context
                .arc(cx, cy, radius, start_angle, start_angle + slice_angle);
            self.context.close_path();

            self.context.set_fill_style_str(&color);
            self.context.fill();

            // Draw slice outline
            self.context.set_stroke_style_str("white");
            self.context.set_line_width(1.0);
            self.context.stroke();

            // Calculate label position
            let label_angle = start_angle + slice_angle / 2.0;
            let label_x = cx + radius * 0.7 * label_angle.cos();
            let label_y = cy + radius * 0.7 * label_angle.sin();

            // Draw percentage label
            let percentage = (value / total * 100.0).round() / 10.0 * 10.0;
            self.context.set_font("12px sans-serif");
            self.context.set_fill_style_str("white");
            self.context.set_text_align("center");

            if percentage >= 5.0 {
                // Only show label if slice is big enough
                let _ = self
                    .context
                    .fill_text(&format!("{}%", percentage), label_x, label_y);
            }

            start_angle += slice_angle;
            i += 1;
        }

        // Draw legend
        if settings.show_legend {
            let legend_x = self.config.width as f64 - 150.0;
            let legend_y = 50.0;
            let mut y_offset = 0.0;

            i = 0;
            for (category, _) in &category_values {
                let (r, g, b) = color_at(i);
                let color = format!("rgb({}, {}, {})", r, g, b);

                // Draw color square
                self.context.set_fill_style_str(&color);
                self.context
                    .fill_rect(legend_x, legend_y + y_offset, 15.0, 15.0);

                // Draw category name
                self.context.set_font("12px sans-serif");
                self.context.set_fill_style_str("black");
                self.context.set_text_align("left");

                // Truncate long category names. Byte-slicing
                // (`&category[0..12]`) panics as soon as a category name
                // contains any multi-byte UTF-8 character (e.g. Japanese
                // text) whose encoding straddles byte offset 12; slicing
                // by `char` instead is always a valid boundary.
                let display_cat = if category.chars().count() > 15 {
                    let truncated: String = category.chars().take(12).collect();
                    format!("{}...", truncated)
                } else {
                    category.clone()
                };

                let _ = self.context.fill_text(
                    &display_cat,
                    legend_x + 20.0,
                    legend_y + y_offset + 12.0,
                );

                y_offset += 20.0;
                i += 1;
            }
        }

        Ok(())
    }

    fn draw_heatmap(
        &self,
        df: &DataFrame,
        columns: &[String],
        settings: &PlotSettings,
    ) -> Result<()> {
        // Get matrix of values
        let mut data_matrix = Vec::new();

        for col in columns {
            let values = df.get_column_numeric_values(col)?;
            data_matrix.push(values);
        }

        // Transpose matrix if needed
        let rows = data_matrix.len();
        if rows == 0 {
            return Err(Error::InvalidInput("No data for heatmap".to_string()));
        }

        // The shortest column's length, not just the first column's: if
        // columns are ragged (one shorter than the rest — e.g. from
        // upstream NA-dropping that only touched some columns),
        // indexing every row up to `data_matrix[0].len()` would run past
        // the end of any shorter column and panic. Bounding by the
        // minimum keeps every `data_matrix[i][j]` access below in
        // bounds for every row `i`; any extra trailing values in longer
        // columns are simply not drawn.
        let cols = data_matrix.iter().map(|c| c.len()).min().unwrap_or(0);
        if cols == 0 {
            return Err(Error::InvalidInput("Empty columns for heatmap".to_string()));
        }

        // Find min and max values
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;

        for row in &data_matrix {
            for &val in row {
                let val = val as f64;
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Draw title. Text-draw failures are ignored (not `.expect()`-ed
        // into a panic that would abort the whole WASM module) since a
        // missing label is not worth losing the rest of the chart over.
        self.context.set_font("16px sans-serif");
        self.context.set_text_align("center");
        self.context.set_fill_style_str("black");
        let _ = self
            .context
            .fill_text(&settings.title, (self.config.width as f64) / 2.0, 30.0);

        // Calculate dimensions
        let margin = 70.0;
        let chart_width = self.config.width as f64 - 2.0 * margin;
        let chart_height = self.config.height as f64 - 2.0 * margin;

        let cell_width = chart_width / cols as f64;
        let cell_height = chart_height / rows as f64;

        // Draw heatmap cells
        for i in 0..rows {
            for j in 0..cols {
                let x = margin + j as f64 * cell_width;
                let y = margin + i as f64 * cell_height;

                let val = data_matrix[i][j] as f64;

                // Normalize value between 0 and 1
                let normalized = if max_val > min_val {
                    (val - min_val) / (max_val - min_val)
                } else {
                    0.5 // Avoid division by zero
                };

                // Use a color gradient from blue to red
                let r = (normalized * 255.0) as u8;
                let b = (255.0 - normalized * 255.0) as u8;
                let color = format!("rgb({}, 0, {})", r, b);

                // Draw cell
                self.context.set_fill_style_str(&color);
                self.context.fill_rect(x, y, cell_width, cell_height);

                // Draw cell outline
                self.context.set_stroke_style_str("rgba(255,255,255,0.2)");
                self.context.set_line_width(0.5);
                self.context.stroke_rect(x, y, cell_width, cell_height);

                // Draw value text if cells are large enough
                if cell_width > 30.0 && cell_height > 20.0 {
                    self.context.set_font("10px sans-serif");
                    self.context.set_fill_style_str("white");
                    self.context.set_text_align("center");

                    let _ = self.context.fill_text(
                        &format!("{:.1}", val),
                        x + cell_width / 2.0,
                        y + cell_height / 2.0 + 3.0,
                    );
                }
            }
        }

        // Draw column labels
        self.context.set_font("12px sans-serif");
        self.context.set_fill_style_str("black");
        self.context.set_text_align("center");

        for (j, col) in columns.iter().enumerate().take(cols) {
            let x = margin + j as f64 * cell_width + cell_width / 2.0;
            let y = margin - 10.0;

            // Truncate long column names. Byte-slicing (`&col[0..7]`)
            // panics as soon as a name contains a multi-byte UTF-8
            // character whose encoding straddles byte offset 7 (e.g. a
            // Japanese column name); slicing by `char` is always valid.
            let display_name = if col.chars().count() > 10 {
                let truncated: String = col.chars().take(7).collect();
                format!("{}...", truncated)
            } else {
                col.clone()
            };

            let _ = self.context.fill_text(&display_name, x, y);
        }

        // Draw row labels (use row indices)
        self.context.set_text_align("right");

        for i in 0..rows {
            let x = margin - 10.0;
            let y = margin + i as f64 * cell_height + cell_height / 2.0 + 5.0;

            let _ = self.context.fill_text(&format!("Row {}", i + 1), x, y);
        }

        // Draw color scale
        let scale_width = 20.0;
        let scale_height = chart_height * 0.7;
        let scale_x = self.config.width as f64 - margin / 2.0;
        let scale_y = margin + (chart_height - scale_height) / 2.0;

        for i in 0..100 {
            let normalized = i as f64 / 100.0;
            let r = (normalized * 255.0) as u8;
            let b = (255.0 - normalized * 255.0) as u8;
            let color = format!("rgb({}, 0, {})", r, b);

            self.context.set_fill_style_str(&color);
            self.context.fill_rect(
                scale_x - scale_width / 2.0,
                scale_y + (1.0 - normalized) * scale_height,
                scale_width,
                scale_height / 100.0,
            );
        }

        // Draw scale labels
        self.context.set_font("10px sans-serif");
        self.context.set_fill_style_str("black");
        self.context.set_text_align("center");

        let _ = self
            .context
            .fill_text(&format!("{:.1}", max_val), scale_x, scale_y - 5.0);

        let _ = self.context.fill_text(
            &format!("{:.1}", min_val),
            scale_x,
            scale_y + scale_height + 15.0,
        );

        Ok(())
    }

    // Set up interactive tooltips
    fn setup_tooltip_listeners(&mut self) -> Result<()> {
        // Clean up any existing event listeners
        for (event, listener) in self.event_listeners.drain(..) {
            self.canvas
                .remove_event_listener_with_callback(&event, listener.as_ref().unchecked_ref())
                .map_err(|_| Error::InvalidInput("Failed to remove event listener".to_string()))?;
        }

        // Add mousemove listener for tooltips. All drawing happens on
        // `overlay_context` — a separate canvas stacked on top of the
        // chart (see the field doc comment on `WebVisualization`) — so
        // hovering the mouse never clears or otherwise touches the
        // chart canvas itself, unlike the previous implementation which
        // called `clear_rect` on the chart's own context every move.
        let canvas_clone = self.canvas.clone();
        let overlay_context_clone = self.overlay_context.clone();
        let config_clone = self.config.clone();
        let data_clone = self.data.clone();

        let mousemove_callback = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
            let rect = canvas_clone.get_bounding_client_rect();
            let x = event.client_x() as f64 - rect.left();
            let y = event.client_y() as f64 - rect.top();
            let width = config_clone.width as f64;
            let height = config_clone.height as f64;

            overlay_context_clone.clear_rect(0.0, 0.0, width, height);

            if x < 0.0 || y < 0.0 || x > width || y > height {
                return;
            }
            let Some(data) = &data_clone else {
                return;
            };
            let df = data.borrow();
            let row_count = df.row_count();
            if row_count == 0 {
                return;
            }

            // Map the cursor's horizontal position proportionally across
            // the canvas to a row index and read that row's real values
            // out of the DataFrame. This does not reproduce plotters'
            // exact per-chart margin/label-area pixel geometry — that
            // transform is internal to each `render_*_chart` call and
            // is not retained afterward — but it is a real, monotonic
            // lookup into the actual plotted data, not a fixed string.
            let frac = (x / width.max(1.0)).clamp(0.0, 1.0);
            let row = ((frac * row_count as f64) as usize).min(row_count - 1);

            let mut lines: Vec<String> = vec![format!("row {}", row)];
            for col in df.column_names() {
                if lines.len() > 5 {
                    lines.push("...".to_string());
                    break;
                }
                if let Ok(values) = df.get_column_string_values(col) {
                    if let Some(v) = values.get(row) {
                        lines.push(format!("{}: {}", col, v));
                    }
                }
            }

            let line_height = 14.0;
            let box_h = line_height * lines.len() as f64 + 10.0;
            let box_w =
                lines.iter().map(|l| l.chars().count()).max().unwrap_or(4) as f64 * 6.5 + 16.0;
            let box_x = (x + 12.0).min((width - box_w).max(0.0));
            let box_y = (y - box_h - 8.0).max(0.0);

            overlay_context_clone.set_fill_style_str("rgba(0,0,0,0.75)");
            overlay_context_clone.fill_rect(box_x, box_y, box_w, box_h);

            overlay_context_clone.set_font("11px sans-serif");
            overlay_context_clone.set_fill_style_str("white");
            overlay_context_clone.set_text_align("left");
            for (i, line) in lines.iter().enumerate() {
                // A failed tooltip draw is not worth aborting the whole
                // WASM module over: ignore it and move on.
                let _ = overlay_context_clone.fill_text(
                    line,
                    box_x + 8.0,
                    box_y + 14.0 + i as f64 * line_height,
                );
            }
        }) as Box<dyn FnMut(_)>);

        self.canvas
            .add_event_listener_with_callback(
                "mousemove",
                mousemove_callback.as_ref().unchecked_ref(),
            )
            .map_err(|_| Error::InvalidInput("Failed to add event listener".to_string()))?;
        self.event_listeners
            .push(("mousemove".to_string(), mousemove_callback));

        // Clear any lingering tooltip once the cursor leaves the chart.
        let overlay_context_leave = self.overlay_context.clone();
        let config_leave = self.config.clone();
        let mouseleave_callback = Closure::wrap(Box::new(move |_event: web_sys::MouseEvent| {
            overlay_context_leave.clear_rect(
                0.0,
                0.0,
                config_leave.width as f64,
                config_leave.height as f64,
            );
        }) as Box<dyn FnMut(_)>);
        self.canvas
            .add_event_listener_with_callback(
                "mouseleave",
                mouseleave_callback.as_ref().unchecked_ref(),
            )
            .map_err(|_| Error::InvalidInput("Failed to add event listener".to_string()))?;
        self.event_listeners
            .push(("mouseleave".to_string(), mouseleave_callback));

        Ok(())
    }

    /// Export the visualization as an image
    #[wasm_bindgen]
    pub fn export_image(&self) -> WebResult<String> {
        self.canvas
            .to_data_url()
            .map_err(|_| to_js_error("Failed to export image"))
    }

    /// Update title
    #[wasm_bindgen]
    pub fn update_title(&mut self, title: &str) -> WebResult<()> {
        self.config.title = title.to_string();
        self.render()
    }

    /// Change visualization type
    #[wasm_bindgen]
    pub fn change_type(&mut self, viz_type: VisualizationType) -> WebResult<()> {
        self.config.viz_type = viz_type;
        self.render()
    }

    /// Update theme
    #[wasm_bindgen]
    pub fn update_theme(&mut self, theme: ColorTheme) -> WebResult<()> {
        self.config.theme = theme;
        self.render()
    }
}

// Real chart rendering for the web canvas backend.
//
// Every drawing routine here is generic over `DB: DrawingBackend`
// instead of being written directly against `CanvasBackend`. This is
// what makes these renderers testable at all under `cargo nextest`:
// `CanvasBackend` can only be constructed from a real
// `web_sys::HtmlCanvasElement`, which requires an actual browser/DOM and
// does not exist when running natively, so a native test exercises the
// exact same drawing code against `SVGBackend`/`BitMapBackend` instead.
// The public `plot_*_for_web` entry points stay concrete over
// `CanvasBackend` because that is what `WebVisualization::render` calls
// them with.
mod plotters_ext_web {
    use super::*;
    use crate::vis::plotters_ext::{PlotKind, PlotSettings};
    use plotters::coord::Shift;
    use plotters::prelude::*;

    /// Draw one or more (name, x, y, color) series as a line, scatter,
    /// bar, or area chart onto an already-created drawing area.
    fn draw_xy_chart<DB: DrawingBackend>(
        drawing_area: &DrawingArea<DB, Shift>,
        settings: &PlotSettings,
        series: &[(String, Vec<f64>, Vec<f64>, (u8, u8, u8))],
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        if series.is_empty() || series.iter().all(|(_, x, _, _)| x.is_empty()) {
            return Err(Error::EmptyData("No data to plot".to_string()));
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for (_, xs, ys, _) in series {
            for &v in xs.iter().filter(|v| v.is_finite()) {
                x_min = x_min.min(v);
                x_max = x_max.max(v);
            }
            for &v in ys.iter().filter(|v| v.is_finite()) {
                y_min = y_min.min(v);
                y_max = y_max.max(v);
            }
        }
        if !x_min.is_finite() || !y_min.is_finite() {
            return Err(Error::EmptyData("No finite data to plot".to_string()));
        }
        // Keep the zero baseline visible for bar/area charts, the same
        // way the SVG bar chart does, instead of letting an all-positive
        // or all-negative series push it off the drawn range.
        if matches!(settings.plot_kind, PlotKind::Bar | PlotKind::Area) {
            y_min = y_min.min(0.0);
            y_max = y_max.max(0.0);
        }

        let x_range = if (x_max - x_min).abs() < f64::EPSILON {
            1.0
        } else {
            x_max - x_min
        };
        let y_range = if (y_max - y_min).abs() < f64::EPSILON {
            1.0
        } else {
            y_max - y_min
        };
        let x_margin = x_range * 0.05;
        let y_margin = y_range * 0.05;

        let mut chart = ChartBuilder::on(drawing_area)
            .caption(&settings.title, ("sans-serif", 24).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (x_min - x_margin)..(x_max + x_margin),
                (y_min - y_margin)..(y_max + y_margin),
            )?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&settings.x_label).y_desc(&settings.y_label);
        if !settings.show_grid {
            mesh.disable_mesh();
        }
        mesh.draw()?;

        for (name, xs, ys, rgb) in series {
            let color = RGBColor(rgb.0, rgb.1, rgb.2);
            let points: Vec<(f64, f64)> = xs
                .iter()
                .zip(ys.iter())
                .map(|(&x, &y)| (x, y))
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .collect();
            if points.is_empty() {
                continue;
            }

            match settings.plot_kind {
                PlotKind::Line => {
                    let handle =
                        chart.draw_series(LineSeries::new(points.iter().copied(), color))?;
                    if settings.show_legend {
                        handle.label(name.clone()).legend(move |(x, y)| {
                            PathElement::new(vec![(x, y), (x + 20, y)], color)
                        });
                    }
                }
                PlotKind::Scatter => {
                    let handle = chart.draw_series(
                        points
                            .iter()
                            .map(|&(x, y)| Circle::new((x, y), 3, color.filled())),
                    )?;
                    if settings.show_legend {
                        handle
                            .label(name.clone())
                            .legend(move |(x, y)| Circle::new((x + 10, y), 3, color.filled()));
                    }
                }
                PlotKind::Bar => {
                    let mut xs_sorted: Vec<f64> = points.iter().map(|&(x, _)| x).collect();
                    xs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let bar_width = if xs_sorted.len() <= 1 {
                        0.5
                    } else {
                        let mut min_diff = f64::INFINITY;
                        for w in xs_sorted.windows(2) {
                            let diff = (w[1] - w[0]).abs();
                            if diff > 0.0 && diff < min_diff {
                                min_diff = diff;
                            }
                        }
                        if min_diff.is_finite() {
                            min_diff * 0.8
                        } else {
                            0.5
                        }
                    };
                    let handle = chart.draw_series(points.iter().map(|&(x, y)| {
                        Rectangle::new(
                            [(x - bar_width / 2.0, 0.0), (x + bar_width / 2.0, y)],
                            color.filled(),
                        )
                    }))?;
                    if settings.show_legend {
                        handle.label(name.clone()).legend(move |(x, y)| {
                            Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled())
                        });
                    }
                }
                PlotKind::Area => {
                    let handle = chart.draw_series(AreaSeries::new(
                        points.iter().copied(),
                        0.0,
                        color.mix(0.2),
                    ))?;
                    if settings.show_legend {
                        handle.label(name.clone()).legend(move |(x, y)| {
                            PathElement::new(vec![(x, y), (x + 20, y)], color)
                        });
                    }
                }
                PlotKind::Histogram | PlotKind::BoxPlot => {
                    return Err(Error::NotImplemented(
                        "use plot_histogram_for_web/plot_boxplot_for_web for this plot kind"
                            .to_string(),
                    ));
                }
            }
        }

        if settings.show_legend {
            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()?;
        }

        Ok(())
    }

    fn numeric_series_for_columns(
        df: &DataFrame,
        columns: &[&str],
        settings: &PlotSettings,
    ) -> Result<Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>> {
        let mut out = Vec::with_capacity(columns.len());
        for (i, &col) in columns.iter().enumerate() {
            let values = df.get_column_numeric_values(col)?;
            let indices: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
            let palette_len = settings.color_palette.len().max(1);
            let color = settings
                .color_palette
                .get(i % palette_len)
                .copied()
                .unwrap_or((70, 130, 180));
            out.push((col.to_string(), indices, values, color));
        }
        Ok(out)
    }

    /// Multi-series line chart (one line per column, sharing a row-index
    /// x-axis) — used for [`VisualizationType::Line`].
    pub fn plot_multi_series_for_web<DB: DrawingBackend>(
        df: &DataFrame,
        columns: &[&str],
        drawing_area: DrawingArea<DB, Shift>,
        settings: &PlotSettings,
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        let series = numeric_series_for_columns(df, columns, settings)?;
        draw_xy_chart(&drawing_area, settings, &series)
    }

    /// Single-column chart (bar or area, per `settings.plot_kind`)
    /// plotted against a row-index x-axis — used for
    /// [`VisualizationType::Bar`] and [`VisualizationType::Area`].
    pub fn plot_column_for_web<DB: DrawingBackend>(
        df: &DataFrame,
        column: &str,
        drawing_area: DrawingArea<DB, Shift>,
        settings: &PlotSettings,
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        let series = numeric_series_for_columns(df, &[column], settings)?;
        draw_xy_chart(&drawing_area, settings, &series)
    }

    /// Scatter plot of two numeric columns — used for
    /// [`VisualizationType::Scatter`].
    pub fn plot_scatter_for_web<DB: DrawingBackend>(
        df: &DataFrame,
        x_column: &str,
        y_column: &str,
        drawing_area: DrawingArea<DB, Shift>,
        settings: &PlotSettings,
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        let x_values = df.get_column_numeric_values(x_column)?;
        let y_values = df.get_column_numeric_values(y_column)?;
        if x_values.len() != y_values.len() {
            return Err(Error::DimensionMismatch(
                "x and y columns must have the same length".to_string(),
            ));
        }
        let color = settings
            .color_palette
            .first()
            .copied()
            .unwrap_or((70, 130, 180));
        let series = vec![(
            format!("{} vs {}", y_column, x_column),
            x_values,
            y_values,
            color,
        )];
        draw_xy_chart(&drawing_area, settings, &series)
    }

    /// Histogram of one numeric column — used for
    /// [`VisualizationType::Histogram`].
    pub fn plot_histogram_for_web<DB: DrawingBackend>(
        df: &DataFrame,
        column: &str,
        bins: usize,
        drawing_area: DrawingArea<DB, Shift>,
        settings: &PlotSettings,
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        let values = df.get_column_numeric_values(column)?;
        let finite: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Err(Error::EmptyData("No data to plot".to_string()));
        }
        if bins == 0 {
            return Err(Error::InvalidInput(
                "Histogram: bins must be greater than 0".to_string(),
            ));
        }

        let min_val = finite.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // A constant column has max_val == min_val; fall back to a
        // single unit-width bin instead of dividing by zero.
        let bin_width = if (max_val - min_val).abs() < f64::EPSILON {
            1.0
        } else {
            (max_val - min_val) / bins as f64
        };
        let mut counts = vec![0usize; bins];
        for &v in &finite {
            let idx = ((v - min_val) / bin_width).floor() as usize;
            counts[idx.min(bins - 1)] += 1;
        }
        let max_freq = *counts.iter().max().unwrap_or(&1) as f64;

        let mut chart = ChartBuilder::on(&drawing_area)
            .caption(&settings.title, ("sans-serif", 24).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                min_val..(max_val + bin_width * 0.1).max(min_val + f64::EPSILON),
                0.0..(max_freq * 1.1).max(1.0),
            )?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&settings.x_label).y_desc("Frequency");
        if !settings.show_grid {
            mesh.disable_mesh();
        }
        mesh.draw()?;

        let color_rgb = settings
            .color_palette
            .first()
            .copied()
            .unwrap_or((70, 130, 180));
        let color = RGBColor(color_rgb.0, color_rgb.1, color_rgb.2);
        chart.draw_series(counts.iter().enumerate().map(|(i, &count)| {
            let x0 = min_val + i as f64 * bin_width;
            let x1 = x0 + bin_width;
            Rectangle::new([(x0, 0.0), (x1, count as f64)], color.mix(0.7).filled())
        }))?;

        Ok(())
    }

    /// Box plot of a numeric column grouped by a category column — used
    /// for [`VisualizationType::BoxPlot`].
    ///
    /// Delegates the statistics and box/whisker/median/outlier drawing
    /// to [`crate::vis::plotters::boxplot_stats`] and
    /// [`crate::vis::plotters::draw_boxplot_series`], the same shared
    /// implementation the PNG/SVG box plot backends use, so a real box
    /// width and real median line are not yet another divergent copy.
    pub fn plot_boxplot_for_web<DB: DrawingBackend>(
        df: &DataFrame,
        category_column: &str,
        value_column: &str,
        drawing_area: DrawingArea<DB, Shift>,
        settings: &PlotSettings,
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        let categories_raw = df.get_column_string_values(category_column)?;
        let values = df.get_column_numeric_values(value_column)?;
        if categories_raw.len() != values.len() {
            return Err(Error::DimensionMismatch(
                "category and value columns must have the same length".to_string(),
            ));
        }

        let mut category_map: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for (cat, val) in categories_raw.into_iter().zip(values.into_iter()) {
            category_map.entry(cat).or_default().push(val);
        }
        let mut categories: Vec<String> = category_map
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        categories.sort();
        if categories.is_empty() {
            return Err(Error::EmptyData("No data to plot".to_string()));
        }

        let all_stats: Vec<crate::vis::plotters::BoxPlotStats> = categories
            .iter()
            .filter_map(|c| category_map.get(c))
            .filter_map(|v| crate::vis::plotters::boxplot_stats(v))
            .collect();
        if all_stats.is_empty() {
            return Err(Error::EmptyData("No finite data to plot".to_string()));
        }
        let y_min = all_stats
            .iter()
            .map(|s| s.min.min(s.whisker_low))
            .fold(f64::INFINITY, f64::min);
        let y_max = all_stats
            .iter()
            .map(|s| s.max.max(s.whisker_high))
            .fold(f64::NEG_INFINITY, f64::max);
        let y_range = if (y_max - y_min).abs() < f64::EPSILON {
            1.0
        } else {
            y_max - y_min
        };
        let y_margin = y_range * 0.1;

        let mut chart = ChartBuilder::on(&drawing_area)
            .caption(&settings.title, ("sans-serif", 24).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                -0.5f64..(categories.len() as f64 - 0.5),
                (y_min - y_margin)..(y_max + y_margin),
            )?;

        let label_formatter = |x: &f64| {
            let i = x.round() as isize;
            if i >= 0 && (i as usize) < categories.len() {
                categories[i as usize].clone()
            } else {
                String::new()
            }
        };
        let mut mesh = chart.configure_mesh();
        mesh.x_labels(categories.len())
            .x_label_formatter(&label_formatter)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label);
        if !settings.show_grid {
            mesh.disable_mesh();
        }
        mesh.draw()?;

        crate::vis::plotters::draw_boxplot_series(
            &mut chart,
            &categories,
            &category_map,
            &settings.color_palette,
        )
    }
}
