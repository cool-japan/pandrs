//! WebAssembly support for interactive browser visualization
//!
//! This module provides WebAssembly (wasm) integration for PandRS, allowing for
//! interactive data visualization in web browsers.

use js_sys::{Array, Function, Object, Reflect};
use plotters_canvas::CanvasBackend;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, Window};

use crate::error::{Error, PandRSError, Result};
use crate::vis::plotters_ext::{PlotKind, PlotSettings};
use crate::DataFrame;
use crate::Series;

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
    data: Option<Rc<RefCell<DataFrame>>>,
    event_listeners: Vec<(String, Closure<dyn FnMut(web_sys::MouseEvent)>)>,
}

#[wasm_bindgen]
impl WebVisualization {
    /// Create a new web visualization
    #[wasm_bindgen(constructor)]
    pub fn new(config: WebVisualizationConfig) -> Result<WebVisualization> {
        // Get the canvas element
        let window =
            web_sys::window().ok_or_else(|| Error::State("No window object available".into()))?;

        let document = window
            .document()
            .ok_or_else(|| Error::State("No document object available".into()))?;

        let canvas = document
            .get_element_by_id(&config.canvas_id)
            .ok_or_else(|| {
                Error::State(format!(
                    "Canvas element with ID '{}' not found",
                    config.canvas_id
                ))
            })?;

        let canvas = canvas
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| Error::State("Element is not a canvas".into()))?;

        // Set canvas dimensions
        canvas.set_width(config.width);
        canvas.set_height(config.height);

        // Get drawing context
        let context = canvas
            .get_context("2d")
            .map_err(|_| Error::State("Failed to get canvas context".into()))?
            .ok_or_else(|| Error::State("Canvas context is null".into()))?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| Error::State("Failed to convert to CanvasRenderingContext2d".into()))?;

        Ok(WebVisualization {
            config,
            canvas,
            context,
            data: None,
            event_listeners: Vec::new(),
        })
    }

    /// Set the DataFrame to visualize
    #[wasm_bindgen]
    pub fn set_data(&mut self, data_json: &str) -> Result<()> {
        // Parse JSON to DataFrame
        let df = DataFrame::from_json(data_json)?;
        self.data = Some(Rc::new(RefCell::new(df)));
        Ok(())
    }

    /// Get the DataFrame being visualized
    pub fn get_data(&self) -> Option<Rc<RefCell<DataFrame>>> {
        self.data.clone()
    }

    /// Render the visualization
    #[wasm_bindgen]
    pub fn render(&mut self) -> Result<()> {
        let df = match &self.data {
            Some(df) => df.borrow(),
            None => return Err(Error::State("No data available to visualize".into())),
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
            return Err(Error::State("DataFrame has no columns".into()));
        }

        // Create a plotters backend with the canvas
        let backend = CanvasBackend::with_canvas_object(self.canvas.clone())
            .map_err(|e| Error::State(format!("Failed to create canvas backend: {}", e)))?;

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
        let mut numeric_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::State(
                "No numeric columns available for line chart".into(),
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
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Line;

        // Use existing plotting functionality from plotters_ext module
        crate::vis::plotters_ext::plot_multi_series_for_web(
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
        let mut numeric_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::State(
                "No numeric columns available for bar chart".into(),
            ));
        }

        // For bar chart, let's just use the first numeric column
        let column = &numeric_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Bar;

        // Use existing plotting functionality
        crate::vis::plotters_ext::plot_column_for_web(df, column, drawing_area, &plot_settings)?;

        Ok(())
    }

    fn render_scatter_chart(
        &self,
        df: &DataFrame,
        backend: CanvasBackend,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Need at least two numeric columns for scatter plot
        let mut numeric_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            }
        }

        if numeric_columns.len() < 2 {
            return Err(Error::State(
                "Need at least two numeric columns for scatter plot".into(),
            ));
        }

        let x_column = &numeric_columns[0];
        let y_column = &numeric_columns[1];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Scatter;

        // Use existing scatter plot functionality
        crate::vis::plotters_ext::plot_scatter_for_web(
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
        let mut numeric_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            }
        }

        if numeric_columns.is_empty() {
            return Err(Error::State(
                "No numeric columns available for area chart".into(),
            ));
        }

        // For area chart, just use the first numeric column
        let column = &numeric_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Area;

        // Use existing plotting functionality
        crate::vis::plotters_ext::plot_column_for_web(df, column, drawing_area, &plot_settings)?;

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
            return Err(Error::State(
                "No numeric columns available for histogram".into(),
            ));
        }

        let column = &numeric_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::Histogram;

        // Use existing histogram functionality
        crate::vis::plotters_ext::plot_histogram_for_web(
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
            return Err(Error::State(
                "Need at least one numeric and one categorical column for boxplot".into(),
            ));
        }

        let value_column = &numeric_columns[0];
        let category_column = &categorical_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

        // Create a temporary plot to set up the chart
        let mut plot_settings = settings.clone();
        plot_settings.plot_kind = PlotKind::BoxPlot;

        // Use existing boxplot functionality
        crate::vis::plotters_ext::plot_boxplot_for_web(
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
            return Err(Error::State(
                "Need at least one numeric and one categorical column for pie chart".into(),
            ));
        }

        let value_column = &numeric_columns[0];
        let category_column = &categorical_columns[0];

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

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
        let mut numeric_columns = Vec::new();

        for col_name in df.column_names() {
            if df.is_numeric_column(&col_name) {
                numeric_columns.push(col_name);
            }
        }

        if numeric_columns.len() < 2 {
            return Err(Error::State(
                "Need at least two numeric columns for heatmap".into(),
            ));
        }

        // Use existing plotters implementation
        let drawing_area = backend.into_drawing_area();
        drawing_area
            .fill(&plotters::style::colors::WHITE)
            .map_err(|e| Error::State(format!("Failed to fill drawing area: {}", e)))?;

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
            return Err(Error::State(
                "Category and value columns must have the same length".into(),
            ));
        }

        // Aggregate values by category
        let mut category_values = std::collections::HashMap::new();
        for (cat, val) in categories.iter().zip(values.iter()) {
            *category_values.entry(cat.clone()).or_insert(0.0) += *val as f64;
        }

        // Calculate total
        let total: f64 = category_values.values().sum();

        // Draw pie chart
        let cx = (self.config.width as f64) / 2.0;
        let cy = (self.config.height as f64) / 2.0;
        let radius = (self.config.width.min(self.config.height) as f64) * 0.4;

        // Draw title
        self.context.set_font("16px sans-serif");
        self.context.set_text_align("center");
        self.context.set_fill_style(&JsValue::from_str("black"));
        self.context
            .fill_text(&settings.title, cx, 30.0)
            .map_err(|_| Error::State("Failed to render text".into()))?;

        // Draw pie slices
        let mut start_angle = 0.0;
        let mut i = 0;
        let colors = settings.color_palette.clone();

        for (category, value) in &category_values {
            let slice_angle = 2.0 * std::f64::consts::PI * (value / total);

            // Choose color from palette
            let color_idx = i % colors.len();
            let (r, g, b) = colors[color_idx];
            let color = format!("rgb({}, {}, {})", r, g, b);

            // Draw slice
            self.context.begin_path();
            self.context.move_to(cx, cy);
            self.context
                .arc(cx, cy, radius, start_angle, start_angle + slice_angle)
                .map_err(|_| Error::State("Failed to draw arc".into()))?;
            self.context.close_path();

            self.context.set_fill_style(&JsValue::from_str(&color));
            self.context.fill();

            // Draw slice outline
            self.context.set_stroke_style(&JsValue::from_str("white"));
            self.context.set_line_width(1.0);
            self.context.stroke();

            // Calculate label position
            let label_angle = start_angle + slice_angle / 2.0;
            let label_x = cx + radius * 0.7 * label_angle.cos();
            let label_y = cy + radius * 0.7 * label_angle.sin();

            // Draw percentage label
            let percentage = (value / total * 100.0).round() / 10.0 * 10.0;
            self.context.set_font("12px sans-serif");
            self.context.set_fill_style(&JsValue::from_str("white"));
            self.context.set_text_align("center");

            if percentage >= 5.0 {
                // Only show label if slice is big enough
                self.context
                    .fill_text(&format!("{}%", percentage), label_x, label_y)
                    .map_err(|_| Error::State("Failed to render text".into()))?;
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
                // Choose color from palette
                let color_idx = i % colors.len();
                let (r, g, b) = colors[color_idx];
                let color = format!("rgb({}, {}, {})", r, g, b);

                // Draw color square
                self.context.set_fill_style(&JsValue::from_str(&color));
                self.context
                    .fill_rect(legend_x, legend_y + y_offset, 15.0, 15.0);

                // Draw category name
                self.context.set_font("12px sans-serif");
                self.context.set_fill_style(&JsValue::from_str("black"));
                self.context.set_text_align("left");

                // Truncate long category names
                let display_cat = if category.len() > 15 {
                    format!("{}...", &category[0..12])
                } else {
                    category.clone()
                };

                self.context
                    .fill_text(&display_cat, legend_x + 20.0, legend_y + y_offset + 12.0)
                    .map_err(|_| Error::State("Failed to render text".into()))?;

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
            return Err(Error::State("No data for heatmap".into()));
        }

        let cols = data_matrix[0].len();
        if cols == 0 {
            return Err(Error::State("Empty columns for heatmap".into()));
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

        // Draw title
        self.context.set_font("16px sans-serif");
        self.context.set_text_align("center");
        self.context.set_fill_style(&JsValue::from_str("black"));
        self.context
            .fill_text(&settings.title, (self.config.width as f64) / 2.0, 30.0)
            .map_err(|_| Error::State("Failed to render text".into()))?;

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
                self.context.set_fill_style(&JsValue::from_str(&color));
                self.context.fill_rect(x, y, cell_width, cell_height);

                // Draw cell outline
                self.context
                    .set_stroke_style(&JsValue::from_str("rgba(255,255,255,0.2)"));
                self.context.set_line_width(0.5);
                self.context.stroke_rect(x, y, cell_width, cell_height);

                // Draw value text if cells are large enough
                if cell_width > 30.0 && cell_height > 20.0 {
                    self.context.set_font("10px sans-serif");
                    self.context.set_fill_style(&JsValue::from_str("white"));
                    self.context.set_text_align("center");

                    self.context
                        .fill_text(
                            &format!("{:.1}", val),
                            x + cell_width / 2.0,
                            y + cell_height / 2.0 + 3.0,
                        )
                        .map_err(|_| Error::State("Failed to render text".into()))?;
                }
            }
        }

        // Draw column labels
        self.context.set_font("12px sans-serif");
        self.context.set_fill_style(&JsValue::from_str("black"));
        self.context.set_text_align("center");

        for (j, col) in columns.iter().enumerate().take(cols) {
            let x = margin + j as f64 * cell_width + cell_width / 2.0;
            let y = margin - 10.0;

            // Truncate long column names
            let display_name = if col.len() > 10 {
                format!("{}...", &col[0..7])
            } else {
                col.clone()
            };

            self.context
                .fill_text(&display_name, x, y)
                .map_err(|_| Error::State("Failed to render text".into()))?;
        }

        // Draw row labels (use row indices)
        self.context.set_text_align("right");

        for i in 0..rows {
            let x = margin - 10.0;
            let y = margin + i as f64 * cell_height + cell_height / 2.0 + 5.0;

            self.context
                .fill_text(&format!("Row {}", i + 1), x, y)
                .map_err(|_| Error::State("Failed to render text".into()))?;
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

            self.context.set_fill_style(&JsValue::from_str(&color));
            self.context.fill_rect(
                scale_x - scale_width / 2.0,
                scale_y + (1.0 - normalized) * scale_height,
                scale_width,
                scale_height / 100.0,
            );
        }

        // Draw scale labels
        self.context.set_font("10px sans-serif");
        self.context.set_fill_style(&JsValue::from_str("black"));
        self.context.set_text_align("center");

        self.context
            .fill_text(&format!("{:.1}", max_val), scale_x, scale_y - 5.0)
            .map_err(|_| Error::State("Failed to render text".into()))?;

        self.context
            .fill_text(
                &format!("{:.1}", min_val),
                scale_x,
                scale_y + scale_height + 15.0,
            )
            .map_err(|_| Error::State("Failed to render text".into()))?;

        Ok(())
    }

    // Set up interactive tooltips
    fn setup_tooltip_listeners(&mut self) -> Result<()> {
        // Clean up any existing event listeners
        for (event, listener) in self.event_listeners.drain(..) {
            self.canvas
                .remove_event_listener_with_callback(&event, listener.as_ref().unchecked_ref())
                .map_err(|_| Error::State("Failed to remove event listener".into()))?;
        }

        // Add mousemove listener for tooltips
        let canvas_clone = self.canvas.clone();
        let context_clone = self.context.clone();
        let config_clone = self.config.clone();
        let data_clone = self.data.clone();

        let mousemove_callback = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
            // Get mouse position relative to canvas
            let rect = canvas_clone.get_bounding_client_rect();
            let x = event.client_x() as f64 - rect.left();
            let y = event.client_y() as f64 - rect.top();

            // Handle tooltip logic based on visualization type
            if let Some(data) = &data_clone {
                let df = data.borrow();

                // Tooltip logic would go here based on chart type
                // This is a simplified placeholder

                // Clear previous tooltip
                context_clone.clear_rect(
                    0.0,
                    0.0,
                    config_clone.width as f64,
                    config_clone.height as f64,
                );

                // Draw tooltip
                context_clone.set_fill_style(&JsValue::from_str("rgba(0,0,0,0.7)"));
                context_clone.fill_rect(x + 10.0, y - 20.0, 100.0, 40.0);

                context_clone.set_font("12px sans-serif");
                context_clone.set_fill_style(&JsValue::from_str("white"));
                context_clone.fill_text("Tooltip", x + 15.0, y).unwrap();
            }
        }) as Box<dyn FnMut(_)>);

        // Add the event listener
        self.canvas
            .add_event_listener_with_callback(
                "mousemove",
                mousemove_callback.as_ref().unchecked_ref(),
            )
            .map_err(|_| Error::State("Failed to add event listener".into()))?;

        // Store the listener to avoid it being dropped
        self.event_listeners
            .push(("mousemove".to_string(), mousemove_callback));

        Ok(())
    }

    /// Export the visualization as an image
    #[wasm_bindgen]
    pub fn export_image(&self) -> Result<String> {
        self.canvas
            .to_data_url()
            .map_err(|_| Error::State("Failed to export image".into()))
    }

    /// Update title
    #[wasm_bindgen]
    pub fn update_title(&mut self, title: &str) -> Result<()> {
        self.config.title = title.to_string();
        self.render()
    }

    /// Change visualization type
    #[wasm_bindgen]
    pub fn change_type(&mut self, viz_type: VisualizationType) -> Result<()> {
        self.config.viz_type = viz_type;
        self.render()
    }

    /// Update theme
    #[wasm_bindgen]
    pub fn update_theme(&mut self, theme: ColorTheme) -> Result<()> {
        self.config.theme = theme;
        self.render()
    }
}

// Add the extension methods to the plotters_ext module
mod plotters_ext_web {
    use super::*;
    use crate::vis::plotters_ext::PlotSettings;
    use plotters::prelude::*;

    /// Implementation of multi-series plotting for web
    pub fn plot_multi_series_for_web(
        df: &DataFrame,
        columns: &[&str],
        drawing_area: DrawingArea<plotters_canvas::CanvasBackend, plotters::coord::Shift>,
        settings: &PlotSettings,
    ) -> Result<()> {
        // This is a placeholder that would be implemented similar to the
        // PNG/SVG versions in plotters_ext.rs
        Ok(())
    }

    /// Implementation of column plotting for web
    pub fn plot_column_for_web(
        df: &DataFrame,
        column: &str,
        drawing_area: DrawingArea<plotters_canvas::CanvasBackend, plotters::coord::Shift>,
        settings: &PlotSettings,
    ) -> Result<()> {
        // This is a placeholder that would be implemented similar to the
        // PNG/SVG versions in plotters_ext.rs
        Ok(())
    }

    /// Implementation of scatter plotting for web
    pub fn plot_scatter_for_web(
        df: &DataFrame,
        x_column: &str,
        y_column: &str,
        drawing_area: DrawingArea<plotters_canvas::CanvasBackend, plotters::coord::Shift>,
        settings: &PlotSettings,
    ) -> Result<()> {
        // This is a placeholder that would be implemented similar to the
        // PNG/SVG versions in plotters_ext.rs
        Ok(())
    }

    /// Implementation of histogram plotting for web
    pub fn plot_histogram_for_web(
        df: &DataFrame,
        column: &str,
        bins: usize,
        drawing_area: DrawingArea<plotters_canvas::CanvasBackend, plotters::coord::Shift>,
        settings: &PlotSettings,
    ) -> Result<()> {
        // This is a placeholder that would be implemented similar to the
        // PNG/SVG versions in plotters_ext.rs
        Ok(())
    }

    /// Implementation of box plotting for web
    pub fn plot_boxplot_for_web(
        df: &DataFrame,
        category_column: &str,
        value_column: &str,
        drawing_area: DrawingArea<plotters_canvas::CanvasBackend, plotters::coord::Shift>,
        settings: &PlotSettings,
    ) -> Result<()> {
        // This is a placeholder that would be implemented similar to the
        // PNG/SVG versions in plotters_ext.rs
        Ok(())
    }
}
