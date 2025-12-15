//! Sparkline visualization for inline mini charts
//!
//! Sparklines are small, inline charts that can be embedded in text or tables.

use super::Chart;

/// Sparkline style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparklineStyle {
    /// Block characters (▁▂▃▄▅▆▇█)
    Block,
    /// Line characters
    Line,
    /// Dot characters
    Dot,
}

impl Default for SparklineStyle {
    fn default() -> Self {
        Self::Block
    }
}

/// Sparkline - a compact inline chart
#[derive(Debug, Clone)]
pub struct Sparkline {
    /// Data values
    values: Vec<f64>,
    /// Style
    style: SparklineStyle,
    /// Custom minimum value (None = auto)
    min: Option<f64>,
    /// Custom maximum value (None = auto)
    max: Option<f64>,
}

impl Sparkline {
    /// Block characters for sparkline (8 levels)
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    /// Create a new sparkline
    pub fn new(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
            style: SparklineStyle::default(),
            min: None,
            max: None,
        }
    }

    /// Set sparkline style
    pub fn with_style(mut self, style: SparklineStyle) -> Self {
        self.style = style;
        self
    }

    /// Set custom minimum value
    pub fn with_min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set custom maximum value
    pub fn with_max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set custom range
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Get sparkline as a single string
    pub fn to_string_compact(&self) -> String {
        self.render()
    }

    /// Get sparkline with stats
    pub fn to_string_with_stats(&self) -> String {
        if self.values.is_empty() {
            return String::from("(empty)");
        }

        let min = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = self.values.iter().sum();
        let mean = sum / self.values.len() as f64;

        format!(
            "{} (min: {:.2}, max: {:.2}, avg: {:.2})",
            self.render(),
            min,
            max,
            mean
        )
    }
}

impl Chart for Sparkline {
    fn render(&self) -> String {
        if self.values.is_empty() {
            return String::new();
        }

        let min = self
            .min
            .unwrap_or_else(|| self.values.iter().cloned().fold(f64::INFINITY, f64::min));
        let max = self.max.unwrap_or_else(|| {
            self.values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        });

        let range = if (max - min).abs() < f64::EPSILON {
            1.0
        } else {
            max - min
        };

        match self.style {
            SparklineStyle::Block => self.render_block(min, range),
            SparklineStyle::Line => self.render_line(min, range),
            SparklineStyle::Dot => self.render_dot(min, range),
        }
    }
}

impl Sparkline {
    fn render_block(&self, min: f64, range: f64) -> String {
        self.values
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                let idx = (normalized * 7.0).round() as usize;
                Self::BLOCKS[idx.min(7)]
            })
            .collect()
    }

    fn render_line(&self, min: f64, range: f64) -> String {
        const LINE_CHARS: [char; 4] = ['_', '⎽', '⎻', '⎺'];
        self.values
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                let idx = (normalized * 3.0).round() as usize;
                LINE_CHARS[idx.min(3)]
            })
            .collect()
    }

    fn render_dot(&self, min: f64, range: f64) -> String {
        const DOT_CHARS: [char; 4] = ['⠁', '⠂', '⠄', '⠆'];
        self.values
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                let idx = (normalized * 3.0).round() as usize;
                DOT_CHARS[idx.min(3)]
            })
            .collect()
    }
}

/// Create sparklines for multiple series
pub struct MultiSparkline {
    /// Series data
    series: Vec<(String, Vec<f64>)>,
}

impl MultiSparkline {
    /// Create new multi-sparkline
    pub fn new() -> Self {
        Self { series: Vec::new() }
    }

    /// Add a series
    pub fn add_series(&mut self, name: &str, values: &[f64]) {
        self.series.push((name.to_string(), values.to_vec()));
    }

    /// Render all series
    pub fn render(&self) -> String {
        let mut output = String::new();
        let max_name_len = self
            .series
            .iter()
            .map(|(name, _)| name.len())
            .max()
            .unwrap_or(0);

        for (name, values) in &self.series {
            let spark = Sparkline::new(values);
            output.push_str(&format!(
                "{:>width$}: {}\n",
                name,
                spark.to_string_with_stats(),
                width = max_name_len
            ));
        }

        output
    }
}

impl Default for MultiSparkline {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a sparkline string
pub fn sparkline(data: &[f64]) -> String {
    Sparkline::new(data).render()
}

/// Convenience function to create a sparkline with stats
pub fn sparkline_with_stats(data: &[f64]) -> String {
    Sparkline::new(data).to_string_with_stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparkline_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spark = Sparkline::new(&data);
        let result = spark.render();
        assert_eq!(result.chars().count(), 5);
        assert!(result.contains('▁') || result.contains('█'));
    }

    #[test]
    fn test_sparkline_empty() {
        let data: Vec<f64> = vec![];
        let spark = Sparkline::new(&data);
        let result = spark.render();
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparkline_constant() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let spark = Sparkline::new(&data);
        let result = spark.render();
        // All values should be at middle level
        assert_eq!(result.chars().count(), 4);
    }

    #[test]
    fn test_sparkline_with_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spark = Sparkline::new(&data);
        let result = spark.to_string_with_stats();
        assert!(result.contains("min:"));
        assert!(result.contains("max:"));
        assert!(result.contains("avg:"));
    }

    #[test]
    fn test_sparkline_custom_range() {
        let data = vec![5.0, 6.0, 7.0];
        let spark = Sparkline::new(&data).with_range(0.0, 10.0);
        let result = spark.render();
        assert_eq!(result.chars().count(), 3);
    }

    #[test]
    fn test_sparkline_styles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let block = Sparkline::new(&data).with_style(SparklineStyle::Block);
        assert!(!block.render().is_empty());

        let line = Sparkline::new(&data).with_style(SparklineStyle::Line);
        assert!(!line.render().is_empty());

        let dot = Sparkline::new(&data).with_style(SparklineStyle::Dot);
        assert!(!dot.render().is_empty());
    }

    #[test]
    fn test_multi_sparkline() {
        let mut multi = MultiSparkline::new();
        multi.add_series("Series A", &[1.0, 2.0, 3.0]);
        multi.add_series("Series B", &[3.0, 2.0, 1.0]);
        let result = multi.render();
        assert!(result.contains("Series A"));
        assert!(result.contains("Series B"));
    }

    #[test]
    fn test_sparkline_convenience() {
        let data = vec![1.0, 2.0, 3.0];
        let result = sparkline(&data);
        assert_eq!(result.chars().count(), 3);

        let result_with_stats = sparkline_with_stats(&data);
        assert!(result_with_stats.contains("min:"));
    }
}
