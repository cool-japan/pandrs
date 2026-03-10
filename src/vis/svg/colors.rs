//! Color schemes and color utilities for SVG visualization

/// RGBA color representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create a color from RGB values (fully opaque)
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a color from RGBA values
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create a color from a hex string like "#RRGGBB" or "#RRGGBBAA"
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some(Self::rgba(r, g, b, a))
            }
            _ => None,
        }
    }

    /// Convert to hex string (e.g. "#RRGGBB")
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Convert to CSS rgba string
    pub fn to_css_rgba(&self) -> String {
        let alpha = self.a as f64 / 255.0;
        format!("rgba({},{},{},{:.3})", self.r, self.g, self.b, alpha)
    }

    /// Convert to SVG fill/stroke attribute string
    pub fn to_svg_color(&self) -> String {
        if self.a == 255 {
            self.to_hex()
        } else {
            self.to_hex()
        }
    }

    /// Get opacity as a 0.0–1.0 float
    pub fn opacity(&self) -> f64 {
        self.a as f64 / 255.0
    }

    /// Interpolate between two colors at position t in [0.0, 1.0]
    pub fn interpolate(&self, other: &Color, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: u8, b: u8| -> u8 { (a as f64 + (b as f64 - a as f64) * t).round() as u8 };
        Color {
            r: lerp(self.r, other.r),
            g: lerp(self.g, other.g),
            b: lerp(self.b, other.b),
            a: lerp(self.a, other.a),
        }
    }

    /// Pre-defined colors
    pub const WHITE: Color = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const RED: Color = Color {
        r: 220,
        g: 50,
        b: 47,
        a: 255,
    };
    pub const GREEN: Color = Color {
        r: 42,
        g: 161,
        b: 152,
        a: 255,
    };
    pub const BLUE: Color = Color {
        r: 38,
        g: 139,
        b: 210,
        a: 255,
    };
    pub const ORANGE: Color = Color {
        r: 203,
        g: 75,
        b: 22,
        a: 255,
    };
    pub const PURPLE: Color = Color {
        r: 108,
        g: 113,
        b: 196,
        a: 255,
    };
    pub const YELLOW: Color = Color {
        r: 181,
        g: 137,
        b: 0,
        a: 255,
    };
    pub const GRAY: Color = Color {
        r: 147,
        g: 161,
        b: 161,
        a: 255,
    };
    pub const LIGHT_GRAY: Color = Color {
        r: 238,
        g: 232,
        b: 213,
        a: 255,
    };
    pub const DARK_GRAY: Color = Color {
        r: 88,
        g: 110,
        b: 117,
        a: 255,
    };
}

impl Default for Color {
    fn default() -> Self {
        Self::BLUE
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Gradient with multiple color stops
#[derive(Debug, Clone)]
pub struct ColorGradient {
    stops: Vec<(f64, Color)>,
}

impl ColorGradient {
    /// Create a gradient from a list of (position, color) stops.
    /// Positions should be in [0.0, 1.0] and sorted ascending.
    pub fn new(stops: Vec<(f64, Color)>) -> Self {
        let mut stops = stops;
        stops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { stops }
    }

    /// Create a two-stop gradient
    pub fn two_color(start: Color, end: Color) -> Self {
        Self::new(vec![(0.0, start), (1.0, end)])
    }

    /// Sample the gradient at position t in [0.0, 1.0]
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        if self.stops.is_empty() {
            return Color::BLACK;
        }
        if self.stops.len() == 1 {
            return self.stops[0].1;
        }
        // Find the surrounding stops
        let first = &self.stops[0];
        let last = self.stops.last().expect("non-empty stops");
        if t <= first.0 {
            return first.1;
        }
        if t >= last.0 {
            return last.1;
        }
        for i in 0..self.stops.len() - 1 {
            let (t0, c0) = self.stops[i];
            let (t1, c1) = self.stops[i + 1];
            if t >= t0 && t <= t1 {
                let local_t = if (t1 - t0).abs() < f64::EPSILON {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };
                return c0.interpolate(&c1, local_t);
            }
        }
        last.1
    }
}

/// Named color schemes for charts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorScheme {
    /// Default PandRS color palette
    #[default]
    Default,
    /// Blue shades
    Blues,
    /// Green shades
    Greens,
    /// Orange shades
    Oranges,
    /// Viridis perceptually uniform
    Viridis,
    /// Plasma perceptually uniform
    Plasma,
    /// Categorical palette for distinct series
    Categorical,
}

impl ColorScheme {
    /// Get a set of colors appropriate for categorical (discrete) use
    pub fn categorical_colors(&self) -> Vec<Color> {
        match self {
            ColorScheme::Default | ColorScheme::Categorical => vec![
                Color::rgb(70, 130, 180), // steel blue
                Color::rgb(220, 100, 60), // orange-red
                Color::rgb(60, 160, 100), // green
                Color::rgb(180, 80, 180), // purple
                Color::rgb(200, 170, 40), // gold
                Color::rgb(60, 180, 200), // cyan
                Color::rgb(200, 80, 100), // rose
                Color::rgb(100, 150, 50), // olive
            ],
            ColorScheme::Blues => vec![
                Color::rgb(198, 219, 239),
                Color::rgb(158, 202, 225),
                Color::rgb(107, 174, 214),
                Color::rgb(66, 146, 198),
                Color::rgb(33, 113, 181),
                Color::rgb(8, 81, 156),
                Color::rgb(8, 48, 107),
            ],
            ColorScheme::Greens => vec![
                Color::rgb(199, 233, 192),
                Color::rgb(161, 217, 155),
                Color::rgb(116, 196, 118),
                Color::rgb(65, 171, 93),
                Color::rgb(35, 139, 69),
                Color::rgb(0, 109, 44),
                Color::rgb(0, 68, 27),
            ],
            ColorScheme::Oranges => vec![
                Color::rgb(253, 224, 182),
                Color::rgb(253, 198, 118),
                Color::rgb(253, 159, 56),
                Color::rgb(241, 105, 19),
                Color::rgb(217, 72, 1),
                Color::rgb(166, 54, 3),
                Color::rgb(127, 39, 4),
            ],
            ColorScheme::Viridis => viridis_colors(),
            ColorScheme::Plasma => plasma_colors(),
        }
    }

    /// Get the color at index, cycling through the palette
    pub fn color_at(&self, index: usize) -> Color {
        let colors = self.categorical_colors();
        if colors.is_empty() {
            return Color::BLUE;
        }
        colors[index % colors.len()]
    }

    /// Get a continuous gradient for this scheme (for heatmaps etc.)
    pub fn gradient(&self) -> ColorGradient {
        match self {
            ColorScheme::Blues => ColorGradient::new(vec![
                (0.0, Color::rgb(239, 243, 255)),
                (1.0, Color::rgb(8, 48, 107)),
            ]),
            ColorScheme::Greens => ColorGradient::new(vec![
                (0.0, Color::rgb(247, 252, 245)),
                (1.0, Color::rgb(0, 68, 27)),
            ]),
            ColorScheme::Oranges => ColorGradient::new(vec![
                (0.0, Color::rgb(255, 245, 235)),
                (1.0, Color::rgb(127, 39, 4)),
            ]),
            ColorScheme::Viridis => {
                let stops: Vec<(f64, Color)> = viridis_colors()
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let colors = viridis_colors();
                        (i as f64 / (colors.len() - 1).max(1) as f64, c)
                    })
                    .collect();
                ColorGradient::new(stops)
            }
            ColorScheme::Plasma => {
                let stops: Vec<(f64, Color)> = plasma_colors()
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let colors = plasma_colors();
                        (i as f64 / (colors.len() - 1).max(1) as f64, c)
                    })
                    .collect();
                ColorGradient::new(stops)
            }
            ColorScheme::Default | ColorScheme::Categorical => {
                ColorGradient::two_color(Color::rgb(173, 216, 230), Color::rgb(0, 0, 139))
            }
        }
    }
}

fn viridis_colors() -> Vec<Color> {
    vec![
        Color::rgb(68, 1, 84),
        Color::rgb(72, 40, 120),
        Color::rgb(62, 74, 137),
        Color::rgb(49, 104, 142),
        Color::rgb(38, 130, 142),
        Color::rgb(31, 158, 137),
        Color::rgb(53, 183, 121),
        Color::rgb(110, 206, 88),
        Color::rgb(181, 222, 43),
        Color::rgb(253, 231, 37),
    ]
}

fn plasma_colors() -> Vec<Color> {
    vec![
        Color::rgb(13, 8, 135),
        Color::rgb(75, 3, 161),
        Color::rgb(125, 3, 168),
        Color::rgb(168, 34, 150),
        Color::rgb(203, 70, 121),
        Color::rgb(229, 107, 93),
        Color::rgb(248, 148, 65),
        Color::rgb(253, 195, 40),
        Color::rgb(240, 249, 33),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_hex_roundtrip() {
        let c = Color::rgb(100, 150, 200);
        let hex = c.to_hex();
        let parsed = Color::from_hex(&hex).expect("parse hex");
        assert_eq!(parsed.r, c.r);
        assert_eq!(parsed.g, c.g);
        assert_eq!(parsed.b, c.b);
    }

    #[test]
    fn test_color_interpolate() {
        let black = Color::rgb(0, 0, 0);
        let white = Color::rgb(255, 255, 255);
        let mid = black.interpolate(&white, 0.5);
        assert!(mid.r > 120 && mid.r < 140);
    }

    #[test]
    fn test_gradient_sample() {
        let grad = ColorGradient::two_color(Color::rgb(0, 0, 0), Color::rgb(255, 255, 255));
        let mid = grad.sample(0.5);
        assert!(mid.r > 120 && mid.r < 140);
        let start = grad.sample(0.0);
        assert_eq!(start.r, 0);
        let end = grad.sample(1.0);
        assert_eq!(end.r, 255);
    }

    #[test]
    fn test_color_scheme_categorical() {
        let scheme = ColorScheme::Default;
        let colors = scheme.categorical_colors();
        assert!(!colors.is_empty());
        // color_at should cycle
        let c0 = scheme.color_at(0);
        let c_wrap = scheme.color_at(colors.len());
        assert_eq!(c0, c_wrap);
    }
}
