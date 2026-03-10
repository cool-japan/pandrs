//! Pure-Rust SVG canvas and element builder
//!
//! Provides a programmatic API to build valid SVG XML documents
//! without any external dependencies.

use crate::vis::svg::colors::Color;

/// Represents a single SVG attribute key-value pair
#[derive(Debug, Clone)]
struct Attr {
    key: String,
    value: String,
}

impl Attr {
    fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// Represents an SVG element node
#[derive(Debug, Clone)]
pub struct SvgElement {
    tag: String,
    attrs: Vec<Attr>,
    children: Vec<SvgNode>,
    text_content: Option<String>,
}

impl SvgElement {
    fn new(tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            attrs: Vec::new(),
            children: Vec::new(),
            text_content: None,
        }
    }

    fn attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attrs.push(Attr::new(key, value));
        self
    }

    fn child(mut self, node: SvgNode) -> Self {
        self.children.push(node);
        self
    }

    fn text(mut self, content: impl Into<String>) -> Self {
        self.text_content = Some(content.into());
        self
    }

    fn render(&self) -> String {
        let mut out = String::new();
        out.push('<');
        out.push_str(&escape_tag(&self.tag));
        for attr in &self.attrs {
            out.push(' ');
            out.push_str(&escape_tag(&attr.key));
            out.push_str("=\"");
            out.push_str(&escape_attr(&attr.value));
            out.push('"');
        }
        if self.children.is_empty() && self.text_content.is_none() {
            out.push_str("/>");
        } else {
            out.push('>');
            if let Some(ref text) = self.text_content {
                out.push_str(&escape_text(text));
            }
            for child in &self.children {
                out.push_str(&child.render());
            }
            out.push_str("</");
            out.push_str(&escape_tag(&self.tag));
            out.push('>');
        }
        out
    }
}

/// An SVG node — either an element or raw text
#[derive(Debug, Clone)]
enum SvgNode {
    Element(SvgElement),
    Raw(String),
}

impl SvgNode {
    fn render(&self) -> String {
        match self {
            SvgNode::Element(e) => e.render(),
            SvgNode::Raw(s) => s.clone(),
        }
    }
}

fn escape_tag(s: &str) -> String {
    // Tag names should not need escaping, but strip dangerous chars
    s.replace(['<', '>', '"', '\'', '&'], "")
}

fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Transform specification for SVG elements
#[derive(Debug, Clone, Default)]
pub struct Transform {
    ops: Vec<String>,
}

impl Transform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn translate(mut self, tx: f64, ty: f64) -> Self {
        self.ops.push(format!("translate({:.3},{:.3})", tx, ty));
        self
    }

    pub fn scale(mut self, sx: f64, sy: f64) -> Self {
        self.ops.push(format!("scale({:.3},{:.3})", sx, sy));
        self
    }

    pub fn rotate(mut self, angle_deg: f64) -> Self {
        self.ops.push(format!("rotate({:.3})", angle_deg));
        self
    }

    pub fn rotate_around(mut self, angle_deg: f64, cx: f64, cy: f64) -> Self {
        self.ops
            .push(format!("rotate({:.3},{:.3},{:.3})", angle_deg, cx, cy));
        self
    }

    pub fn to_string(&self) -> String {
        self.ops.join(" ")
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Style options for SVG drawing primitives
#[derive(Debug, Clone)]
pub struct DrawStyle {
    pub fill: Option<Color>,
    pub fill_opacity: f64,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub stroke_opacity: f64,
    pub stroke_dasharray: Option<String>,
    pub font_size: f64,
    pub font_family: String,
    pub font_weight: String,
    pub text_anchor: String,
    pub dominant_baseline: String,
}

impl Default for DrawStyle {
    fn default() -> Self {
        Self {
            fill: Some(Color::BLUE),
            fill_opacity: 1.0,
            stroke: None,
            stroke_width: 1.0,
            stroke_opacity: 1.0,
            stroke_dasharray: None,
            font_size: 12.0,
            font_family: "Arial, Helvetica, sans-serif".to_string(),
            font_weight: "normal".to_string(),
            text_anchor: "middle".to_string(),
            dominant_baseline: "auto".to_string(),
        }
    }
}

impl DrawStyle {
    pub fn no_fill() -> Self {
        Self {
            fill: None,
            ..Default::default()
        }
    }

    pub fn with_fill(color: Color) -> Self {
        Self {
            fill: Some(color),
            stroke: None,
            ..Default::default()
        }
    }

    pub fn with_stroke(color: Color, width: f64) -> Self {
        Self {
            fill: None,
            stroke: Some(color),
            stroke_width: width,
            ..Default::default()
        }
    }

    fn apply_to_attrs(&self, mut elem: SvgElement) -> SvgElement {
        let fill_str = match &self.fill {
            Some(c) => c.to_hex(),
            None => "none".to_string(),
        };
        elem = elem.attr("fill", &fill_str);
        if self.fill_opacity < 1.0 - f64::EPSILON {
            elem = elem.attr("fill-opacity", format!("{:.3}", self.fill_opacity));
        }
        let stroke_str = match &self.stroke {
            Some(c) => c.to_hex(),
            None => "none".to_string(),
        };
        elem = elem.attr("stroke", &stroke_str);
        if self.stroke.is_some() {
            elem = elem.attr("stroke-width", format!("{:.2}", self.stroke_width));
            if self.stroke_opacity < 1.0 - f64::EPSILON {
                elem = elem.attr("stroke-opacity", format!("{:.3}", self.stroke_opacity));
            }
        }
        if let Some(ref dash) = self.stroke_dasharray {
            elem = elem.attr("stroke-dasharray", dash);
        }
        elem
    }
}

/// An SVG group (`<g>`) that accumulates child elements
#[derive(Debug, Clone)]
pub struct SvgGroup {
    children: Vec<SvgNode>,
    transform: Option<Transform>,
    id: Option<String>,
    class: Option<String>,
    clip_path: Option<String>,
}

impl SvgGroup {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            transform: None,
            id: None,
            class: None,
            clip_path: None,
        }
    }

    pub fn with_transform(mut self, t: Transform) -> Self {
        self.transform = Some(t);
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_class(mut self, class: impl Into<String>) -> Self {
        self.class = Some(class.into());
        self
    }

    pub fn with_clip_path(mut self, clip: impl Into<String>) -> Self {
        self.clip_path = Some(clip.into());
        self
    }

    fn to_element(self) -> SvgElement {
        let mut elem = SvgElement::new("g");
        if let Some(ref t) = self.transform {
            if !t.is_empty() {
                elem = elem.attr("transform", t.to_string());
            }
        }
        if let Some(ref id) = self.id {
            elem = elem.attr("id", id);
        }
        if let Some(ref class) = self.class {
            elem = elem.attr("class", class);
        }
        if let Some(ref clip) = self.clip_path {
            elem = elem.attr("clip-path", format!("url(#{})", clip));
        }
        for child in self.children {
            elem = elem.child(child);
        }
        elem
    }
}

impl Default for SvgGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Defs section — stores gradients, clip paths, etc.
#[derive(Debug, Clone, Default)]
pub struct SvgDefs {
    items: Vec<SvgNode>,
}

impl SvgDefs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_linear_gradient(
        &mut self,
        id: &str,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stops: &[(f64, Color)],
    ) {
        let mut grad = SvgElement::new("linearGradient")
            .attr("id", id)
            .attr("x1", format!("{:.1}%", x1 * 100.0))
            .attr("y1", format!("{:.1}%", y1 * 100.0))
            .attr("x2", format!("{:.1}%", x2 * 100.0))
            .attr("y2", format!("{:.1}%", y2 * 100.0));
        for (pos, color) in stops {
            let stop = SvgElement::new("stop")
                .attr("offset", format!("{:.1}%", pos * 100.0))
                .attr(
                    "style",
                    format!(
                        "stop-color:{};stop-opacity:{:.3}",
                        color.to_hex(),
                        color.opacity()
                    ),
                );
            grad = grad.child(SvgNode::Element(stop));
        }
        self.items.push(SvgNode::Element(grad));
    }

    pub fn add_clip_rect(&mut self, id: &str, x: f64, y: f64, width: f64, height: f64) {
        let rect = SvgElement::new("rect")
            .attr("x", format!("{:.2}", x))
            .attr("y", format!("{:.2}", y))
            .attr("width", format!("{:.2}", width))
            .attr("height", format!("{:.2}", height));
        let clip = SvgElement::new("clipPath")
            .attr("id", id)
            .child(SvgNode::Element(rect));
        self.items.push(SvgNode::Element(clip));
    }

    fn to_element(self) -> SvgElement {
        let mut defs = SvgElement::new("defs");
        for item in self.items {
            defs = defs.child(item);
        }
        defs
    }
}

/// Main SVG canvas — accumulates elements and emits SVG XML
#[derive(Debug, Clone)]
pub struct SvgCanvas {
    width: f64,
    height: f64,
    defs: SvgDefs,
    nodes: Vec<SvgNode>,
    background: Option<Color>,
    view_box: Option<(f64, f64, f64, f64)>,
}

impl SvgCanvas {
    /// Create a new canvas with given pixel dimensions
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            defs: SvgDefs::new(),
            nodes: Vec::new(),
            background: None,
            view_box: None,
        }
    }

    /// Set a background color for the whole canvas
    pub fn set_background(&mut self, color: Color) {
        self.background = Some(color);
    }

    /// Set a custom viewBox
    pub fn set_view_box(&mut self, x: f64, y: f64, w: f64, h: f64) {
        self.view_box = Some((x, y, w, h));
    }

    /// Access defs for adding gradients and clip paths
    pub fn defs_mut(&mut self) -> &mut SvgDefs {
        &mut self.defs
    }

    /// Add a group to the canvas
    pub fn group(&mut self, group: SvgGroup) {
        self.nodes.push(SvgNode::Element(group.to_element()));
    }

    /// Draw a rectangle
    pub fn rect(&mut self, x: f64, y: f64, width: f64, height: f64, style: &DrawStyle) {
        let mut elem = SvgElement::new("rect")
            .attr("x", format!("{:.2}", x))
            .attr("y", format!("{:.2}", y))
            .attr("width", format!("{:.2}", width))
            .attr("height", format!("{:.2}", height));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw a rectangle with corner radius
    pub fn rect_rounded(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        rx: f64,
        ry: f64,
        style: &DrawStyle,
    ) {
        let mut elem = SvgElement::new("rect")
            .attr("x", format!("{:.2}", x))
            .attr("y", format!("{:.2}", y))
            .attr("width", format!("{:.2}", width))
            .attr("height", format!("{:.2}", height))
            .attr("rx", format!("{:.2}", rx))
            .attr("ry", format!("{:.2}", ry));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw a line
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, style: &DrawStyle) {
        let mut elem = SvgElement::new("line")
            .attr("x1", format!("{:.2}", x1))
            .attr("y1", format!("{:.2}", y1))
            .attr("x2", format!("{:.2}", x2))
            .attr("y2", format!("{:.2}", y2));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw a circle
    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, style: &DrawStyle) {
        let mut elem = SvgElement::new("circle")
            .attr("cx", format!("{:.2}", cx))
            .attr("cy", format!("{:.2}", cy))
            .attr("r", format!("{:.2}", r));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw an ellipse
    pub fn ellipse(&mut self, cx: f64, cy: f64, rx: f64, ry: f64, style: &DrawStyle) {
        let mut elem = SvgElement::new("ellipse")
            .attr("cx", format!("{:.2}", cx))
            .attr("cy", format!("{:.2}", cy))
            .attr("rx", format!("{:.2}", rx))
            .attr("ry", format!("{:.2}", ry));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw a polyline (open path)
    pub fn polyline(&mut self, points: &[(f64, f64)], style: &DrawStyle) {
        if points.is_empty() {
            return;
        }
        let pts: Vec<String> = points
            .iter()
            .map(|(x, y)| format!("{:.2},{:.2}", x, y))
            .collect();
        let mut elem = SvgElement::new("polyline").attr("points", pts.join(" "));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw a polygon (closed path)
    pub fn polygon(&mut self, points: &[(f64, f64)], style: &DrawStyle) {
        if points.is_empty() {
            return;
        }
        let pts: Vec<String> = points
            .iter()
            .map(|(x, y)| format!("{:.2},{:.2}", x, y))
            .collect();
        let mut elem = SvgElement::new("polygon").attr("points", pts.join(" "));
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw an arbitrary SVG path
    pub fn path(&mut self, d: impl Into<String>, style: &DrawStyle) {
        let mut elem = SvgElement::new("path").attr("d", d);
        elem = style.apply_to_attrs(elem);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw text
    pub fn text(&mut self, content: impl Into<String>, x: f64, y: f64, style: &DrawStyle) {
        let mut elem = SvgElement::new("text")
            .attr("x", format!("{:.2}", x))
            .attr("y", format!("{:.2}", y))
            .attr("font-size", format!("{:.1}", style.font_size))
            .attr("font-family", &style.font_family)
            .attr("font-weight", &style.font_weight)
            .attr("text-anchor", &style.text_anchor)
            .attr("dominant-baseline", &style.dominant_baseline)
            .text(content);
        // For text, fill is the font color; stroke usually none
        let fill_str = match &style.fill {
            Some(c) => c.to_hex(),
            None => "#000000".to_string(),
        };
        elem = elem.attr("fill", &fill_str);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw text with a transform (e.g., for rotated axis labels)
    pub fn text_transformed(
        &mut self,
        content: impl Into<String>,
        x: f64,
        y: f64,
        transform: &Transform,
        style: &DrawStyle,
    ) {
        let mut elem = SvgElement::new("text")
            .attr("x", format!("{:.2}", x))
            .attr("y", format!("{:.2}", y))
            .attr("font-size", format!("{:.1}", style.font_size))
            .attr("font-family", &style.font_family)
            .attr("font-weight", &style.font_weight)
            .attr("text-anchor", &style.text_anchor)
            .attr("dominant-baseline", &style.dominant_baseline);
        if !transform.is_empty() {
            elem = elem.attr("transform", transform.to_string());
        }
        elem = elem.text(content);
        let fill_str = match &style.fill {
            Some(c) => c.to_hex(),
            None => "#000000".to_string(),
        };
        elem = elem.attr("fill", &fill_str);
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Draw a rect filled with a gradient reference
    pub fn rect_with_gradient(
        &mut self,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        gradient_id: &str,
        stroke: Option<Color>,
        stroke_width: f64,
    ) {
        let fill = format!("url(#{})", gradient_id);
        let mut elem = SvgElement::new("rect")
            .attr("x", format!("{:.2}", x))
            .attr("y", format!("{:.2}", y))
            .attr("width", format!("{:.2}", width))
            .attr("height", format!("{:.2}", height))
            .attr("fill", &fill);
        if let Some(s) = stroke {
            elem = elem
                .attr("stroke", s.to_hex())
                .attr("stroke-width", format!("{:.2}", stroke_width));
        } else {
            elem = elem.attr("stroke", "none");
        }
        self.nodes.push(SvgNode::Element(elem));
    }

    /// Embed a raw SVG snippet (use sparingly)
    pub fn raw(&mut self, svg_snippet: impl Into<String>) {
        self.nodes.push(SvgNode::Raw(svg_snippet.into()));
    }

    /// Generate the complete SVG XML string
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        // SVG header
        let vb = match self.view_box {
            Some((x, y, w, h)) => format!("{:.2} {:.2} {:.2} {:.2}", x, y, w, h),
            None => format!("0 0 {:.2} {:.2}", self.width, self.height),
        };
        out.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"{}\">",
            self.width, self.height, vb
        ));
        // Defs
        let defs_elem = self.defs.clone().to_element();
        out.push_str(&defs_elem.render());
        // Background rect
        if let Some(bg) = &self.background {
            let bg_rect = SvgElement::new("rect")
                .attr("width", "100%")
                .attr("height", "100%")
                .attr("fill", bg.to_hex());
            out.push_str(&bg_rect.render());
        }
        // All nodes
        for node in &self.nodes {
            out.push_str(&node.render());
        }
        out.push_str("</svg>");
        out
    }
}

/// Path builder for complex SVG paths
#[derive(Debug, Clone, Default)]
pub struct PathBuilder {
    commands: Vec<String>,
}

impl PathBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn move_to(mut self, x: f64, y: f64) -> Self {
        self.commands.push(format!("M {:.2} {:.2}", x, y));
        self
    }

    pub fn line_to(mut self, x: f64, y: f64) -> Self {
        self.commands.push(format!("L {:.2} {:.2}", x, y));
        self
    }

    pub fn horizontal_to(mut self, x: f64) -> Self {
        self.commands.push(format!("H {:.2}", x));
        self
    }

    pub fn vertical_to(mut self, y: f64) -> Self {
        self.commands.push(format!("V {:.2}", y));
        self
    }

    pub fn cubic_bezier(mut self, cx1: f64, cy1: f64, cx2: f64, cy2: f64, x: f64, y: f64) -> Self {
        self.commands.push(format!(
            "C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
            cx1, cy1, cx2, cy2, x, y
        ));
        self
    }

    pub fn quadratic_bezier(mut self, cx: f64, cy: f64, x: f64, y: f64) -> Self {
        self.commands
            .push(format!("Q {:.2} {:.2} {:.2} {:.2}", cx, cy, x, y));
        self
    }

    /// Elliptical arc
    pub fn arc(
        mut self,
        rx: f64,
        ry: f64,
        x_rotation: f64,
        large_arc: bool,
        sweep: bool,
        x: f64,
        y: f64,
    ) -> Self {
        self.commands.push(format!(
            "A {:.2} {:.2} {:.2} {} {} {:.2} {:.2}",
            rx,
            ry,
            x_rotation,
            if large_arc { 1 } else { 0 },
            if sweep { 1 } else { 0 },
            x,
            y
        ));
        self
    }

    pub fn close(mut self) -> Self {
        self.commands.push("Z".to_string());
        self
    }

    pub fn build(self) -> String {
        self.commands.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_basic_output() {
        let mut canvas = SvgCanvas::new(400.0, 300.0);
        canvas.set_background(Color::WHITE);
        let style = DrawStyle::with_fill(Color::BLUE);
        canvas.rect(10.0, 10.0, 100.0, 50.0, &style);
        let svg = canvas.to_string();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn test_path_builder() {
        let path = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0)
            .close()
            .build();
        assert!(path.starts_with("M"));
        assert!(path.ends_with("Z"));
    }

    #[test]
    fn test_svg_escape() {
        let mut canvas = SvgCanvas::new(200.0, 100.0);
        let style = DrawStyle {
            fill: Some(Color::BLACK),
            font_size: 14.0,
            ..Default::default()
        };
        canvas.text("Hello & <World>", 50.0, 50.0, &style);
        let svg = canvas.to_string();
        assert!(svg.contains("&amp;"));
        assert!(svg.contains("&lt;"));
    }

    #[test]
    fn test_transform() {
        let t = Transform::new().translate(10.0, 20.0).rotate(45.0);
        let s = t.to_string();
        assert!(s.contains("translate"));
        assert!(s.contains("rotate"));
    }

    #[test]
    fn test_defs_gradient() {
        let mut canvas = SvgCanvas::new(200.0, 200.0);
        canvas.defs_mut().add_linear_gradient(
            "grad1",
            0.0,
            0.0,
            1.0,
            0.0,
            &[(0.0, Color::WHITE), (1.0, Color::BLUE)],
        );
        canvas.rect_with_gradient(0.0, 0.0, 200.0, 200.0, "grad1", None, 0.0);
        let svg = canvas.to_string();
        assert!(svg.contains("linearGradient"));
        assert!(svg.contains("grad1"));
    }
}
