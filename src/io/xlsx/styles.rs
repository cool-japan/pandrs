//! Minimal `xl/styles.xml` parsing: just enough to detect which cells are
//! date/time-formatted and convert their numeric serials to ISO-8601 text.
//!
//! xlsx has no distinct "date" cell type — a date is a plain IEEE-754
//! numeric value (days since an epoch, with the fractional part encoding
//! time-of-day) that is *styled* to display as a date via a number format
//! (`numFmt`). Recovering "is this cell a date" therefore requires:
//!
//! 1. Reading each cell's style index (the `s` attribute on `<c>`).
//! 2. Resolving that style index to a `numFmtId` via `xl/styles.xml`'s
//!    `<cellXfs>` list.
//! 3. Classifying that `numFmtId` as date/time/datetime/not-a-date — either
//!    via the small set of well-known builtin ids, or (for custom formats,
//!    id >= 164) by scanning the format code's text for date/time tokens.
//!
//! We deliberately only recognise the common builtin date/time format ids
//! (14-22, 45-47); the locale-specific builtin ids in the 27-36 / 50-58
//! ranges (Japanese/Chinese/Korean calendar variants, per ECMA-376
//! §18.8.30) are not covered and are treated as non-date. This is an
//! intentional, documented scope limit rather than a silent gap.

use std::collections::HashMap;
use std::io::Cursor;

use quick_xml::events::attributes::Attribute;
use quick_xml::events::Event;
use quick_xml::Reader;
use quick_xml::XmlVersion;

use crate::error::Result;

use super::error::xml_err;

/// Coarse classification of a numFmt's date/time semantics, used to decide
/// how a numeric serial should be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DateKind {
    /// Not a date/time format.
    None,
    /// Calendar date only (e.g. `"yyyy-mm-dd"`) -> renders as `"2024-01-31"`.
    Date,
    /// Calendar date + time-of-day (e.g. `"m/d/yy h:mm"`) -> renders as
    /// `"2024-01-31T13:45:00"`.
    DateTime,
    /// Time-of-day only, no calendar date component (e.g. `"h:mm:ss"`) ->
    /// renders as `"13:45:00"`.
    Time,
}

/// Parsed subset of `xl/styles.xml` needed to detect date-formatted cells.
#[derive(Debug, Clone, Default)]
pub(super) struct StylesInfo {
    /// `cell_xf_num_fmts[style_index] == numFmtId` for that `<xf>` entry in
    /// `<cellXfs>` (the table the `s` attribute on `<c>` indexes into).
    cell_xf_num_fmts: Vec<u32>,
    /// Custom format codes keyed by `numFmtId` (only ids >= 164 are custom
    /// per the spec; builtin ids are recognised structurally).
    custom_formats: HashMap<u32, String>,
}

impl StylesInfo {
    /// Resolve the [`DateKind`] implied by a cell's style index (the `s`
    /// attribute on `<c>`). A missing style index, an out-of-range index, or
    /// a style with no recognisable date format all safely resolve to
    /// `DateKind::None`.
    pub(super) fn date_kind(&self, style_index: Option<u32>) -> DateKind {
        let Some(idx) = style_index else {
            return DateKind::None;
        };
        let Some(&num_fmt_id) = self.cell_xf_num_fmts.get(idx as usize) else {
            return DateKind::None;
        };
        if let Some(code) = self.custom_formats.get(&num_fmt_id) {
            classify_format_code(code)
        } else {
            classify_builtin_num_fmt_id(num_fmt_id)
        }
    }
}

/// Parse `xl/styles.xml`, extracting `<numFmts>` (custom format codes) and
/// `<cellXfs>` (the numFmtId used by each cell style).
pub(super) fn parse_styles(bytes: &[u8]) -> Result<StylesInfo> {
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();

    let mut custom_formats: HashMap<u32, String> = HashMap::new();
    let mut cell_xf_num_fmts: Vec<u32> = Vec::new();
    let mut in_cell_xfs = false;

    loop {
        match reader.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Eof => break,
            Event::Start(e) => {
                let tag = e.name();
                let tag = tag.as_ref();
                if tag == b"cellXfs" {
                    in_cell_xfs = true;
                } else if tag == b"xf" && in_cell_xfs {
                    cell_xf_num_fmts.push(read_num_fmt_id_attr(&e));
                } else if tag == b"numFmt" {
                    if let Some((id, code)) = read_num_fmt_def(&e)? {
                        custom_formats.insert(id, code);
                    }
                }
            }
            Event::Empty(e) => {
                let tag = e.name();
                let tag = tag.as_ref();
                if tag == b"xf" && in_cell_xfs {
                    cell_xf_num_fmts.push(read_num_fmt_id_attr(&e));
                } else if tag == b"numFmt" {
                    if let Some((id, code)) = read_num_fmt_def(&e)? {
                        custom_formats.insert(id, code);
                    }
                }
            }
            Event::End(e) => {
                if e.name().as_ref() == b"cellXfs" {
                    in_cell_xfs = false;
                }
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(StylesInfo {
        cell_xf_num_fmts,
        custom_formats,
    })
}

/// Read the `numFmtId` attribute off an `<xf>` element. Absent or
/// unparseable defaults to `0` ("General" — not a date), which is the safe
/// choice since it just means "no date detected for this style".
fn read_num_fmt_id_attr(e: &quick_xml::events::BytesStart<'_>) -> u32 {
    for a in e.attributes().with_checks(false).flatten() {
        if a.key.as_ref() == b"numFmtId" {
            if let Ok(s) = std::str::from_utf8(&a.value) {
                if let Ok(v) = s.parse::<u32>() {
                    return v;
                }
            }
        }
    }
    0
}

/// Read `numFmtId` + `formatCode` off a `<numFmt>` element.
fn read_num_fmt_def(e: &quick_xml::events::BytesStart<'_>) -> Result<Option<(u32, String)>> {
    let mut id: Option<u32> = None;
    let mut code: Option<String> = None;
    for a in e.attributes().with_checks(false).flatten() {
        match a.key.as_ref() {
            b"numFmtId" => {
                id = std::str::from_utf8(&a.value)
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok());
            }
            b"formatCode" => {
                code = Some(decode_attr_value(&a)?);
            }
            _ => {}
        }
    }
    Ok(match (id, code) {
        (Some(id), Some(code)) => Some((id, code)),
        _ => None,
    })
}

fn decode_attr_value(a: &Attribute<'_>) -> Result<String> {
    let cow = a
        .normalized_value(XmlVersion::Implicit1_0)
        .map_err(xml_err)?;
    Ok(cow.into_owned())
}

/// Classify one of the well-known builtin numFmtIds (ECMA-376 §18.8.30).
/// Only the common, locale-independent date/time ids are recognised (see
/// module docs for the intentional scope limit).
fn classify_builtin_num_fmt_id(id: u32) -> DateKind {
    match id {
        14 | 15 | 16 | 17 => DateKind::Date, // mm-dd-yy, d-mmm-yy, d-mmm, mmm-yy
        18 | 19 | 20 | 21 => DateKind::Time, // h:mm[:ss] [AM/PM]
        22 => DateKind::DateTime,            // m/d/yy h:mm
        45 | 46 | 47 => DateKind::Time,      // mm:ss, [h]:mm:ss, mmss.0
        _ => DateKind::None,
    }
}

/// Classify a custom format code by scanning it for date/time tokens,
/// ignoring quoted literal sections (`"..."`) and bracketed
/// locale/color/condition sections (`[...]`), neither of which carry
/// date/time meaning.
fn classify_format_code(code: &str) -> DateKind {
    let mut cleaned = String::with_capacity(code.len());
    let mut chars = code.chars().peekable();
    let mut in_quotes = false;
    while let Some(c) = chars.next() {
        match c {
            '"' => in_quotes = !in_quotes,
            '[' if !in_quotes => {
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                }
            }
            _ if in_quotes => {}
            _ => cleaned.push(c),
        }
    }

    let lower = cleaned.to_ascii_lowercase();
    if lower.trim().is_empty() || lower.trim() == "general" || lower.contains('@') {
        return DateKind::None;
    }
    let has_date = lower.contains('y') || lower.contains('d');
    let has_time = lower.contains('h') || lower.contains('s');
    match (has_date, has_time) {
        (true, true) => DateKind::DateTime,
        (true, false) => DateKind::Date,
        (false, true) => DateKind::Time,
        (false, false) => DateKind::None,
    }
}

/// Convert an xlsx numeric serial into an ISO-8601 string, honoring the
/// workbook's date system (`date1904`) and the target [`DateKind`].
/// Returns `None` when the serial can't plausibly be a date (non-finite,
/// wildly out of range, or the underlying calendar arithmetic fails) —
/// callers should leave the original numeric value untouched in that case
/// rather than emit a fabricated date.
pub(super) fn excel_serial_to_iso(serial: f64, date1904: bool, kind: DateKind) -> Option<String> {
    if kind == DateKind::None || !serial.is_finite() {
        return None;
    }
    let days_f = serial.floor();
    // Comfortably covers the full realistic calendar range (roughly +/- 2700
    // years); anything further out is not a plausible date, so we leave the
    // raw number untouched rather than emit nonsense.
    if !(-1_000_000.0..=1_000_000.0).contains(&days_f) {
        return None;
    }
    let frac = serial - days_f;
    let mut day_offset = days_f as i64;

    let base = if date1904 {
        chrono::NaiveDate::from_ymd_opt(1904, 1, 1)?
    } else {
        // Excel's 1900 date system uses 1899-12-30 as its epoch and (in)
        // famously treats 1900 as a leap year. Every serial in [0, 60) needs
        // one extra day added to compensate; serial 60 itself (the
        // fictitious "29 Feb 1900") collapses onto the same real date as
        // 59, matching Excel's own documented, widely-implemented behavior
        // (see e.g. Microsoft KB214326).
        if (0..60).contains(&day_offset) {
            day_offset += 1;
        }
        chrono::NaiveDate::from_ymd_opt(1899, 12, 30)?
    };

    let date = base.checked_add_signed(chrono::Duration::days(day_offset))?;
    let secs = (frac * 86_400.0).round().clamp(0.0, 86_399.0) as u32;

    match kind {
        DateKind::Date => Some(date.format("%Y-%m-%d").to_string()),
        DateKind::Time => {
            let time = chrono::NaiveTime::from_num_seconds_from_midnight_opt(secs, 0)?;
            Some(time.format("%H:%M:%S").to_string())
        }
        DateKind::DateTime => {
            let time = chrono::NaiveTime::from_num_seconds_from_midnight_opt(secs, 0)?;
            Some(format!(
                "{}T{}",
                date.format("%Y-%m-%d"),
                time.format("%H:%M:%S")
            ))
        }
        DateKind::None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_builtin_ids_matches_ecma_376() {
        assert_eq!(classify_builtin_num_fmt_id(0), DateKind::None); // General
        assert_eq!(classify_builtin_num_fmt_id(2), DateKind::None); // 0.00
        assert_eq!(classify_builtin_num_fmt_id(14), DateKind::Date); // mm-dd-yy
        assert_eq!(classify_builtin_num_fmt_id(21), DateKind::Time); // h:mm:ss
        assert_eq!(classify_builtin_num_fmt_id(22), DateKind::DateTime); // m/d/yy h:mm
        assert_eq!(classify_builtin_num_fmt_id(46), DateKind::Time); // [h]:mm:ss
    }

    #[test]
    fn classify_custom_format_code_detects_date_time_tokens() {
        assert_eq!(classify_format_code("yyyy-mm-dd"), DateKind::Date);
        assert_eq!(classify_format_code("hh:mm:ss"), DateKind::Time);
        assert_eq!(
            classify_format_code("yyyy-mm-dd hh:mm:ss"),
            DateKind::DateTime
        );
        assert_eq!(classify_format_code("General"), DateKind::None);
        assert_eq!(classify_format_code("0.00%"), DateKind::None);
        assert_eq!(classify_format_code("#,##0.00"), DateKind::None);
        assert_eq!(classify_format_code("@"), DateKind::None);
    }

    #[test]
    fn classify_custom_format_code_ignores_quoted_and_bracketed_sections() {
        // A currency format with a quoted literal that happens to contain
        // date-ish letters must not be misclassified as a date.
        assert_eq!(
            classify_format_code("[$-409]\"USD\" #,##0.00"),
            DateKind::None
        );
        // Colored negative-number format: the bracket section is not a date.
        assert_eq!(classify_format_code("[Red]-0.00"), DateKind::None);
    }

    #[test]
    fn excel_serial_to_iso_matches_known_fixed_points() {
        // 1970-01-01 is the canonical cross-check used by every xlsx date
        // implementation.
        assert_eq!(
            excel_serial_to_iso(25569.0, false, DateKind::Date),
            Some("1970-01-01".to_string())
        );
        assert_eq!(
            excel_serial_to_iso(44197.0, false, DateKind::Date),
            Some("2021-01-01".to_string())
        );
        // The famous leap-year-bug collision: both 59 and 60 land on the
        // same (real) date, 1900-02-28.
        assert_eq!(
            excel_serial_to_iso(59.0, false, DateKind::Date),
            Some("1900-02-28".to_string())
        );
        assert_eq!(
            excel_serial_to_iso(60.0, false, DateKind::Date),
            Some("1900-02-28".to_string())
        );
        assert_eq!(
            excel_serial_to_iso(61.0, false, DateKind::Date),
            Some("1900-03-01".to_string())
        );
    }

    #[test]
    fn excel_serial_to_iso_handles_date1904() {
        assert_eq!(
            excel_serial_to_iso(0.0, true, DateKind::Date),
            Some("1904-01-01".to_string())
        );
        assert_eq!(
            excel_serial_to_iso(1.0, true, DateKind::Date),
            Some("1904-01-02".to_string())
        );
    }

    #[test]
    fn excel_serial_to_iso_renders_datetime_and_time() {
        // 45000.5 -> half a day past the date component.
        let dt = excel_serial_to_iso(45000.5, false, DateKind::DateTime).expect("datetime");
        assert!(dt.contains('T'));
        assert!(dt.ends_with("12:00:00"));

        let t = excel_serial_to_iso(0.75, false, DateKind::Time).expect("time");
        assert_eq!(t, "18:00:00");
    }

    #[test]
    fn excel_serial_to_iso_rejects_non_finite_and_out_of_range() {
        assert_eq!(excel_serial_to_iso(f64::NAN, false, DateKind::Date), None);
        assert_eq!(
            excel_serial_to_iso(f64::INFINITY, false, DateKind::Date),
            None
        );
        assert_eq!(
            excel_serial_to_iso(1.0e18, false, DateKind::Date),
            None,
            "wildly out-of-range serials must not fabricate a date"
        );
    }
}
