//! XML part generation for the minimal .xlsx (OOXML SpreadsheetML) package.
//!
//! A valid xlsx file is a ZIP that contains at minimum:
//!
//! - `[Content_Types].xml`  — MIME overrides for each part.
//! - `_rels/.rels`          — root relationships (pointing at `xl/workbook.xml`).
//! - `xl/workbook.xml`      — workbook listing its sheets.
//! - `xl/_rels/workbook.xml.rels` — per-workbook relationships (sheets, styles, sharedStrings).
//! - `xl/styles.xml`        — at least one style set (font/fill/border/cellXfs).
//! - `xl/sharedStrings.xml` — interned text values.
//! - `xl/worksheets/sheet{N}.xml` — cell data per sheet.
//!
//! The helpers in this file generate only the skeletal parts: `[Content_Types].xml`,
//! `_rels/.rels`, `xl/workbook.xml`, `xl/_rels/workbook.xml.rels`, and
//! `xl/styles.xml`. Worksheet and sharedStrings XML are built in the writer
//! because they depend on runtime data.

use super::cell::xml_escape;

/// `[Content_Types].xml` body.
///
/// We opt for `Default`-based extensions for binary parts (rels, xml) and
/// `Override` entries for every sheet, styles, and sharedStrings. This matches
/// the structure most xlsx consumers (LibreOffice, calamine, Excel) expect.
pub(super) fn content_types(sheet_count: usize) -> String {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">"#);
    xml.push_str(r#"<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>"#);
    xml.push_str(r#"<Default Extension="xml" ContentType="application/xml"/>"#);
    xml.push_str(r#"<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>"#);
    for i in 1..=sheet_count {
        xml.push_str(&format!(
            r#"<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#
        ));
    }
    xml.push_str(r#"<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>"#);
    xml.push_str(r#"<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>"#);
    xml.push_str("</Types>");
    xml
}

/// Root `_rels/.rels` body. Points at the workbook.
pub(super) fn root_rels() -> String {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    );
    xml.push_str(r#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>"#);
    xml.push_str("</Relationships>");
    xml
}

/// `xl/workbook.xml` body. Each sheet gets a sequential `sheetId` and `r:id`
/// so the consumer can link back to a worksheet part.
pub(super) fn workbook_xml(sheet_names: &[String]) -> String {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#);
    xml.push_str("<sheets>");
    for (i, name) in sheet_names.iter().enumerate() {
        let sheet_id = i + 1;
        let rid = sheet_id; // rId{N} in workbook.xml.rels
        xml.push_str(&format!(
            r#"<sheet name="{}" sheetId="{}" r:id="rId{}"/>"#,
            xml_escape(name),
            sheet_id,
            rid
        ));
    }
    xml.push_str("</sheets>");
    xml.push_str("</workbook>");
    xml
}

/// `xl/_rels/workbook.xml.rels` body. Maps each `rId` used in workbook.xml back
/// to a worksheet file. Also wires in styles + sharedStrings with `rId` values
/// that come after the sheets.
pub(super) fn workbook_rels(sheet_count: usize) -> String {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    );
    for i in 1..=sheet_count {
        xml.push_str(&format!(
            r#"<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>"#
        ));
    }
    let styles_id = sheet_count + 1;
    let shared_id = sheet_count + 2;
    xml.push_str(&format!(
        r#"<Relationship Id="rId{styles_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>"#
    ));
    xml.push_str(&format!(
        r#"<Relationship Id="rId{shared_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>"#
    ));
    xml.push_str("</Relationships>");
    xml
}

/// Minimal `xl/styles.xml`. A single default cellXf is sufficient for all our
/// numeric/text/boolean values. We don't use number formats, because we write
/// booleans via `t="b"` and numbers as raw values — the consumer picks a
/// reasonable default.
pub(super) fn styles_xml() -> String {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    );
    xml.push_str(r#"<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>"#);
    xml.push_str(r#"<fills count="1"><fill><patternFill patternType="none"/></fill></fills>"#);
    xml.push_str(r#"<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"#);
    xml.push_str(
        r#"<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>"#,
    );
    xml.push_str(
        r#"<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>"#,
    );
    xml.push_str(
        r#"<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>"#,
    );
    xml.push_str(r#"<dxfs count="0"/>"#);
    xml.push_str(r#"<tableStyles count="0"/>"#);
    xml.push_str("</styleSheet>");
    xml
}

/// `xl/sharedStrings.xml` body built from a flat list of unique strings plus
/// the total number of cell references made against the table.
///
/// Per the OOXML spec these are two distinct counts: `uniqueCount` is the
/// number of `<si>` entries (`strings.len()`), while `count` is the total
/// number of cells across the workbook that reference the table — which is
/// normally *larger* than `uniqueCount` whenever any string repeats (e.g. a
/// column of ten "Alice" cells contributes 1 to `uniqueCount` but 10 to
/// `count`). The consumer indexes into this file via the 0-based position of
/// each `<si>`.
pub(super) fn shared_strings_xml(strings: &[String], total_refs: u32) -> String {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(&format!(
        r#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{}" uniqueCount="{}">"#,
        total_refs,
        strings.len()
    ));
    for s in strings {
        xml.push_str("<si><t xml:space=\"preserve\">");
        xml.push_str(&xml_escape(s));
        xml.push_str("</t></si>");
    }
    xml.push_str("</sst>");
    xml
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_types_includes_sheet_overrides() {
        let ct = content_types(2);
        assert!(ct.contains("sheet1.xml"));
        assert!(ct.contains("sheet2.xml"));
        assert!(ct.contains("styles.xml"));
        assert!(ct.contains("sharedStrings.xml"));
    }

    #[test]
    fn workbook_xml_contains_sheet_name() {
        let wb = workbook_xml(&["Sales & Marketing".to_string(), "Other".to_string()]);
        // `&` must be escaped:
        assert!(wb.contains("Sales &amp; Marketing"));
        assert!(wb.contains("Other"));
        assert!(wb.contains(r#"r:id="rId1""#));
        assert!(wb.contains(r#"r:id="rId2""#));
    }

    #[test]
    fn workbook_rels_points_to_each_sheet_plus_styles_and_sst() {
        let rels = workbook_rels(3);
        assert!(rels.contains("worksheets/sheet1.xml"));
        assert!(rels.contains("worksheets/sheet3.xml"));
        assert!(rels.contains("styles.xml"));
        assert!(rels.contains("sharedStrings.xml"));
    }

    #[test]
    fn shared_strings_roundtrips_ordering() {
        let ss = shared_strings_xml(&["a".to_string(), "b".to_string(), "a<b".to_string()], 3);
        assert!(ss.contains("a"));
        assert!(ss.contains("b"));
        // `<` is escaped.
        assert!(ss.contains("a&lt;b"));
        // 3 unique strings, each referenced exactly once here.
        assert!(ss.contains(r#"count="3""#));
        assert!(ss.contains(r#"uniqueCount="3""#));
    }

    #[test]
    fn shared_strings_distinguishes_count_from_unique_count() {
        // 2 unique strings, but referenced 5 times in total (e.g. because a
        // column repeats "a" four times).
        let ss = shared_strings_xml(&["a".to_string(), "b".to_string()], 5);
        assert!(ss.contains(r#"count="5""#));
        assert!(ss.contains(r#"uniqueCount="2""#));
    }
}
