use pandrs::core::error::Result;
use pandrs::Series;

#[test]
fn test_string_accessor_integration() -> Result<()> {
    // Create a Series with string data
    let data = vec![
        "Hello World".to_string(),
        "  RUST Programming  ".to_string(),
        "pandas-like operations".to_string(),
        "Test123".to_string(),
        "".to_string(),
    ];
    let series = Series::new(data, Some("text_column".to_string()))?;

    // Test .str() accessor returns StringAccessor
    let str_accessor = series.str()?;

    // Test case conversion methods
    let upper_result = str_accessor.upper()?;
    let expected_upper = vec![
        "HELLO WORLD".to_string(),
        "  RUST PROGRAMMING  ".to_string(),
        "PANDAS-LIKE OPERATIONS".to_string(),
        "TEST123".to_string(),
        "".to_string(),
    ];
    assert_eq!(upper_result.values(), &expected_upper);

    let lower_result = str_accessor.lower()?;
    let expected_lower = vec![
        "hello world".to_string(),
        "  rust programming  ".to_string(),
        "pandas-like operations".to_string(),
        "test123".to_string(),
        "".to_string(),
    ];
    assert_eq!(lower_result.values(), &expected_lower);

    // Test string length
    let len_result = str_accessor.len()?;
    let expected_lengths = vec![11i64, 20i64, 22i64, 7i64, 0i64];
    assert_eq!(len_result.values(), &expected_lengths);

    // Test contains method
    let contains_result = str_accessor.contains("RUST", true, false)?;
    let expected_contains = vec![false, true, false, false, false];
    assert_eq!(contains_result.values(), &expected_contains);

    // Test case-insensitive contains
    let contains_insensitive = str_accessor.contains("rust", false, false)?;
    let expected_contains_insensitive = vec![false, true, false, false, false];
    assert_eq!(
        contains_insensitive.values(),
        &expected_contains_insensitive
    );

    // Test startswith
    let startswith_result = str_accessor.startswith("Hello", true)?;
    let expected_startswith = vec![true, false, false, false, false];
    assert_eq!(startswith_result.values(), &expected_startswith);

    // Test endswith
    let endswith_result = str_accessor.endswith("123", true)?;
    let expected_endswith = vec![false, false, false, true, false];
    assert_eq!(endswith_result.values(), &expected_endswith);

    // Test strip whitespace
    let strip_result = str_accessor.strip(None)?;
    let expected_strip = vec![
        "Hello World".to_string(),
        "RUST Programming".to_string(),
        "pandas-like operations".to_string(),
        "Test123".to_string(),
        "".to_string(),
    ];
    assert_eq!(strip_result.values(), &expected_strip);

    Ok(())
}

#[test]
fn test_string_accessor_advanced_operations() -> Result<()> {
    let data = vec![
        "apple,banana,cherry".to_string(),
        "dog123cat456".to_string(),
        "Hello World Test".to_string(),
        "replace_me_please".to_string(),
    ];
    let series = Series::new(data, Some("advanced_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test replace operation
    let replace_result = str_accessor.replace("_", "-", false, true)?;
    let expected_replace = vec![
        "apple,banana,cherry".to_string(),
        "dog123cat456".to_string(),
        "Hello World Test".to_string(),
        "replace-me-please".to_string(),
    ];
    assert_eq!(replace_result.values(), &expected_replace);

    // Test split operation (returns string representation)
    let split_result = str_accessor.split(",", None, false)?;
    assert_eq!(split_result.len(), 4);
    // First element should be split into parts
    assert!(split_result.values()[0].contains("apple"));
    assert!(split_result.values()[0].contains("banana"));
    assert!(split_result.values()[0].contains("cherry"));

    // Test title case
    let title_result = str_accessor.title()?;
    let expected_title = vec![
        "Apple,banana,cherry".to_string(), // Only first letter after whitespace is capitalized
        "Dog123cat456".to_string(),
        "Hello World Test".to_string(),
        "Replace_me_please".to_string(),
    ];
    assert_eq!(title_result.values(), &expected_title);

    // Test capitalize
    let capitalize_result = str_accessor.capitalize()?;
    let expected_capitalize = vec![
        "Apple,banana,cherry".to_string(),
        "Dog123cat456".to_string(),
        "Hello World Test".to_string(), // Capitalize first char, rest unchanged
        "Replace_me_please".to_string(),
    ];
    assert_eq!(capitalize_result.values(), &expected_capitalize);

    Ok(())
}

#[test]
fn test_string_accessor_regex_operations() -> Result<()> {
    let data = vec![
        "abc123def".to_string(),
        "xyz789ghi".to_string(),
        "nodigits".to_string(),
        "mix3d_n0mb3rs".to_string(),
    ];
    let series = Series::new(data, Some("regex_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test regex contains
    let regex_contains = str_accessor.contains(r"\d+", true, true)?;
    let expected_regex = vec![true, true, false, true];
    assert_eq!(regex_contains.values(), &expected_regex);

    // Test extract with regex (first capture group or whole match)
    let extract_result = str_accessor.extract(r"(\d+)", None)?;
    let expected_extract = vec![
        "123".to_string(),
        "789".to_string(),
        "".to_string(),  // No match
        "3".to_string(), // First digit match
    ];
    assert_eq!(extract_result.values(), &expected_extract);

    // Test count regex matches
    let count_result = str_accessor.count(r"\d", None)?;
    let expected_count = vec![3i64, 3i64, 0i64, 3i64]; // "mix3d_n0mb3rs" has digits: 3, 0, 3 = 3 total
    assert_eq!(count_result.values(), &expected_count);

    // Test findall
    let findall_result = str_accessor.findall(r"\d", None)?;
    assert!(findall_result.values()[0].contains("1"));
    assert!(findall_result.values()[0].contains("2"));
    assert!(findall_result.values()[0].contains("3"));

    Ok(())
}

#[test]
fn test_string_accessor_padding_operations() -> Result<()> {
    let data = vec![
        "short".to_string(),
        "medium text".to_string(),
        "a".to_string(),
        "exactly_ten".to_string(),
    ];
    let series = Series::new(data, Some("padding_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test left padding
    let left_pad = str_accessor.pad(10, "left", '*')?;
    let expected_left = vec![
        "*****short".to_string(),
        "medium text".to_string(), // Already longer than 10
        "*********a".to_string(),
        "exactly_ten".to_string(), // Exactly 10, no padding
    ];
    assert_eq!(left_pad.values(), &expected_left);

    // Test right padding
    let right_pad = str_accessor.pad(8, "right", '-')?;
    let expected_right = vec![
        "short---".to_string(),
        "medium text".to_string(), // Already longer
        "a-------".to_string(),
        "exactly_ten".to_string(), // Already longer
    ];
    assert_eq!(right_pad.values(), &expected_right);

    // Test both sides padding
    let both_pad = str_accessor.pad(7, "both", '=')?;
    let expected_both = vec![
        "=short=".to_string(),
        "medium text".to_string(), // Already longer
        "===a===".to_string(),
        "exactly_ten".to_string(), // Already longer
    ];
    assert_eq!(both_pad.values(), &expected_both);

    Ok(())
}

#[test]
fn test_string_accessor_strip_operations() -> Result<()> {
    let data = vec![
        "  hello  ".to_string(),
        "***world***".to_string(),
        "abctestcba".to_string(),
        "\t\nspaces\t\n".to_string(),
    ];
    let series = Series::new(data, Some("strip_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test strip with default (whitespace)
    let strip_default = str_accessor.strip(None)?;
    let expected_strip = vec![
        "hello".to_string(),
        "***world***".to_string(),
        "abctestcba".to_string(),
        "spaces".to_string(),
    ];
    assert_eq!(strip_default.values(), &expected_strip);

    // Test strip with custom characters
    let strip_custom = str_accessor.strip(Some("*abc"))?;
    let expected_custom = vec![
        "  hello  ".to_string(), // No * or abc to strip
        "world".to_string(),
        "test".to_string(),
        "\t\nspaces\t\n".to_string(), // No * or abc
    ];
    assert_eq!(strip_custom.values(), &expected_custom);

    // Test lstrip (left only)
    let lstrip_result = str_accessor.lstrip(None)?;
    let expected_lstrip = vec![
        "hello  ".to_string(),
        "***world***".to_string(),
        "abctestcba".to_string(),
        "spaces\t\n".to_string(),
    ];
    assert_eq!(lstrip_result.values(), &expected_lstrip);

    // Test rstrip (right only)
    let rstrip_result = str_accessor.rstrip(None)?;
    let expected_rstrip = vec![
        "  hello".to_string(),
        "***world***".to_string(),
        "abctestcba".to_string(),
        "\t\nspaces".to_string(),
    ];
    assert_eq!(rstrip_result.values(), &expected_rstrip);

    Ok(())
}

#[test]
fn test_string_accessor_edge_cases() -> Result<()> {
    // Test with empty strings and special characters
    let data = vec![
        "".to_string(),
        "🦀".to_string(),             // Unicode emoji
        "café".to_string(),           // Unicode accented characters
        "hello\nworld".to_string(),   // Newline
        "tab\tseparated".to_string(), // Tab
    ];
    let series = Series::new(data, Some("edge_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test length with Unicode (now returns character count, not byte count)
    let len_result = str_accessor.len()?;
    // Note: Now returns character count for proper Unicode support
    assert_eq!(len_result.values()[0], 0i64); // Empty string
    assert_eq!(len_result.values()[1], 1i64); // 🦀 is 1 character
    assert_eq!(len_result.values()[2], 4i64); // café is 4 characters (é is 1 char)

    // Test upper/lower with Unicode
    let upper_result = str_accessor.upper()?;
    assert_eq!(upper_result.values()[2], "CAFÉ".to_string());

    let lower_result = str_accessor.lower()?;
    assert_eq!(lower_result.values()[2], "café".to_string());

    // Test contains with special characters
    let contains_newline = str_accessor.contains("\n", true, false)?;
    let expected_newline = vec![false, false, false, true, false];
    assert_eq!(contains_newline.values(), &expected_newline);

    let contains_tab = str_accessor.contains("\t", true, false)?;
    let expected_tab = vec![false, false, false, false, true];
    assert_eq!(contains_tab.values(), &expected_tab);

    Ok(())
}

#[test]
fn test_string_accessor_method_chaining_concept() -> Result<()> {
    // Test that we can chain operations by applying multiple string operations
    let data = vec![
        "  Hello World  ".to_string(),
        "  RUST PROGRAMMING  ".to_string(),
        "  test data  ".to_string(),
    ];
    let series = Series::new(data, Some("chain_test".to_string()))?;

    // Simulate chaining: strip -> lower -> contains
    let stripped = series.str()?.strip(None)?;
    let lowered = stripped.str()?.lower()?;
    let contains_result = lowered.str()?.contains("rust", true, false)?;

    let expected_contains = vec![false, true, false];
    assert_eq!(contains_result.values(), &expected_contains);

    // Simulate chaining: strip -> title -> length
    let stripped2 = series.str()?.strip(None)?;
    let titled = stripped2.str()?.title()?;
    let lengths = titled.str()?.len()?;

    let expected_lengths = vec![11i64, 16i64, 9i64];
    assert_eq!(lengths.values(), &expected_lengths);

    Ok(())
}
