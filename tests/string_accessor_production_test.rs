use pandrs::core::error::Result;
use pandrs::Series;

/// Test all new string validation methods  
#[test]
fn test_string_validation_methods() -> Result<()> {
    // Test each method individually to avoid confusion

    // Test isalpha
    let alpha_data = vec![
        "hello".to_string(),
        "123".to_string(),
        "hello123".to_string(),
    ];
    let alpha_series = Series::new(alpha_data, Some("test".to_string()))?;
    let alpha_result = alpha_series.str()?.isalpha()?;
    assert_eq!(alpha_result.values(), &[true, false, false]);

    // Test isdigit
    let digit_data = vec!["hello".to_string(), "123".to_string(), "12.3".to_string()];
    let digit_series = Series::new(digit_data, Some("test".to_string()))?;
    let digit_result = digit_series.str()?.isdigit()?;
    assert_eq!(digit_result.values(), &[false, true, false]);

    // Test isalnum
    let alnum_data = vec![
        "hello".to_string(),
        "123".to_string(),
        "hello123".to_string(),
        "hello world".to_string(),
    ];
    let alnum_series = Series::new(alnum_data, Some("test".to_string()))?;
    let alnum_result = alnum_series.str()?.isalnum()?;
    assert_eq!(alnum_result.values(), &[true, true, true, false]); // "hello world" has space

    // Test isspace
    let space_data = vec!["hello".to_string(), "   ".to_string(), "\t\n".to_string()];
    let space_series = Series::new(space_data, Some("test".to_string()))?;
    let space_result = space_series.str()?.isspace()?;
    assert_eq!(space_result.values(), &[false, true, true]);

    // Test isupper
    let upper_data = vec!["HELLO".to_string(), "Hello".to_string(), "123".to_string()];
    let upper_series = Series::new(upper_data, Some("test".to_string()))?;
    let upper_result = upper_series.str()?.isupper()?;
    assert_eq!(upper_result.values(), &[true, false, false]);

    // Test islower
    let lower_data = vec!["hello".to_string(), "Hello".to_string(), "123".to_string()];
    let lower_series = Series::new(lower_data, Some("test".to_string()))?;
    let lower_result = lower_series.str()?.islower()?;
    assert_eq!(lower_result.values(), &[true, false, false]);

    Ok(())
}

/// Test swapcase functionality
#[test]
fn test_swapcase() -> Result<()> {
    let data = vec![
        "Hello World".to_string(),
        "RUST programming".to_string(),
        "123".to_string(),
        "mixedCASE".to_string(),
    ];
    let series = Series::new(data, Some("test".to_string()))?;
    let str_accessor = series.str()?;

    let result = str_accessor.swapcase()?;
    let expected = vec![
        "hELLO wORLD".to_string(),
        "rust PROGRAMMING".to_string(),
        "123".to_string(),
        "MIXEDcase".to_string(),
    ];
    assert_eq!(result.values(), &expected);

    Ok(())
}

/// Test Unicode character length calculation
#[test]
fn test_unicode_length() -> Result<()> {
    let data = vec![
        "hello".to_string(),      // 5 ASCII chars
        "café".to_string(),       // 4 Unicode chars (é is 1 char, 2 bytes)
        "🦀".to_string(),         // 1 Unicode char (4 bytes)
        "🦀🦀🦀".to_string(),     // 3 Unicode chars (12 bytes)
        "".to_string(),           // 0 chars
        "Hello 世界".to_string(), // 8 chars (mixed ASCII + Chinese)
    ];
    let series = Series::new(data, Some("unicode_test".to_string()))?;
    let str_accessor = series.str()?;

    let result = str_accessor.len()?;
    let expected = vec![5i64, 4i64, 1i64, 3i64, 0i64, 8i64];
    assert_eq!(result.values(), &expected);

    Ok(())
}

/// Test Unicode padding operations
#[test]
fn test_unicode_padding() -> Result<()> {
    let data = vec![
        "café".to_string(),  // 4 Unicode chars
        "🦀".to_string(),    // 1 Unicode char
        "world".to_string(), // 5 ASCII chars
    ];
    let series = Series::new(data, Some("unicode_pad_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test left padding to width 8
    let left_pad = str_accessor.pad(8, "left", '*')?;
    let expected_left = vec![
        "****café".to_string(),
        "*******🦀".to_string(),
        "***world".to_string(),
    ];
    assert_eq!(left_pad.values(), &expected_left);

    Ok(())
}

/// Test input validation
#[test]
fn test_input_validation() -> Result<()> {
    let data = vec!["test".to_string()];
    let series = Series::new(data, Some("validation_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test invalid side parameter for padding
    let result = str_accessor.pad(10, "invalid", '*');
    assert!(result.is_err());

    if let Err(err) = result {
        let error_msg = format!("{:?}", err);
        assert!(error_msg.contains("Invalid side parameter"));
        assert!(error_msg.contains("Must be 'left', 'right', or 'both'"));
    }

    Ok(())
}

/// Test regex caching performance
#[test]
fn test_regex_caching_performance() -> Result<()> {
    let large_data: Vec<String> = (0..1000)
        .map(|i| format!("test_string_{}_data", i))
        .collect();
    let series = Series::new(large_data, Some("perf_test".to_string()))?;
    let str_accessor = series.str()?;

    // First call - regex should be compiled and cached
    let start1 = std::time::Instant::now();
    let _result1 = str_accessor.contains(r"\d+", true, true)?;
    let duration1 = start1.elapsed();

    // Second call - regex should be retrieved from cache (faster)
    let start2 = std::time::Instant::now();
    let _result2 = str_accessor.contains(r"\d+", true, true)?;
    let duration2 = start2.elapsed();

    // Cache should make second call faster (though this might not always be true due to CPU caching)
    println!("First call: {:?}, Second call: {:?}", duration1, duration2);

    // Both should complete quickly
    assert!(duration1 < std::time::Duration::from_millis(100));
    assert!(duration2 < std::time::Duration::from_millis(100));

    Ok(())
}

/// Test error handling with contextual information
#[test]
fn test_contextual_error_handling() -> Result<()> {
    let data = vec!["test".to_string()];
    let series = Series::new(data, Some("error_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test invalid regex pattern
    let result = str_accessor.contains("[invalid", true, true);
    assert!(result.is_err());

    if let Err(err) = result {
        let error_msg = format!("{:?}", err);
        assert!(error_msg.contains("Invalid regex pattern"));
        assert!(error_msg.contains("[invalid"));
    }

    Ok(())
}

/// Test performance with large datasets
#[test]
fn test_large_dataset_performance() -> Result<()> {
    let large_data: Vec<String> = (0..10_000)
        .map(|i| format!("Performance test string number {}", i))
        .collect();
    let series = Series::new(large_data, Some("large_perf_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test various operations for performance
    let start = std::time::Instant::now();

    let _upper = str_accessor.upper()?;
    let _contains = str_accessor.contains("test", false, false)?;
    let _lengths = str_accessor.len()?;
    let _alpha = str_accessor.isalpha()?;

    let total_duration = start.elapsed();

    // Should complete within reasonable time
    assert!(total_duration < std::time::Duration::from_millis(1000));
    println!("Large dataset operations took: {:?}", total_duration);

    Ok(())
}

/// Test edge cases and boundary conditions
#[test]
fn test_edge_cases() -> Result<()> {
    // Test with empty series
    let empty_data: Vec<String> = vec![];
    let empty_series = Series::new(empty_data, Some("empty".to_string()))?;
    let empty_accessor = empty_series.str()?;

    let empty_result = empty_accessor.upper()?;
    assert_eq!(empty_result.len(), 0);

    // Test with very long strings
    let long_string = "a".repeat(10_000);
    let long_data = vec![long_string];
    let long_series = Series::new(long_data, Some("long".to_string()))?;
    let long_accessor = long_series.str()?;

    let long_result = long_accessor.upper()?;
    assert_eq!(long_result.values()[0].len(), 10_000);
    assert!(long_result.values()[0].chars().all(|c| c == 'A'));

    // Test with only special characters
    let special_data = vec!["!@#$%^&*()".to_string(), "[]{}".to_string()];
    let special_series = Series::new(special_data, Some("special".to_string()))?;
    let special_accessor = special_series.str()?;

    let special_alpha = special_accessor.isalpha()?;
    assert_eq!(special_alpha.values(), &[false, false]);

    Ok(())
}

/// Test method chaining behavior
#[test]
fn test_method_chaining() -> Result<()> {
    let data = vec!["  Hello World  ".to_string(), "  TEST DATA  ".to_string()];
    let series = Series::new(data, Some("chain_test".to_string()))?;

    // Chain multiple operations: strip -> lower -> contains
    let stripped = series.str()?.strip(None)?;
    let lowered = stripped.str()?.lower()?;
    let contains_result = lowered.str()?.contains("hello", true, false)?;

    assert_eq!(contains_result.values(), &[true, false]);

    // Chain: strip -> title -> length
    let stripped2 = series.str()?.strip(None)?;
    let titled = stripped2.str()?.title()?;
    let lengths = titled.str()?.len()?;

    assert_eq!(lengths.values(), &[11i64, 9i64]);

    Ok(())
}

/// Test all regex methods with caching
#[test]
fn test_all_regex_methods() -> Result<()> {
    let data = vec![
        "abc123def456".to_string(),
        "hello world".to_string(),
        "TEST123".to_string(),
        "".to_string(),
    ];
    let series = Series::new(data, Some("regex_test".to_string()))?;
    let str_accessor = series.str()?;

    // Test contains with regex
    let contains_regex = str_accessor.contains(r"\d+", true, true)?;
    assert_eq!(contains_regex.values(), &[true, false, true, false]);

    // Test replace with regex
    let replace_regex = str_accessor.replace(r"\d+", "XXX", true, true)?;
    assert_eq!(replace_regex.values()[0], "abcXXXdefXXX");
    assert_eq!(replace_regex.values()[2], "TESTXXX");

    // Test extract with regex
    let extract_regex = str_accessor.extract(r"(\d+)", None)?;
    assert_eq!(extract_regex.values()[0], "123"); // First capture group
    assert_eq!(extract_regex.values()[2], "123");

    // Test count with regex
    let count_regex = str_accessor.count(r"\d", None)?;
    assert_eq!(count_regex.values(), &[6i64, 0i64, 3i64, 0i64]);

    // Test findall with regex
    let findall_regex = str_accessor.findall(r"\d+", None)?;
    assert!(findall_regex.values()[0].contains("123"));
    assert!(findall_regex.values()[0].contains("456"));

    Ok(())
}
