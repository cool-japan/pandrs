use pandrs::Series;

#[test]
fn debug_isalnum() {
    let test_cases = vec![
        ("hello", true),
        ("123", true),
        ("hello123", true),
        ("   ", false),
        ("Hello World", false), // has space
        ("RUST", true),
        ("", false),
    ];

    for (input, expected) in test_cases {
        let data = vec![input.to_string()];
        let series = Series::new(data, None).unwrap();
        let result = series.str().unwrap().isalnum().unwrap();
        let actual = result.values()[0];

        println!(
            "Input: '{}' | Expected: {} | Actual: {} | Match: {}",
            input,
            expected,
            actual,
            expected == actual
        );

        // Check individual characters
        if !input.is_empty() {
            let chars: Vec<char> = input.chars().collect();
            let all_alnum = chars.iter().all(|c| c.is_alphanumeric());
            println!(
                "  Characters: {:?} | All alphanumeric: {}",
                chars, all_alnum
            );
        }
    }
}
