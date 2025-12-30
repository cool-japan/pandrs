use pandrs::index::{MultiIndex, StringMultiIndex};

#[test]
fn test_multi_index_creation() {
    // Create by specifying levels, codes, and names
    let levels = vec![
        ["A".to_string(), "B".to_string()].to_vec(),
        ["1".to_string(), "2".to_string(), "3".to_string()].to_vec(),
    ];

    let codes = vec![vec![0, 0, 1, 1], vec![0, 1, 1, 2]];

    let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);

    let multi_idx = MultiIndex::new(levels.clone(), codes.clone(), names.clone()).unwrap();

    assert_eq!(multi_idx.len(), 4);
    assert_eq!(multi_idx.n_levels(), 2);
    assert_eq!(multi_idx.levels(), &levels);
    assert_eq!(multi_idx.codes(), &codes);
    assert_eq!(
        multi_idx.names(),
        &[Some("first".to_string()), Some("second".to_string())]
    );
}

#[test]
fn test_multi_index_from_tuples() {
    // Create from a list of tuples
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "2".to_string()],
        vec!["B".to_string(), "3".to_string()],
    ];

    let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);

    let multi_idx = StringMultiIndex::from_tuples(tuples.clone(), names).unwrap();

    assert_eq!(multi_idx.len(), 4);
    assert_eq!(multi_idx.n_levels(), 2);

    // Since order is not guaranteed, just check for containment
    assert!(multi_idx.levels()[0].contains(&"A".to_string()));
    assert!(multi_idx.levels()[0].contains(&"B".to_string()));
    assert!(multi_idx.levels()[1].contains(&"1".to_string()));
    assert!(multi_idx.levels()[1].contains(&"2".to_string()));
    assert!(multi_idx.levels()[1].contains(&"3".to_string()));

    // Search for a tuple
    let tuple = vec!["A".to_string(), "1".to_string()];
    assert!(multi_idx.get_loc(&tuple).is_some());
}

#[test]
fn test_get_tuple() {
    // Simple MultiIndex
    let levels = vec![
        vec!["A".to_string(), "B".to_string()],
        vec!["1".to_string(), "2".to_string()],
    ];

    let codes = vec![vec![0, 0, 1, 1], vec![0, 1, 0, 1]];

    let multi_idx = MultiIndex::new(levels, codes, None).unwrap();

    // Get tuple at each position
    assert_eq!(
        multi_idx.get_tuple(0).unwrap(),
        vec!["A".to_string(), "1".to_string()]
    );
    assert_eq!(
        multi_idx.get_tuple(1).unwrap(),
        vec!["A".to_string(), "2".to_string()]
    );
    assert_eq!(
        multi_idx.get_tuple(2).unwrap(),
        vec!["B".to_string(), "1".to_string()]
    );
    assert_eq!(
        multi_idx.get_tuple(3).unwrap(),
        vec!["B".to_string(), "2".to_string()]
    );
}

#[test]
fn test_get_level_values() {
    // Create MultiIndex
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "1".to_string()],
        vec!["B".to_string(), "2".to_string()],
    ];

    let multi_idx = StringMultiIndex::from_tuples(tuples, None).unwrap();

    // Get values at each level
    let level0 = multi_idx.get_level_values(0).unwrap();
    let level1 = multi_idx.get_level_values(1).unwrap();

    assert_eq!(level0, ["A", "A", "B", "B"]);
    assert_eq!(level1, ["1", "2", "1", "2"]);
}

#[test]
fn test_swaplevel() {
    // Create MultiIndex
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "1".to_string()],
        vec!["B".to_string(), "2".to_string()],
    ];

    let names = Some(vec![Some("upper".to_string()), Some("lower".to_string())]);
    let multi_idx = StringMultiIndex::from_tuples(tuples, names).unwrap();

    // Swap levels
    let swapped = multi_idx.swaplevel(0, 1).unwrap();

    assert_eq!(
        swapped.names(),
        &[Some("lower".to_string()), Some("upper".to_string())]
    );

    // Verify that the same information is retained after swapping
    assert_eq!(swapped.len(), 4);

    // Check swapped levels
    let orig_level0 = multi_idx.get_level_values(0).unwrap();
    let orig_level1 = multi_idx.get_level_values(1).unwrap();

    let swap_level0 = swapped.get_level_values(0).unwrap();
    let swap_level1 = swapped.get_level_values(1).unwrap();

    assert_eq!(orig_level0, swap_level1);
    assert_eq!(orig_level1, swap_level0);
}

#[test]
fn test_set_names() {
    // Create MultiIndex without names
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "1".to_string()],
        vec!["B".to_string(), "2".to_string()],
    ];

    let mut multi_idx = StringMultiIndex::from_tuples(tuples, None).unwrap();

    // Set names
    multi_idx
        .set_names(vec![Some("region".to_string()), Some("id".to_string())])
        .unwrap();

    assert_eq!(
        multi_idx.names(),
        &[Some("region".to_string()), Some("id".to_string())]
    );
}

#[test]
fn test_invalid_creation() {
    // Invalid input (level and code lengths don't match)
    let levels = vec![vec!["A".to_string(), "B".to_string()]];
    let codes = vec![
        vec![0, 1],
        vec![0, 1], // Extra code level
    ];

    let result = MultiIndex::new(levels, codes, None);
    assert!(result.is_err());

    // Invalid code value
    let levels = vec![vec!["A".to_string()]];
    let codes = vec![vec![0, 1]]; // 1 is out of range

    let result = MultiIndex::new(levels, codes, None);
    assert!(result.is_err());

    // Invalid number of names
    let levels = vec![
        vec!["A".to_string(), "B".to_string()],
        vec!["1".to_string(), "2".to_string()],
    ];
    let codes = vec![vec![0, 1], vec![0, 1]];
    let names = Some(vec![Some("first".to_string())]); // Only one name

    let result = MultiIndex::new(levels, codes, names);
    assert!(result.is_err());
}

#[test]
fn test_empty_tuples() {
    // Empty tuple list
    let tuples: Vec<Vec<String>> = vec![];
    let result = StringMultiIndex::from_tuples(tuples, None);
    assert!(result.is_err());
}

#[test]
fn test_inconsistent_tuples() {
    // Tuples of different lengths
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["B".to_string()], // Only one element
    ];

    let result = StringMultiIndex::from_tuples(tuples, None);
    assert!(result.is_err());
}
