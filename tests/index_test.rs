use pandrs::index::{Index, RangeIndex};

#[test]
fn test_range_index_creation() {
    // Create index from range
    let index = RangeIndex::from_range(0..5).unwrap();

    assert_eq!(index.len(), 5);
    assert_eq!(index.get_value(0), Some(&0));
    assert_eq!(index.get_value(4), Some(&4));
    assert_eq!(index.get_value(5), None);

    // Position -> value mapping
    assert_eq!(index.get_loc(&0), Some(0));
    assert_eq!(index.get_loc(&4), Some(4));
    assert_eq!(index.get_loc(&5), None);
}

#[test]
fn test_string_index_creation() {
    // Create index from string values
    let values = vec![
        "apple".to_string(),
        "banana".to_string(),
        "cherry".to_string(),
    ];

    let index = Index::new(values.clone()).unwrap();

    assert_eq!(index.len(), 3);
    assert_eq!(index.get_value(0), Some(&"apple".to_string()));
    assert_eq!(index.get_value(2), Some(&"cherry".to_string()));

    // Value -> position mapping
    assert_eq!(index.get_loc(&"apple".to_string()), Some(0));
    assert_eq!(index.get_loc(&"banana".to_string()), Some(1));
    assert_eq!(index.get_loc(&"orange".to_string()), None);
}

#[test]
fn test_duplicate_index_values() {
    // Creating index with duplicate values (should result in error)
    let values = [1, 2, 3, 2, 4];
    let result = Index::new(values.to_vec());

    assert!(result.is_err());
}

#[test]
fn test_empty_index() {
    // Empty index
    let index = RangeIndex::from_range(0..0).unwrap();

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}
