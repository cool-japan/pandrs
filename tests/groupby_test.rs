use pandrs::{GroupBy, Series};

#[test]
fn test_groupby_creation() {
    // Basic creation of GroupBy
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = ["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    assert_eq!(group_by.group_count(), 3); // 3 groups: A, B, C
}

#[test]
fn test_groupby_size() {
    // Calculate group sizes
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = ["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    let sizes = group_by.size();
    assert_eq!(sizes.get("A"), Some(&2));
    assert_eq!(sizes.get("B"), Some(&2));
    assert_eq!(sizes.get("C"), Some(&1));
}

#[test]
fn test_groupby_sum() {
    // Sum by group
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = ["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    let sums = group_by.sum().unwrap();
    assert_eq!(sums.get("A"), Some(&40)); // 10 + 30
    assert_eq!(sums.get("B"), Some(&60)); // 20 + 40
    assert_eq!(sums.get("C"), Some(&50)); // 50
}

#[test]
fn test_groupby_mean() {
    // Mean by group
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = ["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    let means = group_by.mean().unwrap();
    assert_eq!(means.get("A"), Some(&20.0)); // (10 + 30) / 2
    assert_eq!(means.get("B"), Some(&30.0)); // (20 + 40) / 2
    assert_eq!(means.get("C"), Some(&50.0)); // 50 / 1
}

#[test]
fn test_groupby_numeric_keys() {
    // Grouping by numeric keys
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = vec![1, 2, 1, 2, 3];

    let group_by = GroupBy::new(keys, &values, Some("numeric_group".to_string())).unwrap();

    assert_eq!(group_by.group_count(), 3); // 3 groups: 1, 2, 3

    let sums = group_by.sum().unwrap();
    assert_eq!(sums.get(&1), Some(&40)); // 10 + 30
    assert_eq!(sums.get(&2), Some(&60)); // 20 + 40
    assert_eq!(sums.get(&3), Some(&50)); // 50
}

#[test]
fn test_groupby_consistent_length() {
    // When key and series lengths don't match
    let values = Series::new(vec![10, 20, 30], Some("values".to_string())).unwrap();
    let keys = ["A", "B"].iter().map(|s| s.to_string()).collect();

    let result = GroupBy::new(keys, &values, Some("test_group".to_string()));
    assert!(result.is_err()); // Error due to length mismatch
}
