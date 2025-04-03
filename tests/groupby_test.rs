use pandrs::{GroupBy, Series};

#[test]
fn test_groupby_creation() {
    // GroupByの基本的な作成
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = vec!["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    assert_eq!(group_by.group_count(), 3); // A, B, C の3グループ
}

#[test]
fn test_groupby_size() {
    // グループサイズの計算
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = vec!["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    let sizes = group_by.size();
    assert_eq!(sizes.get(&"A".to_string()), Some(&2));
    assert_eq!(sizes.get(&"B".to_string()), Some(&2));
    assert_eq!(sizes.get(&"C".to_string()), Some(&1));
}

#[test]
fn test_groupby_sum() {
    // グループごとの合計
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = vec!["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    let sums = group_by.sum().unwrap();
    assert_eq!(sums.get(&"A".to_string()), Some(&40)); // 10 + 30
    assert_eq!(sums.get(&"B".to_string()), Some(&60)); // 20 + 40
    assert_eq!(sums.get(&"C".to_string()), Some(&50)); // 50
}

#[test]
fn test_groupby_mean() {
    // グループごとの平均
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = vec!["A", "B", "A", "B", "C"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let group_by = GroupBy::new(keys, &values, Some("test_group".to_string())).unwrap();

    let means = group_by.mean().unwrap();
    assert_eq!(means.get(&"A".to_string()), Some(&20.0)); // (10 + 30) / 2
    assert_eq!(means.get(&"B".to_string()), Some(&30.0)); // (20 + 40) / 2
    assert_eq!(means.get(&"C".to_string()), Some(&50.0)); // 50 / 1
}

#[test]
fn test_groupby_numeric_keys() {
    // 数値キーによるグループ化
    let values = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string())).unwrap();
    let keys = vec![1, 2, 1, 2, 3];

    let group_by = GroupBy::new(keys, &values, Some("numeric_group".to_string())).unwrap();

    assert_eq!(group_by.group_count(), 3); // 1, 2, 3 の3グループ

    let sums = group_by.sum().unwrap();
    assert_eq!(sums.get(&1), Some(&40)); // 10 + 30
    assert_eq!(sums.get(&2), Some(&60)); // 20 + 40
    assert_eq!(sums.get(&3), Some(&50)); // 50
}

#[test]
fn test_groupby_consistent_length() {
    // キーとシリーズの長さが一致しない場合
    let values = Series::new(vec![10, 20, 30], Some("values".to_string())).unwrap();
    let keys = vec!["A", "B"].iter().map(|s| s.to_string()).collect();

    let result = GroupBy::new(keys, &values, Some("test_group".to_string()));
    assert!(result.is_err()); // 長さが一致しないのでエラー
}
