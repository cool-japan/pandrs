use pandrs::series::Series;

#[test]
fn test_series_creation() {
    // 整数型シリーズの作成
    let series = Series::new(vec![1, 2, 3, 4, 5], Some("test".to_string())).unwrap();
    assert_eq!(series.len(), 5);
    assert_eq!(series.name(), Some(&"test".to_string()));
    assert_eq!(series.get(0), Some(&1));
    assert_eq!(series.get(4), Some(&5));
    assert_eq!(series.get(5), None);
}

#[test]
fn test_series_numeric_operations() {
    // 整数型シリーズでの数値操作
    let series = Series::new(vec![10, 20, 30, 40, 50], Some("numbers".to_string())).unwrap();

    // 合計
    assert_eq!(series.sum(), 150);

    // 平均
    assert_eq!(series.mean().unwrap(), 30);

    // 最小値
    assert_eq!(series.min().unwrap(), 10);

    // 最大値
    assert_eq!(series.max().unwrap(), 50);
}

#[test]
fn test_empty_series() {
    // 空のシリーズ
    let empty_series: Series<i32> = Series::new(vec![], Some("empty".to_string())).unwrap();

    assert_eq!(empty_series.len(), 0);
    assert!(empty_series.is_empty());

    // 空のシリーズでの合計は0（デフォルト値）になるはず
    assert_eq!(empty_series.sum(), 0);

    // 空のシリーズでの統計計算はエラーになるはず
    assert!(empty_series.mean().is_err());
    assert!(empty_series.min().is_err());
    assert!(empty_series.max().is_err());
}

#[test]
fn test_series_with_strings() {
    // 文字列型シリーズ
    let series = Series::new(
        vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ],
        Some("fruits".to_string()),
    )
    .unwrap();

    assert_eq!(series.len(), 3);
    assert_eq!(series.name(), Some(&"fruits".to_string()));
    assert_eq!(series.get(0), Some(&"apple".to_string()));
}
