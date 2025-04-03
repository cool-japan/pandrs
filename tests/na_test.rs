use pandrs::{NA, NASeries};

#[test]
fn test_na_creation() {
    // NA型の基本的な作成と操作
    let value: NA<i32> = NA::Value(42);
    let na: NA<i32> = NA::NA;
    
    assert!(!value.is_na());
    assert!(value.is_value());
    assert_eq!(value.value(), Some(&42));
    
    assert!(na.is_na());
    assert!(!na.is_value());
    assert_eq!(na.value(), None);
}

#[test]
fn test_na_operations() {
    // NA型の演算
    let a = NA::Value(10);
    let b = NA::Value(5);
    let na = NA::<i32>::NA;
    
    // 基本演算
    assert_eq!(a + b, NA::Value(15));
    assert_eq!(a - b, NA::Value(5));
    assert_eq!(a * b, NA::Value(50));
    assert_eq!(a / b, NA::Value(2));
    
    // NAとの演算
    assert_eq!(a + na, NA::NA);
    assert_eq!(na * b, NA::NA);
    assert_eq!(na / na, NA::NA);
    
    // ゼロ除算
    assert_eq!(a / NA::Value(0), NA::NA);
}

#[test]
fn test_na_series_creation() {
    // NASeriesの作成
    let data = vec![NA::Value(10), NA::Value(20), NA::NA, NA::Value(40)];
    let series = NASeries::new(data, Some("test".to_string())).unwrap();
    
    assert_eq!(series.len(), 4);
    assert_eq!(series.na_count(), 1);
    assert_eq!(series.value_count(), 3);
    assert!(series.has_na());
}

#[test]
fn test_na_series_from_options() {
    // OptionからNASeriesを作成
    let options = vec![Some(10), None, Some(30), None, Some(50)];
    let series = NASeries::from_options(options, Some("from_options".to_string())).unwrap();
    
    assert_eq!(series.len(), 5);
    assert_eq!(series.na_count(), 2);
    assert_eq!(series.value_count(), 3);
    assert!(series.has_na());
}

#[test]
fn test_na_series_operations() {
    // NASeriesの数値操作
    let data = vec![NA::Value(10), NA::Value(20), NA::NA, NA::Value(40), NA::NA];
    let series = NASeries::new(data, Some("test".to_string())).unwrap();
    
    // 集計関数
    assert_eq!(series.sum(), NA::Value(70));  // NAは無視
    assert_eq!(series.mean(), NA::Value(70 / 3));  // NAは無視
    assert_eq!(series.min(), NA::Value(10));
    assert_eq!(series.max(), NA::Value(40));
    
    // 空のSeriesの場合
    let empty_series = NASeries::<i32>::new(vec![], Some("empty".to_string())).unwrap();
    assert_eq!(empty_series.sum(), NA::NA);
    assert_eq!(empty_series.mean(), NA::NA);
    assert_eq!(empty_series.min(), NA::NA);
    assert_eq!(empty_series.max(), NA::NA);
}

#[test]
fn test_na_series_handling() {
    // NAの処理メソッド
    let data = vec![NA::Value(10), NA::Value(20), NA::NA, NA::Value(40), NA::NA];
    let series = NASeries::new(data, Some("test".to_string())).unwrap();
    
    // NAの削除
    let dropped = series.dropna().unwrap();
    assert_eq!(dropped.len(), 3);
    assert_eq!(dropped.na_count(), 0);
    assert!(!dropped.has_na());
    
    // NAの埋め合わせ
    let filled = series.fillna(0).unwrap();
    assert_eq!(filled.len(), 5);
    assert_eq!(filled.na_count(), 0);
    assert!(!filled.has_na());
    
    // 値のチェック（fill後）
    assert_eq!(filled.get(0), Some(&NA::Value(10)));
    assert_eq!(filled.get(2), Some(&NA::Value(0)));  // 埋められたNA
}