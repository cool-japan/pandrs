use pandrs::index::{Index, RangeIndex};

#[test]
fn test_range_index_creation() {
    // 範囲からインデックスを作成
    let index = RangeIndex::from_range(0..5).unwrap();
    
    assert_eq!(index.len(), 5);
    assert_eq!(index.get_value(0), Some(&0));
    assert_eq!(index.get_value(4), Some(&4));
    assert_eq!(index.get_value(5), None);
    
    // 位置 -> 値のマッピング
    assert_eq!(index.get_loc(&0), Some(0));
    assert_eq!(index.get_loc(&4), Some(4));
    assert_eq!(index.get_loc(&5), None);
}

#[test]
fn test_string_index_creation() {
    // 文字列値からインデックスを作成
    let values = vec![
        "apple".to_string(),
        "banana".to_string(),
        "cherry".to_string()
    ];
    
    let index = Index::new(values.clone()).unwrap();
    
    assert_eq!(index.len(), 3);
    assert_eq!(index.get_value(0), Some(&"apple".to_string()));
    assert_eq!(index.get_value(2), Some(&"cherry".to_string()));
    
    // 値 -> 位置のマッピング
    assert_eq!(index.get_loc(&"apple".to_string()), Some(0));
    assert_eq!(index.get_loc(&"banana".to_string()), Some(1));
    assert_eq!(index.get_loc(&"orange".to_string()), None);
}

#[test]
fn test_duplicate_index_values() {
    // 重複した値を持つインデックスの作成（エラーになるはず）
    let values = vec![1, 2, 3, 2, 4];
    let result = Index::new(values);
    
    assert!(result.is_err());
}

#[test]
fn test_empty_index() {
    // 空のインデックス
    let index = RangeIndex::from_range(0..0).unwrap();
    
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}