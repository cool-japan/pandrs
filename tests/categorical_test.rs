use pandrs::series::{CategoricalOrder, StringCategorical};

#[test]
fn test_categorical_creation() {
    // 基本的なカテゴリカルの作成
    let values = vec!["a", "b", "a", "c", "b", "a"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let cat = StringCategorical::new(
        values_str,
        None,
        Some(CategoricalOrder::Unordered),
    )
    .unwrap();
    
    assert_eq!(cat.len(), 6);
    assert_eq!(cat.categories().len(), 3); // a, b, c
    assert!(cat.categories().contains(&"a".to_string()));
    assert!(cat.categories().contains(&"b".to_string()));
    assert!(cat.categories().contains(&"c".to_string()));
}

#[test]
fn test_categorical_with_explicit_categories() {
    // 明示的なカテゴリのリストを使用
    let values = vec!["a", "b", "a"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let categories = vec!["a", "b", "c", "d"];
    let categories_str = categories.iter().map(|s| s.to_string()).collect();
    
    let cat = StringCategorical::new(
        values_str,
        Some(categories_str),
        Some(CategoricalOrder::Ordered),
    )
    .unwrap();
    
    assert_eq!(cat.len(), 3);
    assert_eq!(cat.categories().len(), 4); // a, b, c, d
    assert_eq!(cat.ordered(), &CategoricalOrder::Ordered);
}

#[test]
fn test_categorical_get_value() {
    let values = vec!["a", "b", "c"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let cat = StringCategorical::new(values_str, None, None).unwrap();
    
    assert_eq!(cat.get(0).unwrap(), "a");
    assert_eq!(cat.get(1).unwrap(), "b");
    assert_eq!(cat.get(2).unwrap(), "c");
    assert!(cat.get(3).is_none()); // 範囲外
}

#[test]
fn test_categorical_reorder() {
    let values = vec!["a", "b", "a", "c"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let mut cat = StringCategorical::new(values_str, None, None).unwrap();
    
    // カテゴリの順序を変更
    let new_order = vec!["c", "b", "a"];
    let new_order_str = new_order.iter().map(|s| s.to_string()).collect();
    
    cat.reorder_categories(new_order_str).unwrap();
    
    // 新しい順序の確認
    assert_eq!(cat.categories()[0], "c");
    assert_eq!(cat.categories()[1], "b");
    assert_eq!(cat.categories()[2], "a");
    
    // 値は変わらないことを確認
    assert_eq!(cat.get(0).unwrap(), "a");
    assert_eq!(cat.get(1).unwrap(), "b");
    assert_eq!(cat.get(2).unwrap(), "a");
    assert_eq!(cat.get(3).unwrap(), "c");
}

#[test]
fn test_categorical_add_categories() {
    let values = vec!["a", "b"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let mut cat = StringCategorical::new(values_str, None, None).unwrap();
    
    // カテゴリを追加
    let new_cats = vec!["c", "d"];
    let new_cats_str = new_cats.iter().map(|s| s.to_string()).collect();
    
    cat.add_categories(new_cats_str).unwrap();
    
    // カテゴリリストの確認
    assert_eq!(cat.categories().len(), 4);
    assert!(cat.categories().contains(&"a".to_string()));
    assert!(cat.categories().contains(&"b".to_string()));
    assert!(cat.categories().contains(&"c".to_string()));
    assert!(cat.categories().contains(&"d".to_string()));
}

#[test]
fn test_categorical_remove_categories() {
    let values = vec!["a", "b", "a", "c"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let mut cat = StringCategorical::new(values_str, None, None).unwrap();
    
    // カテゴリを削除（値も削除される）
    let cats_to_remove = vec!["b".to_string()];
    
    cat.remove_categories(&cats_to_remove).unwrap();
    
    // カテゴリリストの確認
    assert_eq!(cat.categories().len(), 2);
    assert!(cat.categories().contains(&"a".to_string()));
    assert!(!cat.categories().contains(&"b".to_string()));
    assert!(cat.categories().contains(&"c".to_string()));
    
    // データ確認（bを含む値は-1コードに変わっている）
    assert_eq!(cat.get(0).unwrap(), "a");
    assert!(cat.get(1).is_none()); // 元々bだった値は欠損値になる
    assert_eq!(cat.get(2).unwrap(), "a");
    assert_eq!(cat.get(3).unwrap(), "c");
}

#[test]
fn test_categorical_value_counts() {
    let values = vec!["a", "b", "a", "c", "a", "b"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let cat = StringCategorical::new(values_str, None, None).unwrap();
    
    // 値の出現回数を計算
    let counts = cat.value_counts().unwrap();
    
    // Seriesの長さを確認（カテゴリカルデータの一意値の数）
    assert_eq!(counts.len(), 3);  // a, b, c の3つのカテゴリ
    
    // Seriesの名前を確認
    assert_eq!(counts.name().unwrap(), "count");
}

#[test]
fn test_categorical_to_series() {
    let values = vec!["a", "b", "a"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let cat = StringCategorical::new(values_str, None, None).unwrap();
    
    // Seriesに変換
    let series = cat.to_series(Some("test".to_string())).unwrap();
    
    assert_eq!(series.name().unwrap(), "test");
    assert_eq!(series.len(), 3);
    assert_eq!(series.values()[0], "a");
    assert_eq!(series.values()[1], "b");
    assert_eq!(series.values()[2], "a");
}

#[test]
fn test_categorical_equality() {
    let values1 = vec!["a", "b", "a"];
    let values1_str = values1.iter().map(|s| s.to_string()).collect();
    
    let values2 = vec!["a", "b", "a"];
    let values2_str = values2.iter().map(|s| s.to_string()).collect();
    
    let cat1 = StringCategorical::new(values1_str, None, None).unwrap();
    let cat2 = StringCategorical::new(values2_str, None, None).unwrap();
    
    // 同じデータなので等しいはず
    assert_eq!(cat1, cat2);
    
    // 異なるカテゴリ順序の場合
    let values3 = vec!["a", "b", "a"];
    let values3_str = values3.iter().map(|s| s.to_string()).collect();
    let categories = vec!["b", "a"]; // 順序が異なる
    let categories_str = categories.iter().map(|s| s.to_string()).collect();
    
    let cat3 = StringCategorical::new(
        values3_str, 
        Some(categories_str), 
        None
    ).unwrap();
    
    // カテゴリの順序が違うので不一致
    assert_ne!(cat1, cat3);
}

#[test]
fn test_invalid_categorical_creation() {
    // カテゴリに含まれない値
    let values = vec!["a", "b", "d"];
    let values_str = values.iter().map(|s| s.to_string()).collect();
    
    let categories = vec!["a", "b", "c"];
    let categories_str = categories.iter().map(|s| s.to_string()).collect();
    
    let result = StringCategorical::new(values_str, Some(categories_str), None);
    assert!(result.is_err());
    
    // 重複したカテゴリ
    let values2 = vec!["a", "b"];
    let values2_str = values2.iter().map(|s| s.to_string()).collect();
    
    let categories2 = vec!["a", "b", "b"]; // 重複
    let categories2_str = categories2.iter().map(|s| s.to_string()).collect();
    
    let result2 = StringCategorical::new(values2_str, Some(categories2_str), None);
    assert!(result2.is_err());
}