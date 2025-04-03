use pandrs::series::{CategoricalOrder, StringCategorical, NASeries};
use pandrs::{DataFrame, NA};
use std::path::Path;

#[test]
fn test_categorical_from_na_vec() {
    let values = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
        NA::Value("c".to_string()),
    ];
    
    let cat = StringCategorical::from_na_vec(
        values,
        None,
        Some(CategoricalOrder::Unordered),
    )
    .unwrap();
    
    assert_eq!(cat.len(), 4);
    assert_eq!(cat.categories().len(), 3); // a, b, c
    
    // コードを確認
    let codes = cat.codes();
    assert_eq!(codes.len(), 4);
    assert_ne!(codes[0], -1);  // 正常な値
    assert_ne!(codes[1], -1);  // 正常な値
    assert_eq!(codes[2], -1);  // NA値
    assert_ne!(codes[3], -1);  // 正常な値
}

#[test]
fn test_categorical_to_na_vec() {
    let values = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
        NA::Value("c".to_string()),
    ];
    
    let cat = StringCategorical::from_na_vec(values, None, None).unwrap();
    let na_values = cat.to_na_vec();
    
    assert_eq!(na_values.len(), 4);
    
    // 値を確認
    match &na_values[0] {
        NA::Value(v) => assert_eq!(v, "a"),
        NA::NA => panic!("Should be a value, not NA"),
    }
    
    match &na_values[1] {
        NA::Value(v) => assert_eq!(v, "b"),
        NA::NA => panic!("Should be a value, not NA"),
    }
    
    match &na_values[2] {
        NA::Value(_) => panic!("Should be NA, not a value"),
        NA::NA => {},  // OK
    }
    
    match &na_values[3] {
        NA::Value(v) => assert_eq!(v, "c"),
        NA::NA => panic!("Should be a value, not NA"),
    }
}

#[test]
fn test_categorical_to_na_series() {
    let values = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
    ];
    
    let cat = StringCategorical::from_na_vec(values, None, None).unwrap();
    let na_series = cat.to_na_series(Some("test".to_string())).unwrap();
    
    assert_eq!(na_series.len(), 3);
    assert_eq!(na_series.name().unwrap(), "test");
    assert_eq!(na_series.na_count(), 1);
    assert_eq!(na_series.value_count(), 2);
}

#[test]
fn test_categorical_set_operations() {
    // カテゴリカルデータを作成
    let values1 = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
    ];
    
    let values2 = vec![
        NA::Value("b".to_string()),
        NA::Value("c".to_string()),
        NA::NA,
    ];
    
    let cat1 = StringCategorical::from_na_vec(values1, None, None).unwrap();
    let cat2 = StringCategorical::from_na_vec(values2, None, None).unwrap();
    
    // 和集合
    let union = cat1.union(&cat2).unwrap();
    let union_cats: Vec<String> = union.categories().to_vec();
    
    assert_eq!(union_cats.len(), 3);
    assert!(union_cats.contains(&"a".to_string()));
    assert!(union_cats.contains(&"b".to_string()));
    assert!(union_cats.contains(&"c".to_string()));
    
    // 積集合
    let intersection = cat1.intersection(&cat2).unwrap();
    let intersection_cats: Vec<String> = intersection.categories().to_vec();
    
    assert_eq!(intersection_cats.len(), 1);
    assert!(intersection_cats.contains(&"b".to_string()));
    
    // 差集合
    let difference = cat1.difference(&cat2).unwrap();
    let difference_cats: Vec<String> = difference.categories().to_vec();
    
    assert_eq!(difference_cats.len(), 1);
    assert!(difference_cats.contains(&"a".to_string()));
}

#[test]
fn test_dataframe_add_na_series_as_categorical() {
    // NA値を含むシリーズを作成
    let values = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
        NA::Value("c".to_string()),
    ];
    
    let na_series = NASeries::new(values, Some("test".to_string())).unwrap();
    
    // DataFrameを作成
    let mut df = DataFrame::new();
    
    // カテゴリカル列として追加
    df.add_na_series_as_categorical(
        "test".to_string(),
        na_series,
        None,
        Some(CategoricalOrder::Ordered),
    ).unwrap();
    
    // 確認
    assert!(df.is_categorical("test"));
    assert_eq!(df.row_count(), 4);
    assert_eq!(df.column_count(), 3); // test列と2つのメタデータ列
    
    // カテゴリカルデータを取得して検証
    let cat = df.get_categorical("test").unwrap();
    assert_eq!(cat.categories().len(), 3); // a, b, c
    
    match cat.ordered() {
        CategoricalOrder::Ordered => {},  // OK
        CategoricalOrder::Unordered => panic!("Should be Ordered"),
    }
}

#[test]
fn test_categorical_csv_io() {
    // DataFrameを作成
    let mut df = DataFrame::new();
    
    // NA値を含むシリーズを作成
    let values1 = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
    ];
    
    let values2 = vec![
        NA::Value("1".to_string()),
        NA::NA,
        NA::Value("3".to_string()),
    ];
    
    let na_series1 = NASeries::new(values1, Some("cat1".to_string())).unwrap();
    let na_series2 = NASeries::new(values2, Some("cat2".to_string())).unwrap();
    
    // カテゴリカル列として追加
    df.add_na_series_as_categorical(
        "cat1".to_string(),
        na_series1,
        None,
        Some(CategoricalOrder::Unordered),
    ).unwrap();
    
    df.add_na_series_as_categorical(
        "cat2".to_string(),
        na_series2,
        None,
        Some(CategoricalOrder::Ordered),
    ).unwrap();
    
    // 一時ファイルに保存
    let temp_path = Path::new("/tmp/categorical_test.csv");
    df.to_csv_with_categorical(temp_path).unwrap();
    
    // ファイルから読み込み
    let df_loaded = DataFrame::from_csv_with_categorical(temp_path, true).unwrap();
    
    // 検証
    assert!(df_loaded.is_categorical("cat1"));
    assert!(df_loaded.is_categorical("cat2"));
    assert_eq!(df_loaded.row_count(), df.row_count());
    
    // カテゴリカルデータを取得して検証
    let cat1 = df_loaded.get_categorical("cat1").unwrap();
    let cat2 = df_loaded.get_categorical("cat2").unwrap();
    
    assert_eq!(cat1.categories().len(), 2); // a, b
    assert_eq!(cat2.categories().len(), 2); // 1, 3
    
    match cat1.ordered() {
        CategoricalOrder::Unordered => {},  // OK
        CategoricalOrder::Ordered => panic!("Should be Unordered"),
    }
    
    match cat2.ordered() {
        CategoricalOrder::Ordered => {},  // OK
        CategoricalOrder::Unordered => panic!("Should be Ordered"),
    }
}