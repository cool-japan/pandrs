use pandrs::{Axis, DataFrame};
use std::collections::HashMap;

#[test]
fn test_dataframe_apply() {
    // テスト用のDataFrameを作成
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["4", "5", "6"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // apply関数で各列の最初の要素を取得
    let result = df.apply(
        |col| col.get(0).unwrap().clone(),
        Axis::Column,
        Some("first_elem".to_string()),
    ).unwrap();
    
    // 結果の確認
    assert_eq!(result.len(), 2);
    assert_eq!(
        result.get(result.index().get_loc(&"col1".to_string()).unwrap()).unwrap(),
        "1"
    );
    assert_eq!(
        result.get(result.index().get_loc(&"col2".to_string()).unwrap()).unwrap(),
        "4"
    );
}

#[test]
fn test_dataframe_applymap() {
    // テスト用のDataFrameを作成
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["4", "5", "6"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // applymap関数で各要素を整数に変換して2倍にする
    let result = df.applymap(|x| x.parse::<i32>().unwrap_or(0) * 2).unwrap();
    
    // 結果の確認
    assert_eq!(result.column_names(), df.column_names());
    
    // 変換後の値を確認
    let result_col1 = result.get_column("col1").unwrap();
    let result_col2 = result.get_column("col2").unwrap();
    
    assert_eq!(result_col1.get(0).unwrap(), "2");
    assert_eq!(result_col1.get(1).unwrap(), "4");
    assert_eq!(result_col1.get(2).unwrap(), "6");
    
    assert_eq!(result_col2.get(0).unwrap(), "8");
    assert_eq!(result_col2.get(1).unwrap(), "10");
    assert_eq!(result_col2.get(2).unwrap(), "12");
}

#[test]
fn test_dataframe_mask() {
    // テスト用のDataFrameを作成
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["4", "5", "6"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // mask関数で2以上の値をXに置換
    let result = df.mask(|x| x.parse::<i32>().unwrap_or(0) >= 2, "X").unwrap();
    
    // 結果の確認
    let result_col1 = result.get_column("col1").unwrap();
    let result_col2 = result.get_column("col2").unwrap();
    
    assert_eq!(result_col1.get(0).unwrap(), "1");  // 1 < 2 なので変更なし
    assert_eq!(result_col1.get(1).unwrap(), "X");  // 2 >= 2 なのでX
    assert_eq!(result_col1.get(2).unwrap(), "X");  // 3 >= 2 なのでX
    
    assert_eq!(result_col2.get(0).unwrap(), "X");  // 4 >= 2 なのでX
    assert_eq!(result_col2.get(1).unwrap(), "X");  // 5 >= 2 なのでX
    assert_eq!(result_col2.get(2).unwrap(), "X");  // 6 >= 2 なのでX
}

#[test]
fn test_dataframe_where_func() {
    // テスト用のDataFrameを作成
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["1", "2", "3"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["4", "5", "6"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // where関数で3以上の値だけ保持し、他はXに置換
    let result = df.where_func(|x| x.parse::<i32>().unwrap_or(0) >= 3, "X").unwrap();
    
    // 結果の確認
    let result_col1 = result.get_column("col1").unwrap();
    let result_col2 = result.get_column("col2").unwrap();
    
    assert_eq!(result_col1.get(0).unwrap(), "X");  // 1 < 3 なのでX
    assert_eq!(result_col1.get(1).unwrap(), "X");  // 2 < 3 なのでX
    assert_eq!(result_col1.get(2).unwrap(), "3");  // 3 >= 3 なので変更なし
    
    assert_eq!(result_col2.get(0).unwrap(), "4");  // 4 >= 3 なので変更なし
    assert_eq!(result_col2.get(1).unwrap(), "5");  // 5 >= 3 なので変更なし
    assert_eq!(result_col2.get(2).unwrap(), "6");  // 6 >= 3 なので変更なし
}

#[test]
fn test_dataframe_replace() {
    // テスト用のDataFrameを作成
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["b", "c", "d"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // 置換マップを作成
    let mut replace_map = HashMap::new();
    replace_map.insert("a".to_string(), "X".to_string());
    replace_map.insert("c".to_string(), "Y".to_string());
    
    // replace関数で値を置換
    let result = df.replace(&replace_map).unwrap();
    
    // 結果の確認
    let result_col1 = result.get_column("col1").unwrap();
    let result_col2 = result.get_column("col2").unwrap();
    
    assert_eq!(result_col1.get(0).unwrap(), "X");  // a -> X
    assert_eq!(result_col1.get(1).unwrap(), "b");  // 変更なし
    assert_eq!(result_col1.get(2).unwrap(), "Y");  // c -> Y
    
    assert_eq!(result_col2.get(0).unwrap(), "b");  // 変更なし
    assert_eq!(result_col2.get(1).unwrap(), "Y");  // c -> Y
    assert_eq!(result_col2.get(2).unwrap(), "d");  // 変更なし
}

#[test]
fn test_dataframe_duplicated() {
    // テスト用のDataFrameを作成（重複行あり）
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["a", "b", "a", "c"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["1", "2", "1", "3"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // 重複行を検出（最初の行を保持）
    let duplicated_first = df.duplicated(None, Some("first")).unwrap();
    
    // 結果の確認
    assert_eq!(duplicated_first.len(), 4);
    assert_eq!(*duplicated_first.get(0).unwrap(), false);  // 最初のa,1は重複ではない
    assert_eq!(*duplicated_first.get(1).unwrap(), false);  // b,2は重複ではない
    assert_eq!(*duplicated_first.get(2).unwrap(), true);   // 2番目のa,1は重複
    assert_eq!(*duplicated_first.get(3).unwrap(), false);  // c,3は重複ではない
    
    // 重複行を検出（最後の行を保持）
    let duplicated_last = df.duplicated(None, Some("last")).unwrap();
    
    // 結果の確認
    assert_eq!(duplicated_last.len(), 4);
    assert_eq!(*duplicated_last.get(0).unwrap(), true);   // 最初のa,1は重複（最後の行を保持）
    assert_eq!(*duplicated_last.get(1).unwrap(), false);  // b,2は重複ではない
    assert_eq!(*duplicated_last.get(2).unwrap(), false);  // 2番目のa,1は最後なので重複ではない
    assert_eq!(*duplicated_last.get(3).unwrap(), false);  // c,3は重複ではない
}

#[test]
fn test_dataframe_drop_duplicates() {
    // テスト用のDataFrameを作成（重複行あり）
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["a", "b", "a", "c"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["1", "2", "1", "3"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // 重複行を削除（最初の行を保持）
    let deduped_first = df.drop_duplicates(None, Some("first")).unwrap();
    
    // 結果の確認
    assert_eq!(deduped_first.row_count(), 3);  // 1行削除されて3行に
    
    let deduped_col1 = deduped_first.get_column("col1").unwrap();
    let deduped_col2 = deduped_first.get_column("col2").unwrap();
    
    assert_eq!(deduped_col1.get(0).unwrap(), "a");  // 最初のa,1は残る
    assert_eq!(deduped_col1.get(1).unwrap(), "b");  // b,2は残る
    assert_eq!(deduped_col1.get(2).unwrap(), "c");  // c,3は残る
    
    assert_eq!(deduped_col2.get(0).unwrap(), "1");
    assert_eq!(deduped_col2.get(1).unwrap(), "2");
    assert_eq!(deduped_col2.get(2).unwrap(), "3");
    
    // 重複行を削除（最後の行を保持）
    let deduped_last = df.drop_duplicates(None, Some("last")).unwrap();
    
    // 結果の確認
    assert_eq!(deduped_last.row_count(), 3);  // 1行削除されて3行に
    
    let deduped_col1_last = deduped_last.get_column("col1").unwrap();
    let deduped_col2_last = deduped_last.get_column("col2").unwrap();
    
    // インデックスが保持されないため、順序で確認
    let values1: Vec<_> = deduped_col1_last.values().iter().collect();
    let values2: Vec<_> = deduped_col2_last.values().iter().collect();
    
    // b, a(2回目), c が残るはず
    assert!(values1.contains(&&"b".to_string()));
    assert!(values1.contains(&&"a".to_string()));
    assert!(values1.contains(&&"c".to_string()));
    
    assert!(values2.contains(&&"2".to_string()));
    assert!(values2.contains(&&"1".to_string()));
    assert!(values2.contains(&&"3".to_string()));
}

#[test]
fn test_duplicated_with_subset() {
    // テスト用のDataFrameを作成
    let mut df = DataFrame::new();
    
    // 列を追加
    let col1 = vec!["a", "b", "a", "c"].iter().map(|s| s.to_string()).collect();
    let col2 = vec!["1", "2", "3", "4"].iter().map(|s| s.to_string()).collect();
    
    let series1 = pandrs::Series::new(col1, Some("col1".to_string())).unwrap();
    let series2 = pandrs::Series::new(col2, Some("col2".to_string())).unwrap();
    
    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();
    
    // col1だけを見て重複を検出
    let subset = vec!["col1".to_string()];
    let duplicated = df.duplicated(Some(&subset), Some("first")).unwrap();
    
    // 結果の確認
    assert_eq!(duplicated.len(), 4);
    assert_eq!(*duplicated.get(0).unwrap(), false);  // 最初のaは重複ではない
    assert_eq!(*duplicated.get(1).unwrap(), false);  // bは重複ではない
    assert_eq!(*duplicated.get(2).unwrap(), true);   // 2番目のaは重複
    assert_eq!(*duplicated.get(3).unwrap(), false);  // cは重複ではない
}