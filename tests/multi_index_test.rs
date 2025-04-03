use pandrs::index::{MultiIndex, StringMultiIndex};

#[test]
fn test_multi_index_creation() {
    // レベル、コード、名前を指定して作成
    let levels = vec![
        vec!["A".to_string(), "B".to_string()],
        vec!["1".to_string(), "2".to_string(), "3".to_string()],
    ];
    
    let codes = vec![
        vec![0, 0, 1, 1],
        vec![0, 1, 1, 2],
    ];
    
    let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);
    
    let multi_idx = MultiIndex::new(levels.clone(), codes.clone(), names.clone()).unwrap();
    
    assert_eq!(multi_idx.len(), 4);
    assert_eq!(multi_idx.n_levels(), 2);
    assert_eq!(multi_idx.levels(), &levels);
    assert_eq!(multi_idx.codes(), &codes);
    assert_eq!(multi_idx.names(), &[Some("first".to_string()), Some("second".to_string())]);
}

#[test]
fn test_multi_index_from_tuples() {
    // タプルリストから作成
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
    
    // 順番が保証されていないので、含まれるかどうかだけ確認
    assert!(multi_idx.levels()[0].contains(&"A".to_string()));
    assert!(multi_idx.levels()[0].contains(&"B".to_string()));
    assert!(multi_idx.levels()[1].contains(&"1".to_string()));
    assert!(multi_idx.levels()[1].contains(&"2".to_string()));
    assert!(multi_idx.levels()[1].contains(&"3".to_string()));
    
    // タプルの検索
    let tuple = vec!["A".to_string(), "1".to_string()];
    assert!(multi_idx.get_loc(&tuple).is_some());
}

#[test]
fn test_get_tuple() {
    // シンプルなMultiIndex
    let levels = vec![
        vec!["A".to_string(), "B".to_string()],
        vec!["1".to_string(), "2".to_string()],
    ];
    
    let codes = vec![
        vec![0, 0, 1, 1],
        vec![0, 1, 0, 1],
    ];
    
    let multi_idx = MultiIndex::new(levels, codes, None).unwrap();
    
    // 各位置のタプルを取得
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
    // MultiIndexを作成
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "1".to_string()],
        vec!["B".to_string(), "2".to_string()],
    ];
    
    let multi_idx = StringMultiIndex::from_tuples(tuples, None).unwrap();
    
    // 各レベルの値を取得
    let level0 = multi_idx.get_level_values(0).unwrap();
    let level1 = multi_idx.get_level_values(1).unwrap();
    
    assert_eq!(level0, vec!["A", "A", "B", "B"]);
    assert_eq!(level1, vec!["1", "2", "1", "2"]);
}

#[test]
fn test_swaplevel() {
    // MultiIndexを作成
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "1".to_string()],
        vec!["B".to_string(), "2".to_string()],
    ];
    
    let names = Some(vec![Some("upper".to_string()), Some("lower".to_string())]);
    let multi_idx = StringMultiIndex::from_tuples(tuples, names).unwrap();
    
    // レベルを交換
    let swapped = multi_idx.swaplevel(0, 1).unwrap();
    
    assert_eq!(
        swapped.names(),
        &[Some("lower".to_string()), Some("upper".to_string())]
    );
    
    // 交換後も同じ情報が保持されていることを確認
    assert_eq!(swapped.len(), 4);
    
    // 交換されたレベルを確認
    let orig_level0 = multi_idx.get_level_values(0).unwrap();
    let orig_level1 = multi_idx.get_level_values(1).unwrap();
    
    let swap_level0 = swapped.get_level_values(0).unwrap();
    let swap_level1 = swapped.get_level_values(1).unwrap();
    
    assert_eq!(orig_level0, swap_level1);
    assert_eq!(orig_level1, swap_level0);
}

#[test]
fn test_set_names() {
    // 名前なしでMultiIndexを作成
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["A".to_string(), "2".to_string()],
        vec!["B".to_string(), "1".to_string()],
        vec!["B".to_string(), "2".to_string()],
    ];
    
    let mut multi_idx = StringMultiIndex::from_tuples(tuples, None).unwrap();
    
    // 名前を設定
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
    // 不正な入力（レベルとコードの長さが一致しない）
    let levels = vec![vec!["A".to_string(), "B".to_string()]];
    let codes = vec![
        vec![0, 1],
        vec![0, 1], // 余分なコードレベル
    ];
    
    let result = MultiIndex::new(levels, codes, None);
    assert!(result.is_err());
    
    // 不正なコード値
    let levels = vec![vec!["A".to_string()]];
    let codes = vec![vec![0, 1]]; // 1は範囲外
    
    let result = MultiIndex::new(levels, codes, None);
    assert!(result.is_err());
    
    // 不正な名前の数
    let levels = vec![
        vec!["A".to_string(), "B".to_string()],
        vec!["1".to_string(), "2".to_string()],
    ];
    let codes = vec![
        vec![0, 1],
        vec![0, 1],
    ];
    let names = Some(vec![Some("first".to_string())]); // 名前が1つしかない
    
    let result = MultiIndex::new(levels, codes, names);
    assert!(result.is_err());
}

#[test]
fn test_empty_tuples() {
    // 空のタプルリスト
    let tuples: Vec<Vec<String>> = vec![];
    let result = StringMultiIndex::from_tuples(tuples, None);
    assert!(result.is_err());
}

#[test]
fn test_inconsistent_tuples() {
    // 長さの異なるタプル
    let tuples = vec![
        vec!["A".to_string(), "1".to_string()],
        vec!["B".to_string()], // 1つしか要素がない
    ];
    
    let result = StringMultiIndex::from_tuples(tuples, None);
    assert!(result.is_err());
}