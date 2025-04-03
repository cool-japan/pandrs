use pandrs::{DataFrame, Index, MultiIndex};
use pandrs::error::Result;

fn main() -> Result<()> {
    println!("=== MultiIndex 使用例 ===\n");

    // =========================================
    // MultiIndex の作成
    // =========================================

    println!("--- from_tuples によるMultiIndex作成 ---");

    // タプル（ベクタのベクタ）からMultiIndexを作成
    let tuples = vec![
        vec!["A".to_string(), "a".to_string()],
        vec!["A".to_string(), "b".to_string()],
        vec!["B".to_string(), "a".to_string()],
        vec!["B".to_string(), "b".to_string()],
    ];

    let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);
    let multi_idx = MultiIndex::from_tuples(tuples, names)?;

    println!("MultiIndex: {:?}\n", multi_idx);
    println!("レベル数: {}", multi_idx.n_levels());
    println!("行数: {}\n", multi_idx.len());

    // =========================================
    // MultiIndex 操作
    // =========================================
    
    println!("--- レベル値の取得 ---");
    let level0_values = multi_idx.get_level_values(0)?;
    println!("レベル0の値: {:?}", level0_values);
    
    let level1_values = multi_idx.get_level_values(1)?;
    println!("レベル1の値: {:?}\n", level1_values);
    
    println!("--- レベルの入れ替え ---");
    let swapped = multi_idx.swaplevel(0, 1)?;
    println!("入れ替え後: {:?}\n", swapped);

    // =========================================
    // MultiIndexを使用したDataFrame
    // =========================================

    println!("--- MultiIndexを持つDataFrame ---");
    
    // データフレームを作成
    let mut df = DataFrame::with_multi_index(multi_idx.clone());
    
    // データを追加
    let data = vec!["data1".to_string(), "data2".to_string(), "data3".to_string(), "data4".to_string()];
    df.add_column("data".to_string(), pandrs::Series::new(data, Some("data".to_string()))?)?;
    
    println!("DataFrame: {:?}\n", df);
    println!("行数: {}", df.row_count());
    println!("列数: {}", df.column_count());
    
    // =========================================
    // シンプルインデックスとMultiIndexの変換
    // =========================================
    
    println!("\n--- インデックス変換例 ---");
    
    // シンプルインデックスからDataFrameを作成
    let simple_idx = Index::new(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])?;
    let mut simple_df = DataFrame::with_index(simple_idx);
    
    // データを追加
    let values = vec![100, 200, 300];
    let str_values: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    simple_df.add_column("values".to_string(), pandrs::Series::new(str_values, Some("values".to_string()))?)?;
    
    println!("シンプルインデックスDF: {:?}", simple_df);
    
    // MultiIndexに変換するための準備
    let tuples = vec![
        vec!["Category".to_string(), "X".to_string()],
        vec!["Category".to_string(), "Y".to_string()],
        vec!["Category".to_string(), "Z".to_string()],
    ];
    
    // MultiIndexを作成して設定
    let new_multi_idx = MultiIndex::from_tuples(tuples, None)?;
    simple_df.set_multi_index(new_multi_idx)?;
    
    println!("MultiIndexに変換後: {:?}", simple_df);
    
    println!("\n=== サンプル完了 ===");
    Ok(())
}