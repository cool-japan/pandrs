
use pandrs::{Series, GroupBy, PandRSError};

fn main() -> Result<(), PandRSError> {
    println!("=== グループ操作のサンプル ===");
    
    // データの準備
    let values = Series::new(vec![10, 20, 15, 30, 25, 15], Some("values".to_string()))?;
    
    // グループ化のためのキー
    let keys = vec!["A", "B", "A", "C", "B", "A"];
    
    // GroupByを作成
    let group_by = GroupBy::new(
        keys.iter().map(|s| s.to_string()).collect(),
        &values,
        Some("by_category".to_string())
    )?;
    
    println!("グループ数: {}", group_by.group_count());
    
    // グループサイズを取得
    let sizes = group_by.size();
    println!("\n--- グループサイズ ---");
    for (key, size) in &sizes {
        println!("グループ '{}': {} 要素", key, size);
    }
    
    // 各グループの合計を計算
    let sums = group_by.sum()?;
    println!("\n--- グループごとの合計 ---");
    for (key, sum) in &sums {
        println!("グループ '{}' の合計: {}", key, sum);
    }
    
    // 各グループの平均を計算
    let means = group_by.mean()?;
    println!("\n--- グループごとの平均 ---");
    for (key, mean) in &means {
        println!("グループ '{}' の平均: {:.2}", key, mean);
    }
    
    // 異なるデータ型でのグループ化
    println!("\n--- 異なるデータ型でのグループ化 ---");
    let ages = Series::new(vec![25, 30, 25, 40, 30, 25], Some("ages".to_string()))?;
    
    let age_group_by = GroupBy::new(
        ages.values().to_vec(),
        &values,
        Some("by_age".to_string())
    )?;
    
    let age_means = age_group_by.mean()?;
    for (key, mean) in &age_means {
        println!("年齢 {} グループの平均値: {:.2}", key, mean);
    }
    
    println!("=== グループ操作サンプル完了 ===");
    Ok(())
}