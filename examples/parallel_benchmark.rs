use std::time::Instant;
use pandrs::{OptimizedDataFrame, LazyFrame, AggregateOp};
use pandrs::column::{Int64Column, Float64Column, BooleanColumn, Column, StringColumn, ColumnTrait};

fn main() {
    println!("並列処理パフォーマンスベンチマーク");
    println!("============================");
    
    // データサイズ
    const ROWS: usize = 1_000_000;
    
    // ====================
    // 大規模データフレーム作成
    // ====================
    println!("\n[1] 大規模DataFrame作成 ({} 行)", ROWS);
    
    let mut int_data = Vec::with_capacity(ROWS);
    let mut float_data = Vec::with_capacity(ROWS);
    let mut str_data = Vec::with_capacity(ROWS);
    let mut bool_data = Vec::with_capacity(ROWS);
    
    for i in 0..ROWS {
        int_data.push(i as i64);
        float_data.push(i as f64 / 100.0);
        str_data.push(format!("value_{}", i % 1000)); // 限定的な文字列セット
        bool_data.push(i % 2 == 0);
    }
    
    let start = Instant::now();
    let mut df = OptimizedDataFrame::new();
    
    df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data))).unwrap();
    df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data))).unwrap();
    df.add_column("category".to_string(), Column::String(StringColumn::new(str_data))).unwrap();
    df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data))).unwrap();
    
    let create_time = start.elapsed();
    println!("通常のDataFrame作成時間: {:?}", create_time);
    
    // ====================
    // 直列 vs 並列フィルタリング
    // ====================
    println!("\n[2] フィルタリング処理 (id > 500000)");
    
    // 条件列の追加
    let condition_data: Vec<bool> = (0..ROWS).map(|i| i > ROWS / 2).collect();
    df.add_column(
        "filter_condition".to_string(),
        Column::Boolean(BooleanColumn::new(condition_data))
    ).unwrap();
    
    // 直列フィルタリング
    let start = Instant::now();
    let filtered_df = df.filter("filter_condition").unwrap();
    let serial_time = start.elapsed();
    println!("直列フィルタリング時間: {:?}", serial_time);
    println!("フィルタ結果行数: {}", filtered_df.row_count());
    
    // 並列フィルタリング
    let start = Instant::now();
    let par_filtered_df = df.par_filter("filter_condition").unwrap();
    let parallel_time = start.elapsed();
    println!("並列フィルタリング時間: {:?}", parallel_time);
    println!("フィルタ結果行数: {}", par_filtered_df.row_count());
    
    println!("高速化率: {:.2}倍", serial_time.as_secs_f64() / parallel_time.as_secs_f64());
    
    // ====================
    // グループ化と集計
    // ====================
    println!("\n[3] グループ化と集計 (categoryごとの平均)");
    
    // 直列グループ化（列選択でcategoryとvalueだけを使用）
    let small_df = df.select(&["category", "value"]).unwrap();
    
    let start = Instant::now();
    let lazy_df = LazyFrame::new(small_df.clone());
    let grouped_df = lazy_df
        .aggregate(
            vec!["category".to_string()],
            vec![
                ("value".to_string(), AggregateOp::Mean, "value_mean".to_string())
            ]
        )
        .execute()
        .unwrap();
    let serial_time = start.elapsed();
    println!("直列グループ化・集計時間: {:?}", serial_time);
    println!("グループ数: {}", grouped_df.row_count());
    
    // 並列グループ化
    let start = Instant::now();
    let grouped_map = small_df.par_groupby(&["category"]).unwrap();
    // 集計結果をDataFrameに変換
    let mut result_df = OptimizedDataFrame::new();
    
    // カテゴリ列
    let mut categories = Vec::with_capacity(grouped_map.len());
    let mut means = Vec::with_capacity(grouped_map.len());
    
    for (category, group_df) in &grouped_map {
        categories.push(category.clone());
        
        // 平均値を計算
        let value_col = group_df.column("value").unwrap();
        if let Some(float_col) = value_col.as_float64() {
            let mut sum = 0.0;
            let mut count = 0;
            
            for i in 0..float_col.len() {
                if let Ok(Some(val)) = float_col.get(i) {
                    sum += val;
                    count += 1;
                }
            }
            
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            means.push(mean);
        }
    }
    
    result_df.add_column("category".to_string(), Column::String(StringColumn::new(categories))).unwrap();
    result_df.add_column("value_mean".to_string(), Column::Float64(Float64Column::new(means))).unwrap();
    
    let parallel_time = start.elapsed();
    println!("並列グループ化・集計時間: {:?}", parallel_time);
    println!("グループ数: {}", result_df.row_count());
    
    println!("高速化率: {:.2}倍", serial_time.as_secs_f64() / parallel_time.as_secs_f64());
    
    // ====================
    // 計算処理（すべての値を2倍）
    // ====================
    println!("\n[4] 計算処理（value列の値を2倍）");
    
    // 直列計算
    let start = Instant::now();
    let mut computed_df = OptimizedDataFrame::new();
    
    for name in df.column_names() {
        let col_view = df.column(name).unwrap();
        
        let new_col = if name == "value" {
            let float_col = col_view.as_float64().unwrap();
            let mut doubled_values = Vec::with_capacity(float_col.len());
            
            for i in 0..float_col.len() {
                if let Ok(Some(val)) = float_col.get(i) {
                    doubled_values.push(val * 2.0);
                } else {
                    doubled_values.push(0.0);
                }
            }
            
            Column::Float64(Float64Column::new(doubled_values))
        } else {
            col_view.into_column()
        };
        
        computed_df.add_column(name.to_string(), new_col).unwrap();
    }
    
    let serial_time = start.elapsed();
    println!("直列計算時間: {:?}", serial_time);
    
    // 並列計算
    let start = Instant::now();
    let par_computed_df = df.par_apply(|view| {
        // 列が'value'かどうか判断
        if view.as_float64().is_some() {
            if let Some(float_col) = view.as_float64() {
                use rayon::prelude::*;
                
                let values = (0..float_col.len()).into_par_iter()
                    .map(|i| {
                        if let Ok(Some(val)) = float_col.get(i) {
                            val * 2.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                
                Ok(Column::Float64(Float64Column::new(values)))
            } else {
                Ok(view.clone().into_column())
            }
        } else {
            Ok(view.clone().into_column())
        }
    }).unwrap();
    
    let parallel_time = start.elapsed();
    println!("並列計算時間: {:?}", parallel_time);
    
    println!("高速化率: {:.2}倍", serial_time.as_secs_f64() / parallel_time.as_secs_f64());
    
    println!("\n並列処理ベンチマーク完了");
}