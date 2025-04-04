use std::error::Error;
use std::time::Instant;

use pandrs::{OptimizedDataFrame, LazyFrame, AggregateOp, Column, Int64Column, Float64Column, StringColumn};
use pandrs::column::ColumnTrait;

fn main() -> Result<(), Box<dyn Error>> {
    println!("遅延評価と並列処理の組み合わせによるパフォーマンス評価\n");
    
    // 大きなデータフレームを生成
    println!("大きなデータセットを生成中...");
    let rows = 100_000;
    let df = generate_large_dataframe(rows)?;
    println!("{}行のデータを生成しました", rows);
    
    // データフレームの一部を表示
    println!("\nデータの最初の行:");
    println!("{:?}\n", df);
    
    // 標準的なアプローチでフィルタリングと集計
    println!("標準的なアプローチでのデータ処理を実行中...");
    let start = Instant::now();
    
    // 年齢フィルター（30歳以上）を作成
    let age_col = df.column("年齢")?;
    let mut age_filter = vec![false; df.row_count()];
    if let Some(int_col) = age_col.as_int64() {
        for i in 0..df.row_count() {
            if let Ok(Some(age)) = int_col.get(i) {
                age_filter[i] = age >= 30;
            }
        }
    }
    
    // フィルターをDataFrameに追加
    let bool_data = pandrs::BooleanColumn::new(age_filter);
    let mut df_with_filter = df.clone();
    df_with_filter.add_column("30歳以上", Column::Boolean(bool_data))?;
    
    // フィルタリングを実行
    let filtered_df = df_with_filter.filter("30歳以上")?;
    
    // 部門ごとの集計を手動で行う
    let dept_col = filtered_df.column("部門")?;
    let salary_col = filtered_df.column("給与")?;
    
    // 部門ごとの集計用
    let mut dept_totals: std::collections::HashMap<String, (f64, i32)> = std::collections::HashMap::new();
    
    if let (Some(str_col), Some(float_col)) = (dept_col.as_string(), salary_col.as_float64()) {
        for i in 0..filtered_df.row_count() {
            if let (Ok(Some(dept)), Ok(Some(salary))) = (str_col.get(i), float_col.get(i)) {
                let entry = dept_totals.entry(dept.to_string()).or_insert((0.0, 0));
                entry.0 += salary;
                entry.1 += 1;
            }
        }
    }
    
    // 結果を構築
    let mut result_depts = Vec::new();
    let mut result_totals = Vec::new();
    let mut result_avgs = Vec::new();
    let mut result_counts = Vec::new();
    
    for (dept, (total, count)) in dept_totals {
        result_depts.push(dept);
        result_totals.push(total);
        result_avgs.push(total / count as f64);
        result_counts.push(count as f64);
    }
    
    // 結果のDataFrameを作成
    let mut result_df = OptimizedDataFrame::new();
    result_df.add_column("部門", Column::String(StringColumn::new(result_depts)))?;
    result_df.add_column("合計給与", Column::Float64(Float64Column::new(result_totals)))?;
    result_df.add_column("平均給与", Column::Float64(Float64Column::new(result_avgs)))?;
    result_df.add_column("人数", Column::Float64(Float64Column::new(result_counts)))?;
    
    let standard_duration = start.elapsed();
    println!("標準的アプローチの処理時間: {:?}", standard_duration);
    println!("\n標準的アプローチの結果:");
    println!("{:?}\n", result_df);
    
    // LazyFrameと並列処理を使ったアプローチ
    println!("LazyFrameと並列処理を使ったアプローチ...");
    let start = Instant::now();
    
    // LazyFrameを使用した処理定義
    let lazy_df = LazyFrame::new(df);
    let result_lazy = lazy_df
        // 年齢でフィルタリング（内部でブール列を作成）
        .map(|col| {
            if col.column_type() == pandrs::ColumnType::Int64 && 
               matches!(col.name(), Some("年齢")) {
                // Int64Columnから年齢>=30のブール列を作成
                if let pandrs::Column::Int64(age_col) = col {
                    let mut filter = vec![false; age_col.len()];
                    for i in 0..age_col.len() {
                        if let Ok(Some(age)) = age_col.get(i) {
                            filter[i] = age >= 30;
                        }
                    }
                    return Ok(Column::Boolean(pandrs::BooleanColumn::new(filter)));
                }
            }
            // その他の列はそのまま返す
            Ok(col.clone())
        })
        .filter("年齢") // 直前でtrueに変換した列を使ってフィルタリング
        .aggregate(
            vec!["部門".to_string()],
            vec![
                ("給与".to_string(), AggregateOp::Sum, "合計給与".to_string()),
                ("給与".to_string(), AggregateOp::Mean, "平均給与".to_string()),
                ("給与".to_string(), AggregateOp::Count, "人数".to_string()),
            ]
        );
    
    // 実行計画の表示
    println!("\n実行計画:");
    println!("{}", result_lazy.explain());
    
    // 実行
    let lazy_result = result_lazy.execute()?;
    
    let lazy_duration = start.elapsed();
    println!("LazyFrameアプローチの処理時間: {:?}", lazy_duration);
    println!("\nLazyFrameアプローチの結果:");
    println!("{:?}\n", lazy_result);
    
    // パフォーマンス比較
    let speedup = standard_duration.as_secs_f64() / lazy_duration.as_secs_f64();
    println!("LazyFrameアプローチは標準的アプローチの{:.2}倍高速です", speedup);
    
    Ok(())
}

// 大きなデータフレームを生成する関数
fn generate_large_dataframe(rows: usize) -> Result<OptimizedDataFrame, Box<dyn Error>> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let mut rng = StdRng::seed_from_u64(42); // 再現性のため固定シード
    
    // データを生成
    let mut ids = Vec::with_capacity(rows);
    let mut ages = Vec::with_capacity(rows);
    let mut depts = Vec::with_capacity(rows);
    let mut salaries = Vec::with_capacity(rows);
    
    // 部門のリスト
    let departments = vec![
        "営業部".to_string(), 
        "開発部".to_string(), 
        "人事部".to_string(),
        "財務部".to_string(),
        "マーケティング部".to_string(),
    ];
    
    for i in 0..rows {
        ids.push(i as i64 + 1000); // ID
        ages.push(rng.gen_range(20..60)); // 年齢
        depts.push(departments[rng.gen_range(0..departments.len())].clone()); // 部門
        
        // 給与（部門と年齢に基づいて乱数を生成）
        let base_salary = match depts.last().unwrap().as_str() {
            "営業部" => 350_000.0,
            "開発部" => 400_000.0,
            "人事部" => 320_000.0,
            "財務部" => 380_000.0,
            "マーケティング部" => 360_000.0,
            _ => 300_000.0,
        };
        
        let age_factor = *ages.last().unwrap() as f64 / 30.0;
        let variation = rng.gen_range(0.8..1.2);
        
        salaries.push(base_salary * age_factor * variation);
    }
    
    // DataFrameを作成
    let mut df = OptimizedDataFrame::new();
    df.add_column("ID", Column::Int64(Int64Column::new(ids)))?;
    df.add_column("年齢", Column::Int64(Int64Column::new(ages)))?;
    df.add_column("部門", Column::String(StringColumn::new(depts)))?;
    df.add_column("給与", Column::Float64(Float64Column::new(salaries)))?;
    
    Ok(df)
}