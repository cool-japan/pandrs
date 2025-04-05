// PandRS統計関数サンプル

use pandrs::{DataFrame, Series, TTestResult};
use pandrs::error::Result;
use rand::Rng;

fn main() -> Result<()> {
    println!("PandRS 統計モジュールサンプル\n");
    
    // 記述統計の例
    descriptive_stats_example()?;
    
    // 推測統計・仮説検定の例
    ttest_example()?;
    
    // 回帰分析の例
    regression_example()?;
    
    Ok(())
}

fn descriptive_stats_example() -> Result<()> {
    println!("1. 記述統計サンプル");
    println!("-----------------");
    
    // データセットの作成
    let mut df = DataFrame::new();
    let values = Series::new(vec![10.5, 12.3, 15.2, 9.8, 11.5, 13.7, 14.3, 12.9, 8.5, 10.2], 
                             Some("値".to_string()))?;
    
    df.add_column("値".to_string(), values)?;
    
    // 記述統計
    let stats = pandrs::stats::describe(df.get_column("値").unwrap().values().iter()
        .map(|v| v.parse::<f64>().unwrap_or(0.0))
        .collect::<Vec<f64>>())?;
    
    // 結果の表示
    println!("データ数: {}", stats.count);
    println!("平均値: {:.2}", stats.mean);
    println!("標準偏差: {:.2}", stats.std);
    println!("最小値: {:.2}", stats.min);
    println!("第一四分位数: {:.2}", stats.q1);
    println!("中央値: {:.2}", stats.median);
    println!("第三四分位数: {:.2}", stats.q3);
    println!("最大値: {:.2}", stats.max);
    
    // 共分散と相関係数
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data2 = vec![1.5, 3.1, 4.2, 5.8, 7.1];
    
    let cov = pandrs::stats::covariance(&data1, &data2)?;
    let corr = pandrs::stats::correlation(&data1, &data2)?;
    
    println!("\n共分散と相関係数:");
    println!("共分散: {:.4}", cov);
    println!("相関係数: {:.4}", corr);
    
    println!();
    Ok(())
}

fn ttest_example() -> Result<()> {
    println!("2. t検定サンプル");
    println!("--------------");
    
    // サンプルデータの作成
    let group1 = vec![5.2, 5.8, 6.1, 5.5, 5.9, 6.2, 5.7, 6.0, 5.6, 5.8];
    let group2 = vec![4.8, 5.1, 5.3, 4.9, 5.0, 5.2, 4.7, 5.1, 4.9, 5.0];
    
    // 有意水準 0.05 (5%)でt検定を実行
    let alpha = 0.05;
    
    // 等分散を仮定した場合のt検定
    let result_equal = pandrs::stats::ttest(&group1, &group2, alpha, true)?;
    
    println!("等分散を仮定したt検定結果:");
    print_ttest_result(&result_equal);
    
    // 等分散を仮定しない場合のt検定 (Welchのt検定)
    let result_welch = pandrs::stats::ttest(&group1, &group2, alpha, false)?;
    
    println!("\nWelchのt検定結果 (等分散を仮定しない):");
    print_ttest_result(&result_welch);
    
    println!();
    Ok(())
}

fn print_ttest_result(result: &TTestResult) {
    println!("t統計量: {:.4}", result.statistic);
    println!("p値: {:.4}", result.pvalue);
    println!("自由度: {}", result.df);
    println!("有意差: {}", if result.significant { "あり" } else { "なし" });
}

fn regression_example() -> Result<()> {
    println!("3. 回帰分析サンプル");
    println!("-----------------");
    
    // データセットの作成
    let mut df = DataFrame::new();
    
    // 説明変数
    let x1 = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], Some("x1".to_string()))?;
    let x2 = Series::new(vec![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0], Some("x2".to_string()))?;
    
    // 目的変数 (y = 2*x1 + 1.5*x2 + 3 + ノイズ)
    let mut y_values = Vec::with_capacity(10);
    let mut rng = rand::thread_rng();
    
    for i in 0..10 {
        let noise = rng.gen_range(-1.0..1.0);
        let y_val = 2.0 * (i as f64 + 1.0) + 1.5 * (5.0 + 3.0 * i as f64) + 3.0 + noise;
        y_values.push(y_val);
    }
    
    let y = Series::new(y_values, Some("y".to_string()))?;
    
    // DataFrameに追加
    df.add_column("x1".to_string(), x1)?;
    df.add_column("x2".to_string(), x2)?;
    df.add_column("y".to_string(), y)?;
    
    // 回帰分析の実行
    let model = pandrs::stats::linear_regression(&df, "y", &["x1", "x2"])?;
    
    // 結果の表示
    println!("線形回帰モデル: y = {:.4} + {:.4} × x1 + {:.4} × x2", 
             model.intercept, model.coefficients[0], model.coefficients[1]);
    println!("決定係数 R²: {:.4}", model.r_squared);
    println!("調整済み決定係数: {:.4}", model.adj_r_squared);
    println!("回帰係数のp値: {:?}", model.p_values);
    
    // 単回帰の例
    println!("\n単回帰モデル（x1のみ）:");
    let model_simple = pandrs::stats::linear_regression(&df, "y", &["x1"])?;
    println!("線形回帰モデル: y = {:.4} + {:.4} × x1", 
             model_simple.intercept, model_simple.coefficients[0]);
    println!("決定係数 R²: {:.4}", model_simple.r_squared);
    
    Ok(())
}