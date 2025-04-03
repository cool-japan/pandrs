use chrono::NaiveDate;
use pandrs::temporal::{date_range, Frequency, TimeSeries};
use pandrs::{PandRSError, NA};
use std::str::FromStr;

fn main() -> Result<(), PandRSError> {
    println!("=== ウィンドウ操作の例 ===\n");

    // 日付範囲を作成
    let start_date =
        NaiveDate::from_str("2023-01-01").map_err(|e| PandRSError::Format(e.to_string()))?;
    let end_date =
        NaiveDate::from_str("2023-01-20").map_err(|e| PandRSError::Format(e.to_string()))?;

    // 日次の日付範囲を生成
    let dates = date_range(start_date, end_date, Frequency::Daily, true)?;
    
    // 時系列データを生成（単純な直線+ノイズ）
    let mut values = Vec::with_capacity(dates.len());
    for i in 0..dates.len() {
        let value = 100.0 + i as f64 * 2.0 + (i as f64 * 0.5).sin() * 5.0;
        
        // 一部の値を欠損値にする（7日ごとに）
        if i % 7 == 0 {
            values.push(NA::NA);
        } else {
            values.push(NA::Value(value));
        }
    }

    // TimeSeries を作成
    let time_series = TimeSeries::new(values, dates, Some("sample_data".to_string()))?;

    // データの表示
    println!("=== 元のデータ ===");
    println!("日付\t\t値");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];
        let value = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        println!("{}\t{}", date, value);
    }

    // 1. 固定長ウィンドウ操作 (Rolling)
    println!("\n=== 固定長ウィンドウ操作 (ウィンドウサイズ: 3) ===");
    
    // 1.1 移動平均
    let window_size = 3;
    let rolling_mean = time_series.rolling(window_size)?.mean()?;
    
    // 1.2 移動合計
    let rolling_sum = time_series.rolling(window_size)?.sum()?;
    
    // 1.3 移動標準偏差
    let rolling_std = time_series.rolling(window_size)?.std(1)?;
    
    // 1.4 移動最小値
    let rolling_min = time_series.rolling(window_size)?.min()?;
    
    // 1.5 移動最大値
    let rolling_max = time_series.rolling(window_size)?.max()?;
    
    // 結果を表示
    println!("日付\t\t元データ\t移動平均\t移動合計\t移動標準偏差\t移動最小値\t移動最大値");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];
        
        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let mean = match rolling_mean.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let sum = match rolling_sum.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let std = match rolling_std.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let min = match rolling_min.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let max = match rolling_max.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}", date, original, mean, sum, std, min, max);
    }

    // 2. 拡大ウィンドウ操作 (Expanding)
    println!("\n=== 拡大ウィンドウ操作 (最小期間: 3) ===");
    
    // 2.1 拡大平均
    let min_periods = 3; // 最低3つのデータポイントが必要
    let expanding_mean = time_series.expanding(min_periods)?.mean()?;
    
    // 2.2 拡大合計
    let expanding_sum = time_series.expanding(min_periods)?.sum()?;
    
    // 結果を表示
    println!("日付\t\t元データ\t拡大平均\t拡大合計");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];
        
        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let exp_mean = match expanding_mean.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let exp_sum = match expanding_sum.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        println!("{}\t{}\t{}\t{}", date, original, exp_mean, exp_sum);
    }

    // 3. 指数加重ウィンドウ操作 (EWM)
    println!("\n=== 指数加重ウィンドウ操作 (span: 5) ===");
    
    // 3.1 EWM平均 (spanを指定)
    let span = 5; // 半減期
    let ewm_mean = time_series.ewm(Some(span), None, false)?.mean()?;
    
    // 3.2 EWM標準偏差
    let ewm_std = time_series.ewm(Some(span), None, false)?.std(1)?;
    
    // 3.3 別のalphaでのEWM
    let alpha = 0.3; // 直接alpha値を指定
    let ewm_mean_alpha = time_series.ewm(None, Some(alpha), false)?.mean()?;
    
    // 結果を表示
    println!("日付\t\t元データ\tEWM平均(span=5)\tEWM標準偏差\tEWM平均(alpha=0.3)");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];
        
        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let ewm = match ewm_mean.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let ewm_s = match ewm_std.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let ewm_a = match ewm_mean_alpha.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        println!("{}\t{}\t{}\t{}\t{}", date, original, ewm, ewm_s, ewm_a);
    }

    // 4. カスタム集計関数の使用例
    println!("\n=== カスタム集計関数の例 (中央値) ===");
    
    // 中央値を計算するカスタム関数
    let median = |values: &[f64]| -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };
    
    // カスタム集計関数を適用
    let rolling_median = time_series.rolling(window_size)?.aggregate(median, Some(1))?;
    
    // 結果を表示
    println!("日付\t\t元データ\t移動中央値");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];
        
        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        let med = match rolling_median.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        
        println!("{}\t{}\t{}", date, original, med);
    }

    println!("\n=== ウィンドウ操作の例を完了 ===");
    Ok(())
}