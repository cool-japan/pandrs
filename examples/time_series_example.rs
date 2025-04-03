use chrono::NaiveDate;
use std::str::FromStr;
use pandrs::{PandRSError, NA};
use pandrs::temporal::{TimeSeries, date_range, Frequency};

fn main() -> Result<(), PandRSError> {
    println!("=== 時系列データの例 ===");
    
    // 日付範囲を作成
    let start_date = NaiveDate::from_str("2023-01-01").map_err(|e| PandRSError::Format(e.to_string()))?;
    let end_date = NaiveDate::from_str("2023-01-31").map_err(|e| PandRSError::Format(e.to_string()))?;
    
    // 日次の日付範囲を生成
    let dates = date_range(start_date, end_date, Frequency::Daily, true)?;
    println!("日付範囲の長さ: {}", dates.len());
    println!("最初の日付: {}", dates[0]);
    println!("最後の日付: {}", dates[dates.len() - 1]);
    
    // 時系列データを生成
    let mut values = Vec::with_capacity(dates.len());
    for i in 0..dates.len() {
        // サンプルデータを生成（単純な正弦波）
        let day = i as f64;
        let value = 100.0 + 10.0 * (day * 0.2).sin();
        
        // 欠損値を含める（5日ごとに）
        if i % 5 == 0 {
            values.push(NA::NA);
        } else {
            values.push(NA::Value(value));
        }
    }
    
    // TimeSeries を作成
    let time_series = TimeSeries::new(values, dates, Some("daily_values".to_string()))?;
    
    println!("\n=== 時系列基本情報 ===");
    println!("長さ: {}", time_series.len());
    
    // 時間フィルタリング
    let start_filter = NaiveDate::from_str("2023-01-10").map_err(|e| PandRSError::Format(e.to_string()))?;
    let end_filter = NaiveDate::from_str("2023-01-20").map_err(|e| PandRSError::Format(e.to_string()))?;
    let filtered = time_series.filter_by_time(&start_filter, &end_filter)?;
    
    println!("\n=== 時間フィルタリング結果 ===");
    println!("元の時系列長: {}", time_series.len());
    println!("フィルター後の長さ: {}", filtered.len());
    
    // 移動平均の計算
    let window_size = 3;
    let moving_avg = time_series.rolling_mean(window_size)?;
    
    println!("\n=== 移動平均 (ウィンドウサイズ: {}) ===", window_size);
    println!("移動平均の長さ: {}", moving_avg.len());
    
    // 最初のいくつかの値を表示
    println!("\n=== 元のデータと移動平均の比較 (最初の10行) ===");
    println!("日付\t\tオリジナル\t移動平均");
    for i in 0..10.min(time_series.len()) {
        let date = time_series.timestamps()[i];
        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        let ma = match moving_avg.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        println!("{}\t{}\t\t{}", date, original, ma);
    }
    
    // 週次へのリサンプリング
    let weekly = time_series.resample(Frequency::Weekly).mean()?;
    
    println!("\n=== 週次リサンプリング ===");
    println!("元の時系列長: {}", time_series.len());
    println!("リサンプル後の長さ: {}", weekly.len());
    
    // 週次データを表示
    println!("\n=== 週次データ ===");
    println!("日付\t\t値");
    for i in 0..weekly.len() {
        let date = weekly.timestamps()[i];
        let value = match weekly.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        println!("{}\t{}", date, value);
    }
    
    println!("\n=== 時系列サンプル完了 ===");
    Ok(())
}