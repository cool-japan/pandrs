use chrono::{DateTime, NaiveDate, NaiveDateTime, Duration, Utc, Datelike};
use crate::error::{PandRSError, Result};
use crate::temporal::{Temporal, Frequency};

/// 日付範囲を生成する構造体
#[derive(Debug, Clone)]
pub struct DateRange<T: Temporal> {
    start: T,
    end: T,
    freq: Frequency,
    inclusive: bool,
}

impl<T: Temporal> DateRange<T> {
    /// 開始・終了・頻度から日付範囲を作成
    pub fn new(start: T, end: T, freq: Frequency, inclusive: bool) -> Result<Self> {
        if start > end {
            return Err(PandRSError::Consistency(
                "開始日時は終了日時よりも前である必要があります".to_string()
            ));
        }
        
        Ok(DateRange { start, end, freq, inclusive })
    }
    
    /// 日付範囲内の全ての時点を取得
    pub fn generate(&self) -> Vec<T> {
        let mut result = Vec::new();
        let mut current = self.start.clone();
        
        // 最初の日時を追加
        result.push(current.clone());
        
        loop {
            // 次の日時に進む
            current = match self.freq {
                Frequency::Secondly => current.add(Duration::seconds(1)),
                Frequency::Minutely => current.add(Duration::minutes(1)),
                Frequency::Hourly => current.add(Duration::hours(1)),
                Frequency::Daily => current.add(Duration::days(1)),
                Frequency::Weekly => current.add(Duration::weeks(1)),
                Frequency::Monthly => {
                    // 月単位での加算は複雑なので、UTCに変換してから操作
                    let utc = current.to_utc();
                    let naive = utc.naive_utc();
                    
                    // 年と月を計算
                    let mut year = naive.year();
                    let mut month = naive.month() + 1;
                    
                    if month > 12 {
                        month = 1;
                        year += 1;
                    }
                    
                    // 新しい日を算出（月の末日を超えないように調整）
                    let day = naive.day().min(days_in_month(year, month));
                    
                    // 新しい日時を作成
                    let new_naive = NaiveDateTime::new(
                        NaiveDate::from_ymd_opt(year, month, day).unwrap(),
                        naive.time()
                    );
                    
                    // 元の型に戻す
                    let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
                    T::from_str(&new_utc.to_rfc3339()).unwrap()
                }
                Frequency::Quarterly => {
                    // 四半期単位での加算（3ヶ月ごと）
                    let utc = current.to_utc();
                    let naive = utc.naive_utc();
                    
                    // 年と月を計算
                    let mut year = naive.year();
                    let mut month = naive.month() + 3;
                    
                    if month > 12 {
                        month = month - 12;
                        year += 1;
                    }
                    
                    // 新しい日を算出（月の末日を超えないように調整）
                    let day = naive.day().min(days_in_month(year, month));
                    
                    // 新しい日時を作成
                    let new_naive = NaiveDateTime::new(
                        NaiveDate::from_ymd_opt(year, month, day).unwrap(),
                        naive.time()
                    );
                    
                    // 元の型に戻す
                    let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
                    T::from_str(&new_utc.to_rfc3339()).unwrap()
                }
                Frequency::Yearly => {
                    // 年単位での加算
                    let utc = current.to_utc();
                    let naive = utc.naive_utc();
                    
                    // 年を増やす
                    let year = naive.year() + 1i32;
                    let month = naive.month();
                    
                    // 2月29日の場合は、閏年でない場合は28日に調整
                    let day = if month == 2 && naive.day() == 29 && !is_leap_year(year as i32) {
                        28
                    } else {
                        naive.day()
                    };
                    
                    // 新しい日時を作成
                    let new_naive = NaiveDateTime::new(
                        NaiveDate::from_ymd_opt(year, month, day).unwrap(),
                        naive.time()
                    );
                    
                    // 元の型に戻す
                    let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
                    T::from_str(&new_utc.to_rfc3339()).unwrap()
                }
                Frequency::Custom(duration) => current.add(duration),
            };
            
            // 終了チェック
            if self.inclusive {
                if current > self.end {
                    break;
                }
            } else if current >= self.end {
                break;
            }
            
            result.push(current.clone());
        }
        
        result
    }
}

/// 指定された年・月の日数を返す
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => panic!("Invalid month: {}", month),
    }
}

/// 閏年かどうかを判定
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// 日付範囲を生成するヘルパー関数
pub fn date_range<T: Temporal>(
    start: T,
    end: T,
    freq: Frequency,
    inclusive: bool
) -> Result<Vec<T>> {
    DateRange::new(start, end, freq, inclusive).map(|range| range.generate())
}