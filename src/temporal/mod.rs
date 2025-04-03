//! 時系列データ操作のためのモジュール

mod date_range;
mod frequency;
mod resample;

use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use std::ops::{Add, Sub};

use crate::error::{PandRSError, Result};
use crate::na::NA;

pub use self::date_range::{date_range, DateRange};
pub use self::frequency::Frequency;
pub use self::resample::Resample;

/// 日時を表す型のトレイト
pub trait Temporal:
    Clone
    + std::fmt::Debug
    + PartialOrd
    + Add<Duration, Output = Self>
    + Sub<Duration, Output = Self>
    + 'static
{
    /// 現在時刻を取得
    fn now() -> Self;

    /// 2つの時間の差を取得
    fn duration_between(&self, other: &Self) -> Duration;

    /// UTCタイムゾーンに変換
    fn to_utc(&self) -> DateTime<Utc>;

    /// 文字列から変換
    fn from_str(s: &str) -> Result<Self>;

    /// 文字列に変換
    fn to_string(&self) -> String;
}

// Chronoの各種日時型に対するTemporalトレイトの実装

impl Temporal for DateTime<Utc> {
    fn now() -> Self {
        Utc::now()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        if self > other {
            *self - *other
        } else {
            *other - *self
        }
    }

    fn to_utc(&self) -> DateTime<Utc> {
        *self
    }

    fn from_str(s: &str) -> Result<Self> {
        match s.parse::<DateTime<Utc>>() {
            Ok(dt) => Ok(dt),
            Err(e) => Err(PandRSError::Format(format!("日時の解析エラー: {}", e))),
        }
    }

    fn to_string(&self) -> String {
        self.to_rfc3339()
    }
}

impl Temporal for DateTime<Local> {
    fn now() -> Self {
        Local::now()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        if self > other {
            *self - *other
        } else {
            *other - *self
        }
    }

    fn to_utc(&self) -> DateTime<Utc> {
        self.with_timezone(&Utc)
    }

    fn from_str(s: &str) -> Result<Self> {
        match s.parse::<DateTime<Local>>() {
            Ok(dt) => Ok(dt),
            Err(e) => Err(PandRSError::Format(format!("日時の解析エラー: {}", e))),
        }
    }

    fn to_string(&self) -> String {
        self.to_rfc3339()
    }
}

impl Temporal for NaiveDateTime {
    fn now() -> Self {
        Local::now().naive_local()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        if self > other {
            *self - *other
        } else {
            *other - *self
        }
    }

    fn to_utc(&self) -> DateTime<Utc> {
        // NaiveDateTimeはタイムゾーン情報を持たないため、UTCと仮定
        DateTime::<Utc>::from_naive_utc_and_offset(*self, Utc)
    }

    fn from_str(s: &str) -> Result<Self> {
        match NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            Ok(dt) => Ok(dt),
            Err(_) => match NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
                Ok(dt) => Ok(dt),
                Err(e) => Err(PandRSError::Format(format!("日時の解析エラー: {}", e))),
            },
        }
    }

    fn to_string(&self) -> String {
        self.format("%Y-%m-%d %H:%M:%S").to_string()
    }
}

impl Temporal for NaiveDate {
    fn now() -> Self {
        Local::now().date_naive()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        let days = if self > other {
            (*self - *other).num_days()
        } else {
            (*other - *self).num_days()
        };
        Duration::days(days)
    }

    fn to_utc(&self) -> DateTime<Utc> {
        // 日付にデフォルトの時間（00:00:00）を追加してUTCとして扱う
        let naive_dt = self.and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc)
    }

    fn from_str(s: &str) -> Result<Self> {
        // Try parsing with standard format first
        match NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            Ok(dt) => Ok(dt),
            Err(_) => {
                // Try parsing from RFC3339 format (from DateTime)
                match chrono::DateTime::parse_from_rfc3339(s) {
                    Ok(dt) => Ok(dt.date_naive()),
                    Err(e) => Err(PandRSError::Format(format!("日付の解析エラー: {}", e))),
                }
            }
        }
    }

    fn to_string(&self) -> String {
        self.format("%Y-%m-%d").to_string()
    }
}

/// 時系列シリーズを表す構造体
#[derive(Debug, Clone)]
pub struct TimeSeries<T: Temporal> {
    /// 時系列データの値
    values: Vec<NA<f64>>,

    /// 時系列の時間インデックス
    timestamps: Vec<T>,

    /// シリーズの名前
    name: Option<String>,

    /// 周波数（オプション）
    frequency: Option<Frequency>,
}

impl<T: Temporal> TimeSeries<T> {
    /// 新しい時系列シリーズを作成
    pub fn new(values: Vec<NA<f64>>, timestamps: Vec<T>, name: Option<String>) -> Result<Self> {
        if values.len() != timestamps.len() {
            return Err(PandRSError::Consistency(format!(
                "値の長さ ({}) と時間インデックスの長さ ({}) が一致しません",
                values.len(),
                timestamps.len()
            )));
        }

        Ok(TimeSeries {
            values,
            timestamps,
            name,
            frequency: None,
        })
    }

    /// 長さを取得
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 空かどうか
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 名前を取得
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// タイムスタンプを取得
    pub fn timestamps(&self) -> &[T] {
        &self.timestamps
    }

    /// 値を取得
    pub fn values(&self) -> &[NA<f64>] {
        &self.values
    }

    /// 周波数を取得
    pub fn frequency(&self) -> Option<&Frequency> {
        self.frequency.as_ref()
    }

    /// 周波数を設定
    pub fn with_frequency(mut self, freq: Frequency) -> Self {
        self.frequency = Some(freq);
        self
    }

    /// 指定された時間範囲でフィルタリング
    pub fn filter_by_time(&self, start: &T, end: &T) -> Result<Self> {
        let mut filtered_values = Vec::new();
        let mut filtered_timestamps = Vec::new();

        for (i, ts) in self.timestamps.iter().enumerate() {
            if ts >= start && ts <= end {
                filtered_values.push(self.values[i].clone());
                filtered_timestamps.push(ts.clone());
            }
        }

        Self::new(filtered_values, filtered_timestamps, self.name.clone())
    }

    /// 指定された期間でリサンプリング
    pub fn resample(&self, freq: Frequency) -> Resample<T> {
        Resample::new(self, freq)
    }

    /// 移動平均を計算
    pub fn rolling_mean(&self, window: usize) -> Result<Self> {
        if window > self.len() || window == 0 {
            return Err(PandRSError::Consistency(format!(
                "ウィンドウサイズ ({}) が無効です。1以上かつデータ長 ({}) 以下である必要があります。",
                window, self.len()
            )));
        }

        let mut result_values = Vec::with_capacity(self.len());

        // 移動平均の計算
        for i in 0..self.len() {
            if i < window - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得 (using checked arithmetic to avoid overflow)
                let start_idx = i.checked_sub(window - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.values[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let sum: f64 = window_values.iter().sum();
                    let mean = sum / window_values.len() as f64;
                    result_values.push(NA::Value(mean));
                }
            }
        }

        Self::new(result_values, self.timestamps.clone(), self.name.clone())
    }
}
