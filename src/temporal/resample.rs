use chrono::TimeZone;
use std::collections::HashMap;

use crate::error::Result;
use crate::temporal::{Frequency, Temporal, TimeSeries};

/// リサンプリング操作を表す構造体
#[derive(Debug)]
pub struct Resample<'a, T: Temporal> {
    /// 元の時系列
    series: &'a TimeSeries<T>,

    /// リサンプリングの周期
    frequency: Frequency,
}

impl<'a, T: Temporal> Resample<'a, T> {
    /// 新しいリサンプリング操作を作成
    pub fn new(series: &'a TimeSeries<T>, frequency: Frequency) -> Self {
        Resample { series, frequency }
    }

    /// 平均値でリサンプリング
    pub fn mean(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| {
            if values.is_empty() {
                return 0.0;
            }
            let sum: f64 = values.iter().sum();
            sum / values.len() as f64
        })
    }

    /// 合計でリサンプリング
    pub fn sum(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| values.iter().sum())
    }

    /// 最大値でリサンプリング
    pub fn max(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| {
            if values.is_empty() {
                return f64::NAN;
            }
            let mut max = values[0];
            for &value in &values[1..] {
                if value > max {
                    max = value;
                }
            }
            max
        })
    }

    /// 最小値でリサンプリング
    pub fn min(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| {
            if values.is_empty() {
                return f64::NAN;
            }
            let mut min = values[0];
            for &value in &values[1..] {
                if value < min {
                    min = value;
                }
            }
            min
        })
    }

    /// カスタム集計関数でリサンプリング
    pub fn aggregate<F>(&self, aggregator: F) -> Result<TimeSeries<T>>
    where
        F: Fn(Vec<f64>) -> f64,
    {
        // 期間ごとにグループ化
        let mut period_groups: HashMap<i64, Vec<f64>> = HashMap::new();
        let freq_seconds = self.frequency.to_seconds();

        // 各データポイントを適切な期間に割り当て
        let start_time = self.series.timestamps()[0].to_utc();
        let start_seconds = start_time.timestamp();

        for (i, timestamp) in self.series.timestamps().iter().enumerate() {
            if let Some(value) = match self.series.values()[i] {
                crate::na::NA::Value(v) => Some(v),
                crate::na::NA::NA => None,
            } {
                // どの期間に属するかを計算
                let ts_seconds = timestamp.to_utc().timestamp();
                let offset = ts_seconds - start_seconds;
                let period = offset / freq_seconds;

                // その期間のグループに追加
                period_groups
                    .entry(period)
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // 期間を時間順にソート
        let mut periods: Vec<i64> = period_groups.keys().cloned().collect();
        periods.sort();

        // 集計結果を作成
        let mut result_values = Vec::with_capacity(periods.len());
        let mut result_timestamps = Vec::with_capacity(periods.len());

        for period in periods {
            // この期間のデータを集計
            if let Some(values) = period_groups.get(&period) {
                let agg_value = aggregator(values.clone());
                result_values.push(crate::na::NA::Value(agg_value));

                // この期間の代表時間を計算
                let period_start_seconds = start_seconds + period * freq_seconds;
                let period_time = chrono::Utc.timestamp_opt(period_start_seconds, 0).unwrap();

                // 適切な型に変換
                let period_timestamp = T::from_str(&period_time.to_rfc3339())?;
                result_timestamps.push(period_timestamp);
            }
        }

        // 新しい時系列を作成
        TimeSeries::new(
            result_values,
            result_timestamps,
            self.series.name().cloned(),
        )
        .map(|ts| ts.with_frequency(self.frequency.clone()))
    }
}
