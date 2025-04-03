//! 時系列データのウィンドウ操作を行うモジュール

use std::fmt;

use crate::error::{PandRSError, Result};
use crate::na::NA;
use crate::temporal::TimeSeries;
use crate::temporal::Temporal;

/// ウィンドウの種類を定義する列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// 固定長ウィンドウ（Rolling Window）
    /// 固定サイズのウィンドウをスライドさせる操作
    Fixed,
    
    /// 拡大ウィンドウ（Expanding Window）
    /// 最初の点から現在の点までのすべての点を含むウィンドウ
    Expanding,
    
    /// 指数加重ウィンドウ（Exponentially Weighted Window）
    /// 直近のデータに高い重みを与えるウィンドウ
    ExponentiallyWeighted,
}

impl fmt::Display for WindowType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WindowType::Fixed => write!(f, "Fixed"),
            WindowType::Expanding => write!(f, "Expanding"),
            WindowType::ExponentiallyWeighted => write!(f, "ExponentiallyWeighted"),
        }
    }
}

/// ウィンドウ操作を行うための構造体
#[derive(Debug)]
pub struct Window<'a, T: Temporal> {
    /// 元の時系列データへの参照
    time_series: &'a TimeSeries<T>,
    
    /// ウィンドウの種類
    window_type: WindowType,
    
    /// ウィンドウのサイズ
    window_size: usize,
    
    /// 指数加重用の減衰係数（alpha）
    /// 0.0 < alpha <= 1.0、大きい値ほど直近のデータに高い重みを与える
    alpha: Option<f64>,
}

impl<'a, T: Temporal> Window<'a, T> {
    /// 新しいウィンドウ操作インスタンスを作成
    pub fn new(
        time_series: &'a TimeSeries<T>,
        window_type: WindowType,
        window_size: usize,
    ) -> Result<Self> {
        // ウィンドウサイズのバリデーション
        if window_size == 0 || (window_type == WindowType::Fixed && window_size > time_series.len()) {
            return Err(PandRSError::Consistency(format!(
                "ウィンドウサイズ ({}) が無効です。1以上かつデータ長 ({}) 以下である必要があります。",
                window_size, time_series.len()
            )));
        }
        
        Ok(Window {
            time_series,
            window_type,
            window_size,
            alpha: None,
        })
    }
    
    /// 指数加重ウィンドウの減衰係数を設定
    /// alpha: 0.0 < alpha <= 1.0、大きい値ほど直近のデータに高い重みを与える
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(PandRSError::Consistency(format!(
                "減衰係数 alpha ({}) は0より大きく1以下である必要があります。", alpha
            )));
        }
        
        self.alpha = Some(alpha);
        Ok(self)
    }
    
    /// 平均を計算
    pub fn mean(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_mean(),
            WindowType::Expanding => self.expanding_window_mean(),
            WindowType::ExponentiallyWeighted => self.ewm_mean(),
        }
    }
    
    /// 合計を計算
    pub fn sum(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_sum(),
            WindowType::Expanding => self.expanding_window_sum(),
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("指数加重ウィンドウでは合計操作はサポートされていません。".to_string()))
            }
        }
    }
    
    /// 標準偏差を計算
    pub fn std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_std(ddof),
            WindowType::Expanding => self.expanding_window_std(ddof),
            WindowType::ExponentiallyWeighted => self.ewm_std(ddof),
        }
    }
    
    /// 最小値を計算
    pub fn min(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_min(),
            WindowType::Expanding => self.expanding_window_min(),
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("指数加重ウィンドウでは最小値操作はサポートされていません。".to_string()))
            }
        }
    }
    
    /// 最大値を計算
    pub fn max(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_max(),
            WindowType::Expanding => self.expanding_window_max(),
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("指数加重ウィンドウでは最大値操作はサポートされていません。".to_string()))
            }
        }
    }
    
    /// 一般的な集計操作を適用
    pub fn aggregate<F>(&self, agg_func: F, min_periods: Option<usize>) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let min_periods = min_periods.unwrap_or(1);
        if min_periods == 0 {
            return Err(PandRSError::Consistency(
                "min_periods は1以上である必要があります。".to_string(),
            ));
        }
        
        match self.window_type {
            WindowType::Fixed => self.fixed_window_aggregate(agg_func, min_periods),
            WindowType::Expanding => self.expanding_window_aggregate(agg_func, min_periods),
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("指数加重ウィンドウでは一般的な集計操作はサポートされていません。".to_string()))
            }
        }
    }
    
    // 以下、各ウィンドウタイプごとの実装
    
    // ------- 固定ウィンドウ実装 -------
    
    /// 固定長ウィンドウの平均を計算
    fn fixed_window_mean(&self) -> Result<TimeSeries<T>> {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 移動平均の計算
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
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
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 固定長ウィンドウの合計を計算
    fn fixed_window_sum(&self) -> Result<TimeSeries<T>> {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 移動合計の計算
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
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
                    result_values.push(NA::Value(sum));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 固定長ウィンドウの標準偏差を計算
    fn fixed_window_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 移動標準偏差の計算
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.len() <= ddof {
                    result_values.push(NA::NA);
                } else {
                    // 平均を計算
                    let mean: f64 = window_values.iter().sum::<f64>() / window_values.len() as f64;
                    
                    // 分散を計算
                    let variance: f64 = window_values
                        .iter()
                        .map(|v| (*v - mean).powi(2))
                        .sum::<f64>()
                        / (window_values.len() - ddof) as f64;
                    
                    // 標準偏差を計算
                    let std_dev = variance.sqrt();
                    result_values.push(NA::Value(std_dev));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 固定長ウィンドウの最小値を計算
    fn fixed_window_min(&self) -> Result<TimeSeries<T>> {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 移動最小値の計算
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let min = window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    result_values.push(NA::Value(min));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 固定長ウィンドウの最大値を計算
    fn fixed_window_max(&self) -> Result<TimeSeries<T>> {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 移動最大値の計算
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let max = window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    result_values.push(NA::Value(max));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 固定長ウィンドウに一般的な集計関数を適用
    fn fixed_window_aggregate<F>(
        &self,
        agg_func: F,
        min_periods: usize,
    ) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 移動集計の計算
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // 最初のwindow-1個の要素はNA
                result_values.push(NA::NA);
            } else {
                // ウィンドウ内の値を取得
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.len() < min_periods {
                    result_values.push(NA::NA);
                } else {
                    let result = agg_func(&window_values);
                    result_values.push(NA::Value(result));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    // ------- 拡大ウィンドウ実装 -------
    
    /// 拡大ウィンドウの平均を計算
    fn expanding_window_mean(&self) -> Result<TimeSeries<T>> {
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 拡大平均の計算
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // 最小ウィンドウサイズに満たない場合はNA
                result_values.push(NA::NA);
            } else {
                // 最初から現在のインデックスまでの値を取得
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
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
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 拡大ウィンドウの合計を計算
    fn expanding_window_sum(&self) -> Result<TimeSeries<T>> {
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 拡大合計の計算
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // 最小ウィンドウサイズに満たない場合はNA
                result_values.push(NA::NA);
            } else {
                // 最初から現在のインデックスまでの値を取得
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
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
                    result_values.push(NA::Value(sum));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 拡大ウィンドウの標準偏差を計算
    fn expanding_window_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 拡大標準偏差の計算
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // 最小ウィンドウサイズに満たない場合はNA
                result_values.push(NA::NA);
            } else {
                // 最初から現在のインデックスまでの値を取得
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.len() <= ddof {
                    result_values.push(NA::NA);
                } else {
                    // 平均を計算
                    let mean: f64 = window_values.iter().sum::<f64>() / window_values.len() as f64;
                    
                    // 分散を計算
                    let variance: f64 = window_values
                        .iter()
                        .map(|v| (*v - mean).powi(2))
                        .sum::<f64>()
                        / (window_values.len() - ddof) as f64;
                    
                    // 標準偏差を計算
                    let std_dev = variance.sqrt();
                    result_values.push(NA::Value(std_dev));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 拡大ウィンドウの最小値を計算
    fn expanding_window_min(&self) -> Result<TimeSeries<T>> {
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 拡大最小値の計算
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // 最小ウィンドウサイズに満たない場合はNA
                result_values.push(NA::NA);
            } else {
                // 最初から現在のインデックスまでの値を取得
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let min = window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    result_values.push(NA::Value(min));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 拡大ウィンドウの最大値を計算
    fn expanding_window_max(&self) -> Result<TimeSeries<T>> {
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 拡大最大値の計算
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // 最小ウィンドウサイズに満たない場合はNA
                result_values.push(NA::NA);
            } else {
                // 最初から現在のインデックスまでの値を取得
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let max = window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    result_values.push(NA::Value(max));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 拡大ウィンドウに一般的な集計関数を適用
    fn expanding_window_aggregate<F>(
        &self,
        agg_func: F,
        min_periods: usize,
    ) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 拡大集計の計算
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // 最小ウィンドウサイズに満たない場合はNA
                result_values.push(NA::NA);
            } else {
                // 最初から現在のインデックスまでの値を取得
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();
                
                if window_values.len() < min_periods {
                    result_values.push(NA::NA);
                } else {
                    let result = agg_func(&window_values);
                    result_values.push(NA::Value(result));
                }
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    // ------- 指数加重ウィンドウ実装 -------
    
    /// 指数加重移動平均を計算
    fn ewm_mean(&self) -> Result<TimeSeries<T>> {
        let alpha = self.alpha.ok_or_else(|| {
            PandRSError::Consistency("指数加重ウィンドウには alpha パラメータが必要です。".to_string())
        })?;
        
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 指数加重移動平均の計算
        let values = self.time_series.values();
        
        // 初期値が無い場合
        if values.is_empty() {
            return Ok(TimeSeries::new(
                Vec::new(),
                Vec::new(),
                self.time_series.name().cloned(),
            )?);
        }
        
        // 最初の非NAのインデックスを見つける
        let first_valid_idx = values.iter().position(|v| !v.is_na());
        
        if let Some(idx) = first_valid_idx {
            // 最初の有効値までNAを追加
            for _ in 0..idx {
                result_values.push(NA::NA);
            }
            
            // 最初の有効値を取得
            let mut weighted_avg = if let NA::Value(first_val) = values[idx] {
                first_val
            } else {
                return Err(PandRSError::Consistency("不正な初期値です".to_string()));
            };
            
            // 最初の値を追加
            result_values.push(NA::Value(weighted_avg));
            
            // 残りの値に対して計算
            for i in (idx + 1)..values.len() {
                match values[i] {
                    NA::Value(val) => {
                        // 指数加重平均の更新: yt = α*xt + (1-α)*yt-1
                        weighted_avg = alpha * val + (1.0 - alpha) * weighted_avg;
                        result_values.push(NA::Value(weighted_avg));
                    }
                    NA::NA => {
                        // NAの場合は前の値を維持（NAは伝播しない）
                        result_values.push(NA::Value(weighted_avg));
                    }
                }
            }
        } else {
            // 有効な値がない場合は全てNA
            for _ in 0..values.len() {
                result_values.push(NA::NA);
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// 指数加重移動標準偏差を計算
    fn ewm_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        let alpha = self.alpha.ok_or_else(|| {
            PandRSError::Consistency("指数加重ウィンドウには alpha パラメータが必要です。".to_string())
        })?;
        
        // 自由度調整
        if ddof >= self.time_series.len() {
            return Err(PandRSError::Consistency(format!(
                "自由度調整 ddof ({}) がサンプルサイズ ({}) 以上です",
                ddof, self.time_series.len()
            )));
        }
        
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // 指数加重移動標準偏差の計算
        let values = self.time_series.values();
        
        // 初期値が無い場合
        if values.is_empty() {
            return Ok(TimeSeries::new(
                Vec::new(),
                Vec::new(),
                self.time_series.name().cloned(),
            )?);
        }
        
        // 最初の非NAのインデックスを見つける
        let first_valid_idx = values.iter().position(|v| !v.is_na());
        
        if let Some(idx) = first_valid_idx {
            // 最初の有効値までNAを追加
            for _ in 0..idx {
                result_values.push(NA::NA);
            }
            
            // 最初の有効値を取得
            let first_val = if let NA::Value(val) = values[idx] {
                val
            } else {
                return Err(PandRSError::Consistency("不正な初期値です".to_string()));
            };
            
            // 初期値の設定
            let mut weighted_avg = first_val;
            let mut weighted_sq_avg = first_val * first_val;
            
            // 最初の値を追加（標準偏差は0）
            result_values.push(NA::Value(0.0));
            
            // 残りの値に対して計算
            for i in (idx + 1)..values.len() {
                match values[i] {
                    NA::Value(val) => {
                        // 指数加重平均の更新
                        weighted_avg = alpha * val + (1.0 - alpha) * weighted_avg;
                        
                        // 指数加重二乗平均の更新
                        weighted_sq_avg = alpha * val * val + (1.0 - alpha) * weighted_sq_avg;
                        
                        // 分散 = E[X^2] - (E[X])^2
                        let variance = weighted_sq_avg - weighted_avg * weighted_avg;
                        
                        // 分散が負になることを防ぐ（数値誤差対策）
                        let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                        
                        result_values.push(NA::Value(std_dev));
                    }
                    NA::NA => {
                        // NAの場合は前の値を維持
                        result_values.push(result_values.last().unwrap().clone());
                    }
                }
            }
        } else {
            // 有効な値がない場合は全てNA
            for _ in 0..values.len() {
                result_values.push(NA::NA);
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
}

// TimeSeries構造体に新しいウィンドウ操作メソッドを追加
impl<T: Temporal> TimeSeries<T> {
    /// 固定長ウィンドウ操作を作成
    pub fn rolling(&self, window_size: usize) -> Result<Window<T>> {
        Window::new(self, WindowType::Fixed, window_size)
    }
    
    /// 拡大ウィンドウ操作を作成
    pub fn expanding(&self, min_periods: usize) -> Result<Window<T>> {
        Window::new(self, WindowType::Expanding, min_periods)
    }
    
    /// 指数加重ウィンドウ操作を作成
    pub fn ewm(&self, span: Option<usize>, alpha: Option<f64>, adjust: bool) -> Result<Window<T>> {
        // spanとalphaの両方が指定された場合はエラー
        if span.is_some() && alpha.is_some() {
            return Err(PandRSError::Consistency(
                "span と alpha は同時に指定できません。いずれか一方を指定してください。".to_string(),
            ));
        }
        
        // alphaを計算またはそのまま使用
        let alpha_value = if let Some(alpha_val) = alpha {
            alpha_val
        } else if let Some(span_val) = span {
            if span_val < 1 {
                return Err(PandRSError::Consistency(
                    "span は1以上である必要があります。".to_string(),
                ));
            }
            // alpha = 2/(span+1)
            2.0 / (span_val as f64 + 1.0)
        } else {
            // デフォルトはspan=5に相当するalpha
            2.0 / (5.0 + 1.0)
        };
        
        // ウィンドウ作成（window_sizeは1に設定、実際には使用されない）
        let mut window = Window::new(self, WindowType::ExponentiallyWeighted, 1)?;
        window = window.with_alpha(alpha_value)?;
        
        // adjust引数は現在のバージョンでは使用していないが、将来的に実装する可能性あり
        
        Ok(window)
    }
}