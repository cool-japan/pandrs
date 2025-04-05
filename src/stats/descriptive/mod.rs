// 記述統計モジュール

use crate::error::{Result, Error};
use crate::stats::DescriptiveStats;

/// 記述統計量を計算する内部実装
pub(crate) fn describe_impl(data: &[f64]) -> Result<DescriptiveStats> {
    if data.is_empty() {
        return Err(Error::EmptyData("記述統計量の計算には少なくとも1つのデータが必要です".into()));
    }

    let count = data.len();
    
    // 平均値の計算
    let mean = data.iter().sum::<f64>() / count as f64;
    
    // 標準偏差の計算（不偏推定量）
    let variance = if count > 1 {
        let sum_squared_diff = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>();
        sum_squared_diff / (count - 1) as f64
    } else {
        0.0
    };
    let std = variance.sqrt();
    
    // データをソートして分位数を計算
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let min = sorted[0];
    let max = sorted[count - 1];
    
    // 分位数の計算
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };
    
    // 第1四分位点 (25%)
    let q1 = percentile(&sorted, 0.25);
    
    // 第3四分位点 (75%)
    let q3 = percentile(&sorted, 0.75);
    
    Ok(DescriptiveStats {
        count,
        mean,
        std,
        min,
        q1,
        median,
        q3,
        max,
    })
}

/// パーセンタイルを計算
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    let n = sorted_data.len();
    let idx = p * (n - 1) as f64;
    let idx_floor = idx.floor() as usize;
    let idx_ceil = idx.ceil() as usize;
    
    if idx_floor == idx_ceil {
        return sorted_data[idx_floor];
    }
    
    let weight_ceil = idx - idx_floor as f64;
    let weight_floor = 1.0 - weight_ceil;
    
    sorted_data[idx_floor] * weight_floor + sorted_data[idx_ceil] * weight_ceil
}

/// 共分散を計算する内部実装
pub(crate) fn covariance_impl(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            format!("共分散計算のデータ長が一致しません: x={}, y={}", x.len(), y.len())
        ));
    }
    
    if x.is_empty() {
        return Err(Error::EmptyData("共分散計算にはデータが必要です".into()));
    }
    
    let n = x.len();
    
    if n <= 1 {
        return Err(Error::InsufficientData("共分散計算には少なくとも2つのデータポイントが必要です".into()));
    }
    
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    
    let cov = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>() / (n - 1) as f64;
    
    Ok(cov)
}

/// 相関係数を計算する内部実装
pub(crate) fn correlation_impl(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            format!("相関係数計算のデータ長が一致しません: x={}, y={}", x.len(), y.len())
        ));
    }
    
    if x.is_empty() {
        return Err(Error::EmptyData("相関係数計算にはデータが必要です".into()));
    }
    
    let n = x.len();
    
    if n <= 1 {
        return Err(Error::InsufficientData("相関係数計算には少なくとも2つのデータポイントが必要です".into()));
    }
    
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    
    // 分子: Σ(xi - x̄)(yi - ȳ)
    let numerator = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>();
    
    // 分母: √[Σ(xi - x̄)² * Σ(yi - ȳ)²]
    let sum_squared_diff_x = x.iter()
        .map(|&xi| (xi - mean_x).powi(2))
        .sum::<f64>();
    
    let sum_squared_diff_y = y.iter()
        .map(|&yi| (yi - mean_y).powi(2))
        .sum::<f64>();
    
    let denominator = (sum_squared_diff_x * sum_squared_diff_y).sqrt();
    
    if denominator.abs() < std::f64::EPSILON {
        return Err(Error::ComputationError("相関係数計算: 分散がゼロです".into()));
    }
    
    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_describe_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = describe_impl(&data).unwrap();
        
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.std - 1.5811388300841898).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        // パーセンタイル計算のアルゴリズム修正に合わせて期待値を更新
        assert!((stats.q1 - 2.0).abs() < 1e-10);
        assert!((stats.q3 - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_describe_empty() {
        let data: Vec<f64> = vec![];
        let result = describe_impl(&data);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cov = covariance_impl(&x, &y).unwrap();
        assert!((cov - 2.5).abs() < 1e-10);
        
        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let cov_neg = covariance_impl(&x, &y_neg).unwrap();
        assert!((cov_neg + 2.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = correlation_impl(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
        
        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = correlation_impl(&x, &y_neg).unwrap();
        assert!((corr_neg + 1.0).abs() < 1e-10);
        
        let y_uncorr = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let result = correlation_impl(&x, &y_uncorr);
        assert!(result.is_err());
    }
}