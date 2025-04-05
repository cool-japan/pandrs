// 推測統計・仮説検定モジュール

use crate::error::{Result, Error};
use crate::stats::TTestResult;
use std::f64::consts::PI;

/// 標準正規分布のCDF（累積分布関数）を計算
fn normal_cdf(z: f64) -> f64 {
    // 誤差関数の近似計算（純Rustで実装）
    // 標準正規分布のCDFの近似計算（Abramowitz and Stegun近似式）
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / (2.0_f64).sqrt();
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    
    0.5 * (1.0 + sign * y)
}

/// t分布のCDF（累積分布関数）を計算
fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    // 標準正規分布の近似を利用（自由度が大きい場合）
    if df > 30 {
        return normal_cdf(t);
    }
    
    // ここでは簡略化した近似を使用
    // 実際の実装では、より高精度な計算が必要
    let df_f64 = df as f64;
    let x = df_f64 / (df_f64 + t * t);
    let a = 0.5 * df_f64;
    let b = 0.5;
    
    // ベータ関数の不完全版の近似計算（正確なt分布CDFの計算には特殊関数が必要）
    // この部分は実際には数値計算ライブラリを使用するべき
    let beta_approx = if t > 0.0 {
        1.0 - 0.5 * x.powf(a)
    } else {
        0.5 * x.powf(a)
    };
    
    beta_approx
}

/// 2標本のt検定を実行する内部実装
pub(crate) fn ttest_impl(
    sample1: &[f64],
    sample2: &[f64],
    alpha: f64,
    equal_var: bool,
) -> Result<TTestResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(Error::EmptyData("t検定にはデータが必要です".into()));
    }
    
    let n1 = sample1.len();
    let n2 = sample2.len();
    
    if n1 < 2 || n2 < 2 {
        return Err(Error::InsufficientData("t検定には各グループに少なくとも2つのデータポイントが必要です".into()));
    }
    
    // 平均値の計算
    let mean1 = sample1.iter().sum::<f64>() / n1 as f64;
    let mean2 = sample2.iter().sum::<f64>() / n2 as f64;
    
    // 分散の計算
    let var1 = sample1.iter()
        .map(|&x| (x - mean1).powi(2))
        .sum::<f64>() / (n1 - 1) as f64;
    
    let var2 = sample2.iter()
        .map(|&x| (x - mean2).powi(2))
        .sum::<f64>() / (n2 - 1) as f64;
    
    let (t_stat, df) = if equal_var {
        // 等分散を仮定した場合のt統計量
        let pooled_var = ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2) / 
                          (n1 + n2 - 2) as f64;
        let std_err = (pooled_var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
        let t_value = (mean1 - mean2) / std_err;
        (t_value, n1 + n2 - 2)
    } else {
        // Welchのt検定（等分散を仮定しない）
        let std_err = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
        let t_value = (mean1 - mean2) / std_err;
        
        // Welch-Satterthwaiteの近似による自由度
        let df_num = (var1 / n1 as f64 + var2 / n2 as f64).powi(2);
        let df_denom = (var1 / n1 as f64).powi(2) / (n1 - 1) as f64 +
                       (var2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df_welch = df_num / df_denom;
        (t_value, df_welch.floor() as usize)
    };
    
    // 両側検定のp値計算
    let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));
    
    Ok(TTestResult {
        statistic: t_stat,
        pvalue: p_value,
        significant: p_value < alpha,
        df,
    })
}

/// カイ二乗値からp値を計算
fn chi2_to_pvalue(chi2: f64, df: usize) -> f64 {
    // 簡易的な実装（実際はより高精度な計算が必要）
    // 現実の実装では特殊関数ライブラリを使用するべき
    let k = df as f64 / 2.0;
    let x = chi2 / 2.0;
    
    // ガンマ関数の近似計算
    let gamma_k = if df % 2 == 0 {
        1.0 // k is integer
    } else {
        (PI * 2.0).sqrt() // k + 0.5 is integer
    };
    
    // 不完全ガンマ関数の下側確率の近似計算
    let p = if chi2 > df as f64 + 2.0 {
        1.0 - gamma_k * (1.0 - x.exp() * (1.0 + x + 0.5 * x.powi(2)))
    } else {
        gamma_k * x.exp() * x.powf(k - 1.0)
    };
    
    1.0 - p.min(1.0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ttest_equal_means() {
        let sample1 = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        
        let result = ttest_impl(&sample1, &sample2, 0.05, true).unwrap();
        
        // 平均の差は1.0、しかし分散が大きいため有意でないはず
        assert!((result.statistic + 1.0).abs() < 1.0); // t値は負になるはず
        assert!(result.pvalue > 0.05); // 有意でないはず
        assert!(!result.significant);
    }
    
    #[test]
    fn test_ttest_different_means() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        
        let result = ttest_impl(&sample1, &sample2, 0.05, true).unwrap();
        
        // 平均の差は大きく、有意差があるはず
        assert!(result.statistic < -5.0); // t値は大きな負の値
        assert!(result.pvalue < 0.05); // 有意差あり
        assert!(result.significant);
    }
    
    #[test]
    fn test_ttest_welch() {
        // 分散が異なるデータ
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![11.0, 13.0, 15.0, 17.0, 19.0];
        
        let result_equal_var = ttest_impl(&sample1, &sample2, 0.05, true).unwrap();
        let result_welch = ttest_impl(&sample1, &sample2, 0.05, false).unwrap();
        
        // どちらも有意だが、自由度や正確な統計量は異なるはず
        assert!(result_equal_var.significant);
        assert!(result_welch.significant);
        assert!(result_equal_var.df != result_welch.df);
    }
    
    #[test]
    fn test_ttest_empty() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2: Vec<f64> = vec![];
        
        let result = ttest_impl(&sample1, &sample2, 0.05, true);
        assert!(result.is_err());
    }
}