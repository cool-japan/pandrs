// 推測統計・仮説検定モジュール

use crate::error::{Result, Error};
use crate::stats::{TTestResult, AnovaResult, MannWhitneyResult, ChiSquareResult};
use std::f64::consts::PI;
use std::collections::HashMap;

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

/// F分布の累積分布関数（CDF）
/// F分布のp値を計算する関数（近似値）
fn f_distribution_cdf(f: f64, df1: usize, df2: usize) -> f64 {
    // F分布の近似計算（自由度が大きい場合は、より精度の高い実装が必要）
    // 実際のライブラリ実装では特殊関数ライブラリを使うべき
    
    // F分布とベータ分布の関係を利用した近似
    let df1_f64 = df1 as f64;
    let df2_f64 = df2 as f64;
    let x = df1_f64 * f / (df1_f64 * f + df2_f64);
    
    // 不完全ベータ関数の近似計算
    // ベータ関数の不完全版の近似計算
    let a = df1_f64 / 2.0;
    let b = df2_f64 / 2.0;
    
    // 近似計算（簡略版）
    let beta_approx = if x > 0.5 {
        // 0.5より大きい場合の近似
        1.0 - (1.0 - x).powf(b) * (1.0 + (1.0 - x) * a / b + 
                                (1.0 - x).powi(2) * a * (a + 1.0) / (b * (b + 1.0)) / 2.0)
    } else {
        // 0.5以下の場合の近似
        x.powf(a) * (1.0 + x * b / a + 
                    x.powi(2) * b * (b + 1.0) / (a * (a + 1.0)) / 2.0)
    };
    
    beta_approx.min(1.0).max(0.0)
}

/// 一元配置分散分析（ANOVA）の実装
pub(crate) fn anova_impl(
    groups: &HashMap<&str, &[f64]>, 
    alpha: f64
) -> Result<AnovaResult> {
    // グループ数と各グループのサンプルサイズをチェック
    if groups.is_empty() {
        return Err(Error::EmptyData("分散分析には少なくとも1つのグループが必要です".into()));
    }
    
    if groups.len() < 2 {
        return Err(Error::InsufficientData("分散分析には少なくとも2つのグループが必要です".into()));
    }
    
    // 総データ数とグループごとの平均、全体の平均を計算
    let mut total_n = 0;
    let mut global_sum = 0.0;
    
    for (_, values) in groups.iter() {
        if values.is_empty() {
            return Err(Error::EmptyData("空のグループがあります".into()));
        }
        
        total_n += values.len();
        global_sum += values.iter().sum::<f64>();
    }
    
    let global_mean = global_sum / total_n as f64;
    
    // グループ間平方和（SSB）、グループ内平方和（SSW）、総平方和（SST）の計算
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    let mut ss_total = 0.0;
    
    for (_, values) in groups.iter() {
        let group_n = values.len();
        let group_mean = values.iter().sum::<f64>() / group_n as f64;
        
        // グループ間平方和の計算
        ss_between += group_n as f64 * (group_mean - global_mean).powi(2);
        
        // グループ内平方和の計算
        for &value in *values {
            // グループ内変動
            ss_within += (value - group_mean).powi(2);
            
            // 総変動（確認用）
            ss_total += (value - global_mean).powi(2);
        }
    }
    
    // 自由度の計算
    let df_between = groups.len() - 1;
    let df_within = total_n - groups.len();
    let df_total = total_n - 1;
    
    // 平均平方（MS）の計算
    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;
    
    // F値の計算
    let f_statistic = ms_between / ms_within;
    
    // p値の計算（F分布を使用）
    let p_value = 1.0 - f_distribution_cdf(f_statistic, df_between, df_within);
    
    // 結果の返却
    Ok(AnovaResult {
        f_statistic,
        p_value,
        ss_between,
        ss_within,
        ss_total,
        df_between,
        df_within,
        df_total,
        ms_between,
        ms_within,
        significant: p_value < alpha,
    })
}

/// Mann-Whitney U検定（ノンパラメトリック検定）の実装
pub(crate) fn mann_whitney_u_impl(
    sample1: &[f64],
    sample2: &[f64],
    alpha: f64
) -> Result<MannWhitneyResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(Error::EmptyData("Mann-Whitney U検定にはデータが必要です".into()));
    }
    
    let n1 = sample1.len();
    let n2 = sample2.len();
    
    // 両方のサンプルを結合してランク付け
    let mut combined: Vec<(f64, usize, usize)> = Vec::with_capacity(n1 + n2);
    
    // グループ1のデータを追加
    for (i, &val) in sample1.iter().enumerate() {
        combined.push((val, 0, i)); // グループ0、インデックスi
    }
    
    // グループ2のデータを追加
    for (i, &val) in sample2.iter().enumerate() {
        combined.push((val, 1, i)); // グループ1、インデックスi
    }
    
    // 値でソート
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // ランク付け
    let mut ranks = vec![0.0; n1 + n2];
    let mut i = 0;
    
    while i < n1 + n2 {
        let mut j = i;
        // 同じ値のデータを見つける
        while j < n1 + n2 - 1 && (combined[j].0 - combined[j + 1].0).abs() < f64::EPSILON {
            j += 1;
        }
        
        // 同点の場合、平均ランクを割り当て
        if j > i {
            let rank_avg = (i + 1 + j + 1) as f64 / 2.0;
            for k in i..=j {
                let (_, group, idx) = combined[k];
                if group == 0 {
                    ranks[idx] = rank_avg;
                } else {
                    ranks[idx + n1] = rank_avg;
                }
            }
        } else {
            let (_, group, idx) = combined[i];
            if group == 0 {
                ranks[idx] = (i + 1) as f64;
            } else {
                ranks[idx + n1] = (i + 1) as f64;
            }
        }
        
        i = j + 1;
    }
    
    // グループ1の順位和を計算
    let r1: f64 = ranks.iter().take(n1).sum();
    
    // U統計量の計算
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    
    // 小さい方のU値を使用
    let u_statistic = u1.min(u2);
    
    // 平均と標準偏差の計算
    let mean_u = (n1 * n2) as f64 / 2.0;
    let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    
    // 正規近似によるp値の計算
    let z = (u_statistic - mean_u) / std_u;
    let p_value = 2.0 * normal_cdf(-z.abs()); // 両側検定
    
    Ok(MannWhitneyResult {
        u_statistic,
        p_value,
        significant: p_value < alpha,
    })
}

/// カイ二乗検定の実装
pub(crate) fn chi_square_test_impl(
    observed: &[Vec<f64>],
    alpha: f64
) -> Result<ChiSquareResult> {
    // 観測データの検証
    if observed.is_empty() {
        return Err(Error::EmptyData("カイ二乗検定には観測データが必要です".into()));
    }
    
    let rows = observed.len();
    if rows < 2 {
        return Err(Error::InsufficientData("カイ二乗検定には少なくとも2行のデータが必要です".into()));
    }
    
    let cols = observed[0].len();
    if cols < 2 {
        return Err(Error::InsufficientData("カイ二乗検定には少なくとも2列のデータが必要です".into()));
    }
    
    // すべての行が同じ列数を持つことを確認
    for row in observed.iter() {
        if row.len() != cols {
            return Err(Error::InvalidInput("すべての行は同じ列数を持つ必要があります".into()));
        }
    }
    
    // 行と列の合計を計算
    let mut row_sums = vec![0.0; rows];
    let mut col_sums = vec![0.0; cols];
    let mut total_sum = 0.0;
    
    for i in 0..rows {
        for j in 0..cols {
            let value = observed[i][j];
            if value < 0.0 {
                return Err(Error::InvalidInput("観測値は負であってはなりません".into()));
            }
            row_sums[i] += value;
            col_sums[j] += value;
            total_sum += value;
        }
    }
    
    if total_sum < 1.0 {
        return Err(Error::InvalidInput("観測データの合計が0です".into()));
    }
    
    // 期待度数の計算
    let mut expected = vec![vec![0.0; cols]; rows];
    let mut chi2_statistic = 0.0;
    
    for i in 0..rows {
        for j in 0..cols {
            // 期待度数 = (行合計 * 列合計) / 総合計
            expected[i][j] = row_sums[i] * col_sums[j] / total_sum;
            
            // 期待度数が5未満の場合は警告（Yatesの補正が必要かもしれない）
            if expected[i][j] < 5.0 {
                // ここでは警告のみ表示（実際のライブラリではログ出力など）
                // println!("警告: 期待度数が5未満のセルがあります。結果の解釈には注意が必要です。");
            }
            
            // カイ二乗統計量の計算
            let diff = observed[i][j] - expected[i][j];
            chi2_statistic += diff * diff / expected[i][j];
        }
    }
    
    // 自由度の計算
    let df = (rows - 1) * (cols - 1);
    
    // p値の計算
    let p_value = chi2_to_pvalue(chi2_statistic, df);
    
    Ok(ChiSquareResult {
        chi2_statistic,
        p_value,
        df,
        significant: p_value < alpha,
        expected_freq: expected,
    })
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
    
    #[test]
    fn test_anova_basic() {
        let mut groups = HashMap::new();
        let a_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b_values = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c_values = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        
        groups.insert("A", a_values.as_slice());
        groups.insert("B", b_values.as_slice());
        groups.insert("C", c_values.as_slice());
        
        let result = anova_impl(&groups, 0.05).unwrap();
        
        // 各グループの平均は順に3, 4, 5で、差は明確だが分散も大きいため
        // F値は正の値で、前後のグループとの差は1.0
        assert!(result.f_statistic > 0.0);
        // 15個のデータ、3グループなので自由度は2, 12
        assert_eq!(result.df_between, 2);
        assert_eq!(result.df_within, 12);
        assert_eq!(result.df_total, 14);
    }
    
    #[test]
    fn test_anova_significant_difference() {
        let mut groups = HashMap::new();
        let a_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b_values = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let c_values = vec![21.0, 22.0, 23.0, 24.0, 25.0];
        
        groups.insert("A", a_values.as_slice());
        groups.insert("B", b_values.as_slice());
        groups.insert("C", c_values.as_slice());
        
        let result = anova_impl(&groups, 0.05).unwrap();
        
        // 大きな差がある場合はF値も大きい
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 0.05);
        assert!(result.significant);
    }
    
    #[test]
    fn test_mann_whitney_u() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        
        let result = mann_whitney_u_impl(&sample1, &sample2, 0.05).unwrap();
        
        // 完全に分離したサンプルなので有意差があるはず
        assert!(result.u_statistic == 0.0); // 最小のU値
        assert!(result.p_value < 0.05);
        assert!(result.significant);
    }
    
    #[test]
    fn test_chi_square() {
        // 2x2のカイ二乗検定（独立性の検定）
        let observed = vec![
            vec![10.0, 10.0],
            vec![10.0, 20.0]
        ];
        
        let result = chi_square_test_impl(&observed, 0.05).unwrap();
        
        assert!(result.chi2_statistic > 0.0);
        assert_eq!(result.df, 1); // (2-1) * (2-1) = 1
        
        // 期待度数の確認
        assert_eq!(result.expected_freq.len(), 2);
        assert_eq!(result.expected_freq[0].len(), 2);
    }
}