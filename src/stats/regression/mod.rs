// 回帰分析モジュール

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use crate::stats::LinearRegressionResult;

/// 線形回帰分析を実行する内部実装
pub(crate) fn linear_regression_impl(
    df: &DataFrame,
    y_column: &str,
    x_columns: &[&str],
) -> Result<LinearRegressionResult> {
    // 対象列の存在確認
    if !df.contains_column(y_column) {
        return Err(Error::ColumnNotFound(y_column.to_string()));
    }
    
    for &x_col in x_columns {
        if !df.contains_column(x_col) {
            return Err(Error::ColumnNotFound(x_col.to_string()));
        }
    }
    
    if x_columns.is_empty() {
        return Err(Error::InvalidOperation("回帰分析には少なくとも1つの説明変数が必要です".into()));
    }
    
    // 目的変数の取得
    let y_series = df.get_column(y_column).ok_or_else(|| Error::ColumnNotFound(y_column.to_string()))?;
    
    // 文字列のSeriesを数値に変換
    let y_values: Vec<f64> = y_series.values().iter()
        .map(|s| s.parse::<f64>().unwrap_or(0.0))
        .collect();
    
    // 説明変数の取得（複数の列）
    let mut x_matrix: Vec<Vec<f64>> = Vec::with_capacity(x_columns.len() + 1);
    
    // 切片用の列（全て1.0）
    let n = y_values.len();
    let intercept_col = vec![1.0; n];
    x_matrix.push(intercept_col);
    
    // 各説明変数の列を追加
    for &x_col in x_columns {
        let x_series = df.get_column(x_col).ok_or_else(|| Error::ColumnNotFound(x_col.to_string()))?;
        
        // 文字列のSeriesを数値に変換
        let x_values: Vec<f64> = x_series.values().iter()
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();
        
        if x_values.len() != n {
            return Err(Error::DimensionMismatch(
                format!("回帰分析: 列の長さが一致しません: y={}, {}={}", n, x_col, x_values.len())
            ));
        }
        
        x_matrix.push(x_values);
    }
    
    // 行列計算による最小二乗法の実装
    // X^T * X の計算
    let xt_x = matrix_multiply_transpose(&x_matrix, &x_matrix);
    
    // (X^T * X)^(-1) の計算
    let xt_x_inv = matrix_inverse(&xt_x)?;
    
    // X^T * y の計算
    let xt_y = vec_multiply_transpose(&x_matrix, &y_values);
    
    // β = (X^T * X)^(-1) * X^T * y の計算
    let mut coefficients = vec![0.0; x_matrix.len()];
    
    for i in 0..coefficients.len() {
        let mut sum = 0.0;
        for j in 0..xt_y.len() {
            sum += xt_x_inv[i][j] * xt_y[j];
        }
        coefficients[i] = sum;
    }
    
    // 切片と係数の分離
    let intercept = coefficients[0];
    let beta_coefs = coefficients[1..].to_vec();
    
    // 予測値の計算
    let mut fitted_values = vec![0.0; n];
    for i in 0..n {
        fitted_values[i] = intercept;
        for j in 0..beta_coefs.len() {
            fitted_values[i] += beta_coefs[j] * x_matrix[j + 1][i];
        }
    }
    
    // 残差の計算
    let residuals: Vec<f64> = y_values.iter()
        .zip(fitted_values.iter())
        .map(|(&y, &y_hat)| y - y_hat)
        .collect();
    
    // 決定係数（R²）の計算
    let y_mean = y_values.iter().sum::<f64>() / n as f64;
    
    let ss_total = y_values.iter()
        .map(|&y| (y - y_mean).powi(2))
        .sum::<f64>();
    
    let ss_residual = residuals.iter()
        .map(|&r| r.powi(2))
        .sum::<f64>();
    
    let r_squared = 1.0 - ss_residual / ss_total;
    
    // 調整済み決定係数
    let p = x_columns.len();
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) as f64 / (n - p - 1) as f64;
    
    // p値の計算（簡易版）
    // 実際の実装ではt分布の確率密度関数を使用するべき
    let mut p_values = vec![0.0; p + 1];
    
    // 標準誤差の計算
    let std_errors = calculate_std_errors(&xt_x_inv, ss_residual, n, p)?;
    
    // 各係数のt値とp値
    for i in 0..p_values.len() {
        let t_value = coefficients[i] / std_errors[i];
        // 両側t検定のp値（簡易計算）
        p_values[i] = 2.0 * (1.0 - normal_cdf(t_value.abs()));
    }
    
    Ok(LinearRegressionResult {
        intercept,
        coefficients: beta_coefs,
        r_squared,
        adj_r_squared,
        p_values,
        fitted_values,
        residuals,
    })
}

/// 行列の転置積（A^T * B）を計算
fn matrix_multiply_transpose(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    
    let mut result = vec![vec![0.0; m]; n];
    
    // 各要素を計算
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..a[i].len() {
                sum += a[i][k] * b[j][k];
            }
            result[i][j] = sum;
        }
    }
    
    result
}

/// ベクトルの転置積（A^T * y）を計算
fn vec_multiply_transpose(a: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0; n];
    
    // 各要素を計算
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..y.len() {
            sum += a[i][k] * y[k];
        }
        result[i] = sum;
    }
    
    result
}

/// 正規分布のCDF（累積分布関数）を計算
fn normal_cdf(z: f64) -> f64 {
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

/// 行列の逆行列を計算（ガウス・ジョルダン法）
fn matrix_inverse(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    
    if n == 0 {
        return Err(Error::InvalidOperation("行列が空です".into()));
    }
    
    for row in matrix {
        if row.len() != n {
            return Err(Error::DimensionMismatch("正方行列である必要があります".into()));
        }
    }
    
    // 拡張行列を作成 [A|I]
    let mut augmented = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(2 * n);
        row.extend_from_slice(&matrix[i]);
        
        // 単位行列部分
        for j in 0..n {
            row.push(if i == j { 1.0 } else { 0.0 });
        }
        
        augmented.push(row);
    }
    
    // ガウス・ジョルダン消去法
    for i in 0..n {
        // ピボット選択
        let mut max_row = i;
        let mut max_val = augmented[i][i].abs();
        
        for j in i + 1..n {
            let abs_val = augmented[j][i].abs();
            if abs_val > max_val {
                max_row = j;
                max_val = abs_val;
            }
        }
        
        if max_val < 1e-10 {
            return Err(Error::ComputationError("行列が特異です（逆行列が存在しません）".into()));
        }
        
        // 行の交換
        if max_row != i {
            augmented.swap(i, max_row);
        }
        
        // ピボット要素を1にする
        let pivot = augmented[i][i];
        for j in 0..2 * n {
            augmented[i][j] /= pivot;
        }
        
        // 他の行の消去
        for j in 0..n {
            if j != i {
                let factor = augmented[j][i];
                for k in 0..2 * n {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }
    
    // 結果の抽出（右半分が逆行列）
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = augmented[i][j + n];
        }
    }
    
    Ok(inverse)
}

/// 回帰係数の標準誤差を計算
fn calculate_std_errors(
    xt_x_inv: &[Vec<f64>],
    ss_residual: f64,
    n: usize,
    p: usize,
) -> Result<Vec<f64>> {
    // 残差の標準誤差（RMSE）
    let df = n - p - 1; // 自由度
    if df <= 0 {
        return Err(Error::InsufficientData("自由度が0以下になっています。十分なデータ数が必要です。".into()));
    }
    
    let mse = ss_residual / df as f64;
    
    // 係数の標準誤差
    let mut std_errors = Vec::with_capacity(xt_x_inv.len());
    for i in 0..xt_x_inv.len() {
        std_errors.push((mse * xt_x_inv[i][i]).sqrt());
    }
    
    Ok(std_errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;
    
    #[test]
    fn test_simple_regression() {
        // 簡単な回帰分析のテスト
        let mut df = DataFrame::new();
        
        let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string())).unwrap();
        let y = Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("y".to_string())).unwrap();
        
        df.add_column("x".to_string(), x).unwrap();
        df.add_column("y".to_string(), y).unwrap();
        
        let result = linear_regression_impl(&df, "y", &["x"]).unwrap();
        
        // y = 2x なので、切片は0, 係数は2になるはず
        assert!((result.intercept - 0.0).abs() < 1e-10);
        assert!((result.coefficients[0] - 2.0).abs() < 1e-10);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_multiple_regression() {
        let mut df = DataFrame::new();
        
        let x1 = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x1".to_string())).unwrap();
        let x2 = Series::new(vec![5.0, 4.0, 3.0, 2.0, 1.0], Some("x2".to_string())).unwrap();
        // y = 2*x1 + 3*x2 + 1
        let y = Series::new(vec![
            2.0*1.0 + 3.0*5.0 + 1.0,
            2.0*2.0 + 3.0*4.0 + 1.0,
            2.0*3.0 + 3.0*3.0 + 1.0,
            2.0*4.0 + 3.0*2.0 + 1.0,
            2.0*5.0 + 3.0*1.0 + 1.0,
        ], Some("y".to_string())).unwrap();
        
        df.add_column("x1".to_string(), x1).unwrap();
        df.add_column("x2".to_string(), x2).unwrap();
        df.add_column("y".to_string(), y).unwrap();
        
        let result = linear_regression_impl(&df, "y", &["x1", "x2"]).unwrap();
        
        // y = 1 + 2*x1 + 3*x2 なので、期待値は:
        assert!((result.intercept - 1.0).abs() < 1e-10);
        assert!((result.coefficients[0] - 2.0).abs() < 1e-10);
        assert!((result.coefficients[1] - 3.0).abs() < 1e-10);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
    }
}