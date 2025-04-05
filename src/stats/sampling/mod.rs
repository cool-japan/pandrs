// サンプリングと乱数生成モジュール

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// データフレームからサンプルを抽出する内部実装
pub(crate) fn sample_impl(
    df: &DataFrame,
    fraction: f64,
    replace: bool,
) -> Result<DataFrame> {
    if fraction <= 0.0 {
        return Err(Error::InvalidValue("サンプル率は正の値である必要があります".into()));
    }
    
    let n_rows = df.len();
    if n_rows == 0 {
        return Ok(DataFrame::new());
    }
    
    let sample_size = (n_rows as f64 * fraction).ceil() as usize;
    if !replace && sample_size > n_rows {
        return Err(Error::InvalidOperation(
            "非復元抽出の場合、サンプルサイズは元のデータサイズ以下である必要があります".into()
        ));
    }
    
    // シード付きの乱数生成器を使用（再現性のため）
    let mut rng = StdRng::from_entropy();
    
    // インデックスの生成
    let indices = if replace {
        // 復元抽出
        (0..sample_size)
            .map(|_| rng.gen_range(0..n_rows))
            .collect::<Vec<_>>()
    } else {
        // 非復元抽出
        let mut idx: Vec<usize> = (0..n_rows).collect();
        idx.shuffle(&mut rng);
        idx[0..sample_size].to_vec()
    };
    
    // サンプルデータフレームの作成
    df.take(&indices)
}

/// ブートストラップサンプルを生成する内部実装
pub(crate) fn bootstrap_impl(
    data: &[f64],
    n_samples: usize,
) -> Result<Vec<Vec<f64>>> {
    if data.is_empty() {
        return Err(Error::EmptyData("ブートストラップにはデータが必要です".into()));
    }
    
    if n_samples == 0 {
        return Err(Error::InvalidValue("サンプル数は正の値である必要があります".into()));
    }
    
    let n = data.len();
    let mut rng = StdRng::from_entropy();
    let mut result = Vec::with_capacity(n_samples);
    
    for _ in 0..n_samples {
        // 復元抽出によるサンプリング
        let sample: Vec<f64> = (0..n)
            .map(|_| data[rng.gen_range(0..n)])
            .collect();
        
        result.push(sample);
    }
    
    Ok(result)
}

/// 層化サンプリングを実行
///
/// 指定した層（グループ）ごとに指定した比率でサンプリングします。
/// 
/// # 引数
/// * `df` - 入力データフレーム
/// * `strata_column` - 層を指定する列名
/// * `fraction` - 各層からのサンプリング率
/// * `replace` - 復元抽出するかどうか
pub(crate) fn stratified_sample_impl(
    df: &DataFrame,
    strata_column: &str,
    fraction: f64,
    replace: bool,
) -> Result<DataFrame> {
    if !df.has_column(strata_column) {
        return Err(Error::ColumnNotFound(strata_column.to_string()));
    }
    
    if fraction <= 0.0 {
        return Err(Error::InvalidValue("サンプル率は正の値である必要があります".into()));
    }
    
    // 層ごとのグループ化
    let groups = df.group_by(strata_column)?;
    let keys = groups.keys();
    
    // 各層からサンプルを抽出し結合
    let mut result = DataFrame::new();
    let mut first = true;
    
    for key in keys {
        let group_df = groups.get(key)?;
        let sample = sample_impl(&group_df, fraction, replace)?;
        
        if first {
            result = sample;
            first = false;
        } else {
            result = result.vstack(&sample)?;
        }
    }
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;
    
    #[test]
    fn test_simple_sample() {
        let mut df = DataFrame::new();
        let data = Series::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        df.add_column("data", data).unwrap();
        
        // 50%サンプリング（非復元）
        let sample = sample_impl(&df, 0.5, false).unwrap();
        assert_eq!(sample.len(), 5);
        
        // 30%サンプリング（復元）
        let sample = sample_impl(&df, 0.3, true).unwrap();
        assert_eq!(sample.len(), 3);
        
        // 200%サンプリング（復元）
        let sample = sample_impl(&df, 2.0, true).unwrap();
        assert_eq!(sample.len(), 20);
        
        // 200%サンプリング（非復元）- エラーになるはず
        let result = sample_impl(&df, 2.0, false);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_bootstrap() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // 10サンプルのブートストラップ
        let bootstrap_samples = bootstrap_impl(&data, 10).unwrap();
        assert_eq!(bootstrap_samples.len(), 10);
        
        // 各サンプルは元データと同じ長さ
        for sample in &bootstrap_samples {
            assert_eq!(sample.len(), data.len());
        }
        
        // サンプルは元データから復元抽出されたもの
        for sample in &bootstrap_samples {
            for value in sample {
                assert!(data.contains(value));
            }
        }
    }
}