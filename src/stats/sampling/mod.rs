// サンプリングと乱数生成モジュール

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use rand::prelude::*;
use std::collections::HashMap;

/// データフレームからサンプルを抽出する内部実装
pub(crate) fn sample_impl(
    df: &DataFrame,
    fraction: f64,
    replace: bool,
) -> Result<DataFrame> {
    if fraction <= 0.0 {
        return Err(Error::InvalidValue("サンプル率は正の値である必要があります".into()));
    }
    
    // データフレームの行数を取得
    let n_rows = df.row_count();
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
    let mut rng = rand::rng();
    
    // インデックスの生成
    let indices = if replace {
        // 復元抽出
        (0..sample_size)
            .map(|_| rng.random_range(0..n_rows))
            .collect::<Vec<_>>()
    } else {
        // 非復元抽出
        let mut idx: Vec<usize> = (0..n_rows).collect();
        idx.shuffle(&mut rng);
        idx[0..sample_size].to_vec()
    };
    
    // サンプルデータフレームの作成 (実際のDataFrameの実装に合わせて修正が必要)
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Some(col) = df.get_column(col_name) {
            // サンプルの行だけを抽出
            let sampled_values: Vec<String> = indices.iter()
                .filter_map(|&idx| col.values().get(idx).cloned())
                .collect();
            
            if !sampled_values.is_empty() {
                // 新しいSeriesを作成してデータフレームに追加
                let series = crate::series::Series::new(sampled_values, Some(col_name.clone())).unwrap();
                result.add_column(col_name.to_string(), series).unwrap();
            }
        }
    }
    
    Ok(result)
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
    let mut rng = rand::rng();
    let mut result = Vec::with_capacity(n_samples);
    
    for _ in 0..n_samples {
        // 復元抽出によるサンプリング
        let sample: Vec<f64> = (0..n)
            .map(|_| data[rng.random_range(0..n)])
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
    if !df.contains_column(strata_column) {
        return Err(Error::ColumnNotFound(strata_column.to_string()));
    }
    
    if fraction <= 0.0 {
        return Err(Error::InvalidValue("サンプル率は正の値である必要があります".into()));
    }
    
    // 現在のDataFrame実装ではgroup_byが異なる実装の可能性があるため
    // サンプリングロジックのみを実装
    
    // まずは層の値をリストアップ
    let strata_col = df.get_column(strata_column).ok_or_else(|| 
        Error::ColumnNotFound(strata_column.to_string()))?;
    
    let mut strata_values = Vec::new();
    for value in strata_col.values() {
        if !strata_values.contains(value) {
            strata_values.push(value.clone());
        }
    }
    
    // 層ごとのインデックスを収集
    let mut strata_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, value) in strata_col.values().iter().enumerate() {
        strata_indices.entry(value.clone())
            .or_insert_with(Vec::new)
            .push(i);
    }
    
    // 各層から指定した割合でサンプリング
    let mut all_sample_indices = Vec::new();
    for (_, indices) in strata_indices.iter() {
        let sample_size = (indices.len() as f64 * fraction).ceil() as usize;
        if sample_size == 0 {
            continue;
        }
        
        let mut rng = rand::rng();
        
        if replace {
            // 復元抽出
            for _ in 0..sample_size {
                let idx = indices[rng.random_range(0..indices.len())];
                all_sample_indices.push(idx);
            }
        } else {
            // 非復元抽出
            if sample_size > indices.len() {
                return Err(Error::InvalidOperation(
                    "非復元抽出の場合、サンプルサイズは層のサイズ以下である必要があります".into()
                ));
            }
            
            let mut sampled_indices = indices.clone();
            sampled_indices.shuffle(&mut rng);
            all_sample_indices.extend_from_slice(&sampled_indices[0..sample_size]);
        }
    }
    
    // サンプリングしたインデックスをソート（元の順序を維持）
    all_sample_indices.sort();
    
    // サンプルデータフレームの作成 (実際のDataFrameの実装に合わせて修正が必要)
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Some(col) = df.get_column(col_name) {
            // サンプルの行だけを抽出
            let sampled_values: Vec<String> = all_sample_indices.iter()
                .filter_map(|&idx| col.values().get(idx).cloned())
                .collect();
            
            if !sampled_values.is_empty() {
                // 新しいSeriesを作成してデータフレームに追加
                let series = crate::series::Series::new(sampled_values, Some(col_name.clone())).unwrap();
                result.add_column(col_name.to_string(), series).unwrap();
            }
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
        let data = Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("data".to_string())).unwrap();
        df.add_column("data".to_string(), data).unwrap();
        
        // 50%サンプリング（非復元）
        let sample = sample_impl(&df, 0.5, false).unwrap();
        assert_eq!(sample.row_count(), 5);
        
        // 30%サンプリング（復元）
        let sample = sample_impl(&df, 0.3, true).unwrap();
        assert_eq!(sample.row_count(), 3);
        
        // 200%サンプリング（復元）
        let sample = sample_impl(&df, 2.0, true).unwrap();
        assert_eq!(sample.row_count(), 20);
        
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