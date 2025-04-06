//! DataFrameの相互変換機能を提供するモジュール

use std::collections::HashMap;

use crate::column::{Column, ColumnTrait, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::dataframe::DataValue;
use crate::error::{Error, Result};
use crate::index::DataFrameIndex;
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

/// 標準のDataFrameからOptimizedDataFrameを作成する
pub(crate) fn from_standard_dataframe(df: &crate::dataframe::DataFrame) -> Result<OptimizedDataFrame> {
    // 新しいSplitDataFrameを作成（内部実装を使用）
    let mut split_df = SplitDataFrame::new();
    
    for col_name in df.column_names() {
        if let Some(col) = df.get_column(col_name) {
            // Seriesのイテレート用に値を一つずつ取り出し
            let mut values = Vec::new();
            for i in 0..col.len() {
                if let Some(val) = col.get(i) {
                    values.push(val.to_string());
                } else {
                    values.push(String::new());
                }
            }
            
            // 型推論して列を追加
            // 整数型
            let all_ints = values.iter().all(|s| s.is_empty() || s.parse::<i64>().is_ok());
            if all_ints {
                let int_values: Vec<i64> = values.iter()
                    .map(|s| s.parse::<i64>().unwrap_or(0))
                    .collect();
                split_df.add_column(col_name.clone(), Column::Int64(Int64Column::new(int_values)))?;
                continue;
            }
            
            // 浮動小数点型
            let all_floats = values.iter().all(|s| s.is_empty() || s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = values.iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                split_df.add_column(col_name.clone(), Column::Float64(Float64Column::new(float_values)))?;
                continue;
            }
            
            // ブール型
            let all_bools = values.iter().all(|s| {
                let s = s.to_lowercase();
                s.is_empty() || s == "true" || s == "false" || s == "1" || s == "0"
            });
            if all_bools {
                let bool_values: Vec<bool> = values.iter()
                    .map(|s| {
                        let s = s.to_lowercase();
                        !s.is_empty() && (s == "true" || s == "1")
                    })
                    .collect();
                split_df.add_column(col_name.clone(), Column::Boolean(BooleanColumn::new(bool_values)))?;
                continue;
            }
            
            // デフォルトは文字列型
            split_df.add_column(col_name.clone(), Column::String(StringColumn::new(values)))?;
        }
    }
    
    // インデックスの取得と設定
    // DataFrameは常にインデックスを持っている
    let df_index = df.get_index();
    
    // DataFrameIndexの種類に応じて処理
    match df_index {
        DataFrameIndex::Simple(simple_index) => {
            // Simple Indexの場合は直接コピー
            split_df.set_index_from_simple_index(simple_index.clone())?;
        },
        DataFrameIndex::Multi(_) => {
            // マルチインデックスの場合は今後の課題
            split_df.set_default_index()?;
        }
    }
    
    // SplitDataFrameをOptimizedDataFrameに変換
    let mut opt_df = OptimizedDataFrame::new();
    
    // 列データをコピー（公開APIを使用）
    for name in split_df.column_names() {
        if let Ok(column_view) = split_df.column(name) {
            let column = column_view.column().clone();
            opt_df.add_column(name.to_string(), column)?;
        }
    }
    
    // インデックスの設定
    if let Some(split_index) = split_df.get_index() {
        // シンプルなインデックスの場合は、SplitDataFrameからOptimizedDataFrameにコピー
        if let DataFrameIndex::Simple(simple_index) = split_index {
            // 互換性を保ちながらインデックスを設定
            let _ = opt_df.set_default_index();  // まずデフォルトのインデックスを作成
            
            // 現時点では公開APIだけで完全な変換は難しいため、既に作成された
            // DataFrameの完全性を確保するために一時的に対応
            // TODO: 適切なパブリックAPIを作成する
            #[allow(deprecated)]
            {
                opt_df.set_index_from_simple_index_internal(simple_index.clone())?;
            }
        }
    }
    
    Ok(opt_df)
}

/// OptimizedDataFrameを標準のDataFrameに変換する
pub(crate) fn to_standard_dataframe(df: &OptimizedDataFrame) -> Result<crate::dataframe::DataFrame> {
    // 内部のSplitDataFrameを使用
    let mut split_df = SplitDataFrame::new();
    
    // 列データを変換
    for col_name in df.column_names() {
        let col_view = df.column(col_name)?;
        let col = col_view.column();
        
        // SplitDataFrameに列を追加
        split_df.add_column(col_name.clone(), col.clone())?;
    }
    
    // インデックスがあれば設定
    if let Some(df_index) = df.get_index() {
        // 直接内部フィールドを設定せず、適切なメソッドを使用
        if let DataFrameIndex::Simple(simple_index) = df_index {
            split_df.set_index_from_simple_index(simple_index.clone())?;
        } else if let DataFrameIndex::Multi(multi_index) = df_index {
            // マルチインデックスはまだサポート外
            split_df.set_default_index()?;
        }
    }
    
    // 標準のDataFrameに変換
    let mut std_df = crate::dataframe::DataFrame::new();
    
    // 各列を処理
    for col_name in split_df.column_names() {
        let col_view = split_df.column(col_name)?;
        let col = col_view.column();
        
        match col {
            Column::Int64(int_col) => {
                // Int64Columnからシリーズを作成
                let series = crate::series::Series::new(
                    (0..int_col.len())
                        .map(|i| {
                            let val = int_col.get(i);
                            match val {
                                Ok(Some(v)) => Some(crate::dataframe::DataBox(Box::new(v.clone()))),
                                _ => None
                            }
                        })
                        .collect(),
                    Some(col_name.clone())
                )?;
                std_df.add_column(col_name.clone(), series)?;
            },
            Column::Float64(float_col) => {
                // Float64Columnからシリーズを作成
                let series = crate::series::Series::new(
                    (0..float_col.len())
                        .map(|i| {
                            let val = float_col.get(i);
                            match val {
                                Ok(Some(v)) => Some(crate::dataframe::DataBox(Box::new(v.clone()))),
                                _ => None
                            }
                        })
                        .collect(),
                    Some(col_name.clone())
                )?;
                std_df.add_column(col_name.clone(), series)?;
            },
            Column::String(str_col) => {
                // StringColumnからシリーズを作成
                let series = crate::series::Series::new(
                    (0..str_col.len())
                        .map(|i| {
                            let val = str_col.get(i);
                            match val {
                                Ok(Some(s)) => Some(crate::dataframe::DataBox(Box::new(s.to_string()))),
                                _ => None
                            }
                        })
                        .collect(),
                    Some(col_name.clone())
                )?;
                std_df.add_column(col_name.clone(), series)?;
            },
            Column::Boolean(bool_col) => {
                // BooleanColumnからシリーズを作成
                let series = crate::series::Series::new(
                    (0..bool_col.len())
                        .map(|i| {
                            let val = bool_col.get(i);
                            match val {
                                Ok(Some(v)) => Some(crate::dataframe::DataBox(Box::new(v.clone()))),
                                _ => None
                            }
                        })
                        .collect(),
                    Some(col_name.clone())
                )?;
                std_df.add_column(col_name.clone(), series)?;
            },
        }
    }
    
    // インデックスの設定
    if let Some(split_index) = split_df.get_index() {
        match split_index {
            DataFrameIndex::Simple(simple_index) => {
                // 文字列ベースのSimple Indexとして設定
                std_df.set_index(simple_index.clone())?;
            },
            DataFrameIndex::Multi(multi_index) => {
                // マルチインデックスの場合
                std_df.set_multi_index(multi_index.clone())?;
            }
        }
    }
    
    Ok(std_df)
}

/// 標準のDataFrameを受け取り、最適化されたOptimizedDataFrameに変換する公開関数
pub fn optimize_dataframe(df: &crate::dataframe::DataFrame) -> Result<OptimizedDataFrame> {
    from_standard_dataframe(df)
}

/// OptimizedDataFrameを標準のDataFrameに変換する公開関数
pub fn standard_dataframe(df: &OptimizedDataFrame) -> Result<crate::dataframe::DataFrame> {
    to_standard_dataframe(df)
}