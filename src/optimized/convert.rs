//! DataFrameの相互変換機能を提供するモジュール

use std::collections::HashMap;

use crate::column::{Column, ColumnTrait, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};
use crate::index::DataFrameIndex;
use crate::optimized::dataframe::OptimizedDataFrame;

/// 標準のDataFrameからOptimizedDataFrameを作成する
pub(crate) fn from_standard_dataframe(df: &crate::dataframe::DataFrame) -> Result<OptimizedDataFrame> {
    let mut opt_df = OptimizedDataFrame::new();
    
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
                opt_df.add_column(col_name.clone(), Column::Int64(Int64Column::new(int_values)))?;
                continue;
            }
            
            // 浮動小数点型
            let all_floats = values.iter().all(|s| s.is_empty() || s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = values.iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                opt_df.add_column(col_name.clone(), Column::Float64(Float64Column::new(float_values)))?;
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
                opt_df.add_column(col_name.clone(), Column::Boolean(BooleanColumn::new(bool_values)))?;
                continue;
            }
            
            // デフォルトは文字列型
            opt_df.add_column(col_name.clone(), Column::String(StringColumn::new(values)))?;
        }
    }
    
    // インデックスのセット（あれば）
    if let Some(ref index) = df.get_index() {
        // 文字列ベースのインデックスとしてコピー
        let string_index = match index {
            DataFrameIndex::Simple(simple_index) => {
                let labels: Vec<String> = (0..simple_index.len())
                    .map(|i| simple_index.get_by_loc(i).map(|v| v.to_string()).unwrap_or_default())
                    .collect();
                DataFrameIndex::Simple(crate::index::Index::new(labels))
            },
            DataFrameIndex::Multi(_) => {
                // マルチインデックスのサポートは今後の課題
                // とりあえず連番インデックスを作成
                let labels: Vec<String> = (0..df.row_count())
                    .map(|i| i.to_string())
                    .collect();
                DataFrameIndex::Simple(crate::index::Index::new(labels))
            }
        };
        
        opt_df.index = Some(string_index);
    }
    
    Ok(opt_df)
}

/// OptimizedDataFrameを標準のDataFrameに変換する
pub(crate) fn to_standard_dataframe(df: &OptimizedDataFrame) -> Result<crate::dataframe::DataFrame> {
    let mut std_df = crate::dataframe::DataFrame::new();
    
    // 列データをコピー
    for col_name in df.column_names() {
        let col_idx = df.column_indices.get(col_name)
            .ok_or_else(|| Error::ColumnNotFound(col_name.clone()))?;
        
        let column = &df.columns[*col_idx];
        
        match column {
            Column::Int64(col) => {
                let mut series = crate::series::Series::new(
                    (0..col.len())
                        .map(|i| col.get(i).map_or(None, |v| v.map(crate::series::DataValue::Int64)))
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_series(series)?;
            },
            Column::Float64(col) => {
                let mut series = crate::series::Series::new(
                    (0..col.len())
                        .map(|i| col.get(i).map_or(None, |v| v.map(crate::series::DataValue::Float64)))
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_series(series)?;
            },
            Column::String(col) => {
                let mut series = crate::series::Series::new(
                    (0..col.len())
                        .map(|i| col.get(i).map_or(None, |v| v.map(|s| crate::series::DataValue::String(s.to_string()))))
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_series(series)?;
            },
            Column::Boolean(col) => {
                let mut series = crate::series::Series::new(
                    (0..col.len())
                        .map(|i| col.get(i).map_or(None, |v| v.map(crate::series::DataValue::Boolean)))
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_series(series)?;
            },
        }
    }
    
    // インデックスの設定（あれば）
    if let Some(ref index) = df.index {
        match index {
            DataFrameIndex::Simple(simple_index) => {
                let values: Vec<String> = (0..simple_index.len())
                    .map(|i| simple_index.get_by_loc(i).unwrap_or_default().to_string())
                    .collect();
                
                std_df.set_index_from_vec(values)?;
            },
            DataFrameIndex::Multi(_) => {
                // マルチインデックスのサポートは今後の課題
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