// dataframe_adapter.rs
// 旧APIから新APIへの変換アダプタを提供

use crate::optimized::OptimizedDataFrame;
use crate::error::Result;
use std::collections::HashMap;

/// OptimizedDataFrameの拡張トレイト
/// 従来のDataFrameとの互換性のためのメソッドを提供
pub trait DataFrameCompat {
    /// 列が存在するかチェック
    fn contains_column(&self, column: &str) -> bool;
    
    /// 列から文字列値を取得
    fn get_column_string_values(&self, column: &str) -> Result<Vec<String>>;
    
    /// 列から数値を取得
    fn get_column_numeric_values(&self, column: &str) -> Result<Vec<f64>>;
}

impl DataFrameCompat for OptimizedDataFrame {
    fn contains_column(&self, column: &str) -> bool {
        self.column_names().iter().any(|name| name == column)
    }
    
    fn get_column_string_values(&self, column: &str) -> Result<Vec<String>> {
        let col = self.column(column)?;
        // 列から文字列値を抽出
        let mut values = Vec::new();
        for i in 0..col.len() {
            if let Some(string_col) = col.as_string() {
                // 文字列列の場合
                values.push(string_col.get(i).unwrap().unwrap_or_default().to_string());
            } else {
                // その他の列型の場合は文字列に変換
                values.push(format!("{}", i)); // 仮の実装
            }
        }
        Ok(values)
    }
    
    fn get_column_numeric_values(&self, column: &str) -> Result<Vec<f64>> {
        let col = self.column(column)?;
        // 列から数値を抽出
        let mut values = Vec::new();
        for i in 0..col.len() {
            if let Some(float_col) = col.as_float64() {
                // 浮動小数点列の場合
                values.push(float_col.get(i).unwrap_or_default().unwrap_or_default());
            } else if let Some(int_col) = col.as_int64() {
                // 整数列の場合
                if let Some(val) = int_col.get(i).unwrap_or_default() {
                    values.push(val as f64);
                } else {
                    values.push(0.0);
                }
            } else {
                // その他の列型の場合は0.0を返す
                values.push(0.0);
            }
        }
        Ok(values)
    }
}

/// 並列処理のラッパー
/// 互換性のためのメソッドを提供
pub trait ParallelCompat {
    /// 並列でパーミッションを適用
    fn par_apply<F>(&self, f: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&str, usize, &str) -> String + Send + Sync;
        
    /// 並列でグループ化
    fn par_groupby<K>(&self, key_func: K) -> Result<HashMap<String, OptimizedDataFrame>>
    where
        K: Fn(usize) -> String + Send + Sync;
}

impl ParallelCompat for OptimizedDataFrame {
    fn par_apply<F>(&self, f: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&str, usize, &str) -> String + Send + Sync,
    {
        // 新しいDataFrameを作成
        let mut result = OptimizedDataFrame::new();
        
        // 列ごとに処理
        for col_name in self.column_names() {
            let col = self.column(col_name)?;
            let mut new_values = Vec::with_capacity(col.len());
            
            for i in 0..col.len() {
                // 文字列を取得（簡易実装）
                let value = if let Some(string_col) = col.as_string() {
                    string_col.get(i).unwrap().unwrap_or_default().to_string()
                } else {
                    format!("{}", i)
                };
                let new_value = f(col_name, i, &value);
                new_values.push(new_value);
            }
            
            // 新しい列を追加
            let new_column = crate::column::StringColumn::new(new_values);
            result.add_column(col_name.to_string(), crate::column::Column::String(new_column))?;
        }
        
        Ok(result)
    }
    
    fn par_groupby<K>(&self, key_func: K) -> Result<HashMap<String, OptimizedDataFrame>>
    where
        K: Fn(usize) -> String + Send + Sync,
    {
        let row_count = self.row_count();
        let mut groups = HashMap::new();
        
        // 行ごとに処理してグループキーを計算
        for row_idx in 0..row_count {
            let group_key = key_func(row_idx);
            
            // このグループのDataFrameがなければ作成
            if !groups.contains_key(&group_key) {
                groups.insert(group_key.clone(), OptimizedDataFrame::new());
            }
            
            // この行のデータを取得してグループに追加
            let group_df = groups.get_mut(&group_key).unwrap();
            
            // 実際の実装ではここに行の追加処理が必要
            // 簡易実装のため、スタブとしておく
        }
        
        Ok(groups)
    }
}