use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::Arc;
use std::path::Path;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};

use crate::column::{Column, ColumnTrait, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait, Index};
use crate::optimized::operations::JoinType;
use crate::optimized::split_dataframe::io::ParquetCompression;
use simple_excel_writer::{Workbook, Sheet};

/// 最適化されたDataFrame実装
/// 列指向ストレージを使用し、高速なデータ処理を実現
#[derive(Clone)]
pub struct OptimizedDataFrame {
    // 列データ
    columns: Vec<Column>,
    // 列名→インデックスのマッピング
    column_indices: HashMap<String, usize>,
    // 列の順序
    column_names: Vec<String>,
    // 行数
    row_count: usize,
    // インデックス (オプション)
    index: Option<DataFrameIndex<String>>,
}

/// 列に対するビュー（参照）を表す構造体
#[derive(Clone)]
pub struct ColumnView {
    column: Column,
}

impl Debug for OptimizedDataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 最大表示行数
        const MAX_ROWS: usize = 10;
        
        if self.columns.is_empty() {
            return write!(f, "OptimizedDataFrame (0 rows x 0 columns)");
        }
        
        writeln!(f, "OptimizedDataFrame ({} rows x {} columns):", self.row_count, self.columns.len())?;
        
        // 列ヘッダーの表示
        write!(f, "{:<5} |", "idx")?;
        for name in &self.column_names {
            write!(f, " {:<15} |", name)?;
        }
        writeln!(f)?;
        
        // 区切り線
        write!(f, "{:-<5}-+", "")?;
        for _ in &self.column_names {
            write!(f, "-{:-<15}-+", "")?;
        }
        writeln!(f)?;
        
        // 最大MAX_ROWS行まで表示
        let display_rows = std::cmp::min(self.row_count, MAX_ROWS);
        for i in 0..display_rows {
            if let Some(ref idx) = self.index {
                let idx_value = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => {
                        if i < simple_idx.len() {
                            simple_idx.get_value(i).map(|s| s.to_string()).unwrap_or_else(|| i.to_string())
                        } else {
                            i.to_string()
                        }
                    },
                    DataFrameIndex::Multi(_) => i.to_string()
                };
                write!(f, "{:<5} |", idx_value)?;
            } else {
                write!(f, "{:<5} |", i)?;
            }
            
            for col_idx in 0..self.columns.len() {
                let col = &self.columns[col_idx];
                let value = match col {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{:.3}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("\"{}\"", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                };
                write!(f, " {:<15} |", value)?;
            }
            writeln!(f)?;
        }
        
        // 省略表示
        if self.row_count > MAX_ROWS {
            writeln!(f, "... ({} more rows)", self.row_count - MAX_ROWS)?;
        }
        
        Ok(())
    }
}

impl OptimizedDataFrame {
    /// 新しい空のDataFrameを作成
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: 0,
            index: None,
        }
    }
    
    /// CSVファイルからDataFrameを作成する（高性能実装）
    /// # Arguments
    /// * `path` - CSVファイルのパス
    /// * `has_header` - ヘッダの有無
    /// # Returns
    /// * `Result<Self>` - 成功時はDataFrame、失敗時はエラー
    pub fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        // split_dataframe/io.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameのfrom_csvを呼び出す
        let split_df = SplitDataFrame::from_csv(path, has_header)?;
        
        // StandardDataFrameに変換（互換性のため）
        let mut df = Self::new();
        
        // 列データをコピー
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                // 以下は元のコードと同じ
                df.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }
        
        Ok(df)
    }
    
    /// データタイプを推測して最適な列を作成する（内部ヘルパー）
    fn infer_and_create_column(data: &[String], name: &str) -> Column {
        // 空のデータの場合は文字列列を返す
        if data.is_empty() {
            return Column::String(StringColumn::new(Vec::new()));
        }
        
        // 整数値チェック
        let is_int64 = data.iter()
            .all(|s| s.parse::<i64>().is_ok() || s.trim().is_empty());
        
        if is_int64 {
            let int_data: Vec<i64> = data.iter()
                .map(|s| s.parse::<i64>().unwrap_or(0))
                .collect();
            return Column::Int64(Int64Column::new(int_data));
        }
        
        // 浮動小数点チェック
        let is_float64 = data.iter()
            .all(|s| s.parse::<f64>().is_ok() || s.trim().is_empty());
        
        if is_float64 {
            let float_data: Vec<f64> = data.iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            return Column::Float64(Float64Column::new(float_data));
        }
        
        // ブール値チェック
        let bool_values = ["true", "false", "0", "1", "yes", "no", "t", "f"];
        let is_boolean = data.iter()
            .all(|s| bool_values.contains(&s.to_lowercase().trim()) || s.trim().is_empty());
        
        if is_boolean {
            let bool_data: Vec<bool> = data.iter()
                .map(|s| {
                    let lower = s.to_lowercase();
                    let trimmed = lower.trim();
                    match trimmed {
                        "true" | "1" | "yes" | "t" => true,
                        "false" | "0" | "no" | "f" => false,
                        _ => false, // 空文字列など
                    }
                })
                .collect();
            return Column::Boolean(BooleanColumn::new(bool_data));
        }
        
        // 他のすべてのケースでは文字列として処理
        Column::String(StringColumn::new(data.to_vec()))
    }
    
    /// DataFrameをCSVファイルに保存する
    /// # Arguments
    /// * `path` - 保存先のパス
    /// * `write_header` - ヘッダを書き込むかどうか
    /// # Returns
    /// * `Result<()>` - 成功時はOk、失敗時はエラー
    pub fn to_csv<P: AsRef<Path>>(&self, path: P, write_header: bool) -> Result<()> {
        // split_dataframe/io.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameに変換
        let mut split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのto_csvを呼び出す
        split_df.to_csv(path, write_header)
    }
    
    /// 列を追加
    pub fn add_column<C: Into<Column>>(&mut self, name: impl Into<String>, column: C) -> Result<()> {
        // split_dataframe/column_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        let name_str = name.into();
        let column_val = column.into();
        
        // オリジナルの実装はそのまま維持
        // 列名の重複チェック
        if self.column_indices.contains_key(&name_str) {
            return Err(Error::DuplicateColumnName(name_str));
        }
        
        // 行数の整合性チェック
        let column_len = column_val.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }
        
        // 列の追加
        let column_idx = self.columns.len();
        self.columns.push(column_val);
        self.column_indices.insert(name_str.clone(), column_idx);
        self.column_names.push(name_str);
        
        // 最初の列の場合は行数を設定
        if self.row_count == 0 {
            self.row_count = column_len;
        }
        
        Ok(())
    }
    
    /// 整数列を追加
    pub fn add_int_column(&mut self, name: impl Into<String>, data: Vec<i64>) -> Result<()> {
        self.add_column(name, Column::Int64(Int64Column::new(data)))
    }
    
    /// 浮動小数点列を追加
    pub fn add_float_column(&mut self, name: impl Into<String>, data: Vec<f64>) -> Result<()> {
        self.add_column(name, Column::Float64(Float64Column::new(data)))
    }
    
    /// 文字列列を追加
    pub fn add_string_column(&mut self, name: impl Into<String>, data: Vec<String>) -> Result<()> {
        self.add_column(name, Column::String(StringColumn::new(data)))
    }
    
    /// ブール列を追加
    pub fn add_boolean_column(&mut self, name: impl Into<String>, data: Vec<bool>) -> Result<()> {
        self.add_column(name, Column::Boolean(BooleanColumn::new(data)))
    }
    
    /// 列の参照を取得
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        // split_dataframe/column_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
    }
    
    /// 列の型を取得
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        // split_dataframe/column_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        Ok(self.columns[*column_idx].column_type())
    }
    
    /// 列名のリストを取得
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }
    
    /// 指定された列が存在するかチェックします
    pub fn contains_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }
    
    /// 列を削除
    pub fn remove_column(&mut self, name: &str) -> Result<Column> {
        // split_dataframe/column_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        // 列とそのインデックスを削除
        let column_idx = *column_idx;
        let removed_column = self.columns.remove(column_idx);
        self.column_indices.remove(name);
        
        // 列名リストから削除
        let name_idx = self.column_names.iter().position(|n| n == name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        self.column_names.remove(name_idx);
        
        // インデックスの再計算
        for (_, idx) in self.column_indices.iter_mut() {
            if *idx > column_idx {
                *idx -= 1;
            }
        }
        
        Ok(removed_column)
    }
    
    /// 列名を変更
    pub fn rename_column(&mut self, old_name: &str, new_name: impl Into<String>) -> Result<()> {
        // split_dataframe/column_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        let new_name = new_name.into();
        
        // 新しい名前が既に存在する場合はエラー
        if self.column_indices.contains_key(&new_name) && old_name != new_name {
            return Err(Error::DuplicateColumnName(new_name));
        }
        
        // 古い名前が存在するか確認
        let column_idx = *self.column_indices.get(old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;
        
        // インデックスと列名を更新
        self.column_indices.remove(old_name);
        self.column_indices.insert(new_name.clone(), column_idx);
        
        // 列名リストを更新
        let name_idx = self.column_names.iter().position(|n| n == old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;
        self.column_names[name_idx] = new_name;
        
        Ok(())
    }
    
    /// 指定された行と列の値を取得
    pub fn get_value(&self, row_idx: usize, column_name: &str) -> Result<Option<String>> {
        // split_dataframe/column_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }
        
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        // 列の型に応じて値を取得
        let value = match column {
            Column::Int64(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
            Column::Float64(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
            Column::String(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
            Column::Boolean(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
        };
        
        Ok(value)
    }
    
    /// DataFrameを縦方向に結合します。
    /// 互換性のある列を持つ2つのDataFrameを結合し、新しいDataFrameを作成します。
    pub fn append(&self, other: &OptimizedDataFrame) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }
        
        if other.columns.is_empty() {
            return Ok(self.clone());
        }
        
        // split_dataframe/data_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // 自身をSplitDataFrameに変換
        let mut self_split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                self_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                self_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // 相手をSplitDataFrameに変換
        let mut other_split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                other_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = other.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                other_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのappendを呼び出す
        let split_result = self_split_df.append(&other_split_df)?;
        
        // 結果を変換して返す
        let mut result = Self::new();
        
        // 列データをコピー
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }
        
        return Ok(result);
    }
    
    /// 行数を取得
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    
    /// 列数を取得
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
    
    /// インデックスを設定
    pub fn set_index(&mut self, name: &str) -> Result<()> {
        // 列の存在チェック
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        // 文字列列からインデックスを作成
        if let Column::String(string_col) = column {
            let mut index_values = Vec::new();
            let mut index_map = HashMap::new();
            
            for i in 0..string_col.len() {
                if let Ok(Some(value)) = string_col.get(i) {
                    let value_string = value.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                } else {
                    let value_string = i.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                }
            }
            
            let index = Index::with_name(index_values, Some(name.to_string()))?;
            self.index = Some(DataFrameIndex::from_simple(index));
            return Ok(());
        }
        
        // 整数列からインデックスを作成
        if let Column::Int64(int_col) = column {
            let mut index_values = Vec::new();
            let mut index_map = HashMap::new();
            
            for i in 0..int_col.len() {
                if let Ok(Some(value)) = int_col.get(i) {
                    let value_string = value.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                } else {
                    let value_string = i.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                }
            }
            
            let index = Index::with_name(index_values, Some(name.to_string()))?;
            self.index = Some(DataFrameIndex::from_simple(index));
            return Ok(());
        }
        
        Err(Error::Operation(format!(
            "列 '{}' はインデックスとして使用できる型ではありません", name
        )))
    }
    
    /// 整数インデックスを使用して行を取得（新しいDataFrameとして）
    pub fn get_row(&self, row_idx: usize) -> Result<Self> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }
        
        let mut result = Self::new();
        
        for (i, name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];
            
            let new_column = match column {
                Column::Int64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Int64(Int64Column::new(vec![value]))
                },
                Column::Float64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Float64(crate::column::Float64Column::new(vec![value]))
                },
                Column::String(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default().to_string();
                    Column::String(crate::column::StringColumn::new(vec![value]))
                },
                Column::Boolean(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Boolean(crate::column::BooleanColumn::new(vec![value]))
                },
            };
            
            result.add_column(name.clone(), new_column)?;
        }
        
        Ok(result)
    }
    
    /// 列の選択（新しいDataFrameとして）
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();
        
        for &name in columns {
            let column_idx = self.column_indices.get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
            
            let column = self.columns[*column_idx].clone();
            result.add_column(name.to_string(), column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// フィルタリング（新しいDataFrameとして）
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        // 条件列の取得
        let column_idx = self.column_indices.get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;
        
        let condition = &self.columns[*column_idx];
        
        // 条件列がブール型であることを確認
        if let Column::Boolean(bool_col) = condition {
            // trueの行のインデックスを収集
            let mut indices = Vec::new();
            for i in 0..bool_col.len() {
                if let Ok(Some(true)) = bool_col.get(i) {
                    indices.push(i);
                }
            }
            
            // 新しいDataFrameを作成
            let mut result = Self::new();
            
            // 各列をフィルタリング
            for (i, name) in self.column_names.iter().enumerate() {
                let column = &self.columns[i];
                
                let filtered_column = match column {
                    Column::Int64(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(0); // デフォルト値
                            }
                        }
                        Column::Int64(Int64Column::new(filtered_data))
                    },
                    Column::Float64(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(0.0); // デフォルト値
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(filtered_data))
                    },
                    Column::String(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val.to_string());
                            } else {
                                filtered_data.push(String::new()); // デフォルト値
                            }
                        }
                        Column::String(crate::column::StringColumn::new(filtered_data))
                    },
                    Column::Boolean(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(false); // デフォルト値
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                    },
                };
                
                result.add_column(name.clone(), filtered_column)?;
            }
            
            // 新しいインデックスの作成
            if let Some(ref idx) = self.index {
                let mut new_index_values = Vec::with_capacity(indices.len());
                let mut new_index_map = HashMap::new();
                
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    if old_idx < idx.len() {
                        // インデックス値を取得する代替方法
                        if let DataFrameIndex::Simple(ref simple_idx) = idx {
                            if old_idx < simple_idx.len() {
                                let value = simple_idx.get_value(old_idx).map(|s| s.to_string()).unwrap_or_else(|| old_idx.to_string());
                                new_index_values.push(value.clone());
                                new_index_map.insert(value, new_idx);
                            } else {
                                let value = old_idx.to_string();
                                new_index_values.push(value.clone());
                                new_index_map.insert(value, new_idx);
                            }
                        } else {
                            let value = old_idx.to_string();
                            new_index_values.push(value.clone());
                            new_index_map.insert(value, new_idx);
                        }
                    } else {
                        let value = old_idx.to_string();
                        new_index_values.push(value.clone());
                        new_index_map.insert(value, new_idx);
                    }
                }
                
                // 新しいインデックスを作成
                let name_opt = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => simple_idx.name().map(|s| s.to_string()),
                    DataFrameIndex::Multi(ref multi_idx) => multi_idx.names().first().cloned().flatten(),
                };
                
                let new_idx = Index::with_name(new_index_values, name_opt)?;
                result.index = Some(DataFrameIndex::from_simple(new_idx));
            }
            
            Ok(result)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }
    
    /// マッピング関数を適用（並列処理対応）
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send,
    {
        use rayon::prelude::*;
        
        let column_views: Vec<_> = self.column_names.iter()
            .map(|name| self.column(name))
            .collect::<Result<_>>()?;
        
        // 並列処理
        let new_columns: Result<Vec<_>> = column_views.par_iter()
            .map(|view| func(view))
            .collect();
        
        // 結果を新しいDataFrameに格納
        let new_columns = new_columns?;
        let mut result = Self::new();
        
        for (name, column) in self.column_names.iter().zip(new_columns) {
            result.add_column(name.clone(), column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 行のフィルタリング実行（データサイズに応じて直列/並列処理を自動選択）
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        use rayon::prelude::*;
        
        // 最適並列化のための閾値（これより小さいデータサイズでは直列処理が有利）
        const PARALLEL_THRESHOLD: usize = 100_000;
        
        // 条件列の取得
        let column_idx = self.column_indices.get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;
        
        let condition = &self.columns[*column_idx];
        
        // 条件列がブール型であることを確認
        if let Column::Boolean(bool_col) = condition {
            let row_count = bool_col.len();
            
            // データサイズに基づいて直列/並列処理を選択
            let indices: Vec<usize> = if row_count < PARALLEL_THRESHOLD {
                // 直列処理（小規模データ）
                (0..row_count)
                    .filter_map(|i| {
                        if let Ok(Some(true)) = bool_col.get(i) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // 並列処理（大規模データ）
                // チャンクサイズを最適化して並列化オーバーヘッドを削減
                let chunk_size = (row_count / rayon::current_num_threads()).max(1000);
                
                // まずレンジを配列に変換してからチャンク処理
                (0..row_count).collect::<Vec<_>>()
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.iter().filter_map(|&i| {
                            if let Ok(Some(true)) = bool_col.get(i) {
                                Some(i)
                            } else {
                                None
                            }
                        }).collect::<Vec<_>>()
                    })
                    .collect()
            };
            
            if indices.is_empty() {
                // 空のデータフレームを返す
                let mut result = Self::new();
                for name in &self.column_names {
                    let col_idx = self.column_indices[name];
                    let empty_col = match &self.columns[col_idx] {
                        Column::Int64(_) => Column::Int64(Int64Column::new(Vec::new())),
                        Column::Float64(_) => Column::Float64(crate::column::Float64Column::new(Vec::new())),
                        Column::String(_) => Column::String(crate::column::StringColumn::new(Vec::new())),
                        Column::Boolean(_) => Column::Boolean(crate::column::BooleanColumn::new(Vec::new())),
                    };
                    result.add_column(name.clone(), empty_col)?;
                }
                return Ok(result);
            }
            
            // 新しいDataFrameを作成
            let mut result = Self::new();
            
            // 結果の列を格納するベクトルを事前に確保
            let mut result_columns = Vec::with_capacity(self.column_names.len());
            
            // データサイズに基づいて列の処理方法を選択
            if indices.len() < PARALLEL_THRESHOLD || self.column_names.len() < 4 {
                // 直列処理（小規模データまたは列数が少ない場合）
                for name in &self.column_names {
                    let i = self.column_indices[name];
                    let column = &self.columns[i];
                    
                    let filtered_column = match column {
                        Column::Int64(col) => {
                            let filtered_data: Vec<i64> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        0 // デフォルト値
                                    }
                                })
                                .collect();
                            Column::Int64(Int64Column::new(filtered_data))
                        },
                        Column::Float64(col) => {
                            let filtered_data: Vec<f64> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        0.0 // デフォルト値
                                    }
                                })
                                .collect();
                            Column::Float64(crate::column::Float64Column::new(filtered_data))
                        },
                        Column::String(col) => {
                            let filtered_data: Vec<String> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val.to_string()
                                    } else {
                                        String::new() // デフォルト値
                                    }
                                })
                                .collect();
                            Column::String(crate::column::StringColumn::new(filtered_data))
                        },
                        Column::Boolean(col) => {
                            let filtered_data: Vec<bool> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        false // デフォルト値
                                    }
                                })
                                .collect();
                            Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                        },
                    };
                    
                    result_columns.push((name.clone(), filtered_column));
                }
            } else {
                // 大規模データの並列処理
                // 各列に対して並列処理を行う（列レベルの粗粒度並列化）
                result_columns = self.column_names.par_iter()
                    .map(|name| {
                        let i = self.column_indices[name];
                        let column = &self.columns[i];
                        
                        let indices_len = indices.len();
                        let filtered_column = match column {
                            Column::Int64(col) => {
                                // 大きなインデックスリストは分割して処理
                                let chunk_size = (indices_len / 8).max(1000);
                                let mut filtered_data = Vec::with_capacity(indices_len);
                                
                                // chunks を使用してすべての要素を確実に処理
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<i64> = chunk.iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val
                                            } else {
                                                0 // デフォルト値
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }
                                
                                Column::Int64(Int64Column::new(filtered_data))
                            },
                            Column::Float64(col) => {
                                // 大きなインデックスリストは分割して処理
                                let chunk_size = (indices_len / 8).max(1000);
                                let mut filtered_data = Vec::with_capacity(indices_len);
                                
                                // chunksを使用してすべての要素を確実に処理
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<f64> = chunk.iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val
                                            } else {
                                                0.0 // デフォルト値
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }
                                
                                Column::Float64(crate::column::Float64Column::new(filtered_data))
                            },
                            Column::String(col) => {
                                // 文字列処理は特に重いので、より細かいチャンク処理
                                let chunk_size = (indices_len / 16).max(500);
                                let mut filtered_data = Vec::with_capacity(indices_len);
                                
                                // chunksを使用してすべての要素を確実に処理
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<String> = chunk.iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val.to_string()
                                            } else {
                                                String::new() // デフォルト値
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }
                                
                                Column::String(crate::column::StringColumn::new(filtered_data))
                            },
                            Column::Boolean(col) => {
                                let filtered_data: Vec<bool> = indices.iter()
                                    .map(|&idx| {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            val
                                        } else {
                                            false // デフォルト値
                                        }
                                    })
                                    .collect();
                                Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                            },
                        };
                        
                        (name.clone(), filtered_column)
                    })
                    .collect();
            }
            
            // 結果をデータフレームに追加
            for (name, column) in result_columns {
                result.add_column(name, column)?;
            }
            
            // インデックスのコピー
            if let Some(ref idx) = self.index {
                result.index = Some(idx.clone());
            }
            
            Ok(result)
        } else {
            Err(Error::OperationFailed(format!(
                "列 '{}' はブール型ではありません", condition_column
            )))
        }
    }
    
    /// グループ化操作を並列で実行（データサイズに応じて最適化）
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        // split_dataframe/group.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameに変換
        let mut split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのpar_groupbyを呼び出す
        let split_result = split_df.par_groupby(group_by_columns)?;
        
        // 結果を変換して返す
        let mut result = HashMap::with_capacity(split_result.len());
        
        for (key, split_group_df) in split_result {
            // 各グループのSplitDataFrameをStandardDataFrameに変換
            let mut group_df = Self::new();
            
            // 列データをコピー
            for name in split_group_df.column_names() {
                if let Ok(column_view) = split_group_df.column(name) {
                    let column = column_view.column;
                    group_df.add_column(name.to_string(), column.clone())?;
                }
            }
            
            // インデックスがあれば設定
            if let Some(index) = split_group_df.get_index() {
                group_df.index = Some(index.clone());
            }
            
            result.insert(key, group_df);
        }
        
        Ok(result)
    }
    
    /// 指定された行インデックスでフィルタリング（内部ヘルパー）
    fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        use rayon::prelude::*;
        
        let mut result = Self::new();
        
        // 各列を並列でフィルタリング
        let column_results: Result<Vec<(String, Column)>> = self.column_names.par_iter()
            .map(|name| {
                let i = self.column_indices[name];
                let column = &self.columns[i];
                
                let filtered_column = match column {
                    Column::Int64(col) => {
                        let filtered_data: Vec<i64> = indices.iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val)
                                    } else {
                                        Some(0) // デフォルト値
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::Int64(Int64Column::new(filtered_data))
                    },
                    Column::Float64(col) => {
                        let filtered_data: Vec<f64> = indices.iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val)
                                    } else {
                                        Some(0.0) // デフォルト値
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::Float64(crate::column::Float64Column::new(filtered_data))
                    },
                    Column::String(col) => {
                        let filtered_data: Vec<String> = indices.iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val.to_string())
                                    } else {
                                        Some(String::new()) // デフォルト値
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::String(crate::column::StringColumn::new(filtered_data))
                    },
                    Column::Boolean(col) => {
                        let filtered_data: Vec<bool> = indices.iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val)
                                    } else {
                                        Some(false) // デフォルト値
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                    },
                };
                
                Ok((name.clone(), filtered_column))
            })
            .collect();
        
        // 結果をデータフレームに追加
        for (name, column) in column_results? {
            result.add_column(name, column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 内部結合
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // split_dataframe/join.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;
        
        // SplitDataFrameに変換
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();
        
        // 左側の列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // 右側の列データをコピー
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定（左側）
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // インデックスがあれば設定（右側）
        if let Some(ref index) = other.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのinner_joinを呼び出す
        let split_result = left_split_df.inner_join(&right_split_df, left_on, right_on)?;
        
        // 結果を変換して返す
        let mut result = Self::new();
        
        // 列データをコピー
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }
        
        Ok(result)
    }
    
    /// 左結合
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // split_dataframe/join.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;
        
        // SplitDataFrameに変換
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();
        
        // 左側の列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // 右側の列データをコピー
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定（左側）
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // インデックスがあれば設定（右側）
        if let Some(ref index) = other.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのleft_joinを呼び出す
        let split_result = left_split_df.left_join(&right_split_df, left_on, right_on)?;
        
        // 結果を変換して返す
        let mut result = Self::new();
        
        // 列データをコピー
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }
        
        Ok(result)
    }
    
    /// 右結合
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // split_dataframe/join.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;
        
        // SplitDataFrameに変換
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();
        
        // 左側の列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // 右側の列データをコピー
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定（左側）
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // インデックスがあれば設定（右側）
        if let Some(ref index) = other.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのright_joinを呼び出す
        let split_result = left_split_df.right_join(&right_split_df, left_on, right_on)?;
        
        // 結果を変換して返す
        let mut result = Self::new();
        
        // 列データをコピー
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }
        
        Ok(result)
    }
    
    /// 外部結合
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // split_dataframe/join.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;
        
        // SplitDataFrameに変換
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();
        
        // 左側の列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // 右側の列データをコピー
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定（左側）
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // インデックスがあれば設定（右側）
        if let Some(ref index) = other.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのouter_joinを呼び出す
        let split_result = left_split_df.outer_join(&right_split_df, left_on, right_on)?;
        
        // 結果を変換して返す
        let mut result = Self::new();
        
        // 列データをコピー
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }
        
        Ok(result)
    }
    
    /// 列に関数を適用し、結果の新しいDataFrameを返す（パフォーマンス最適化版）
    ///
    /// # Arguments
    /// * `f` - 適用する関数（列のビューを取り、新しい列を返す）
    /// * `columns` - 処理対象の列名（Noneの場合はすべての列）
    /// # Returns
    /// * `Result<Self>` - 処理結果のDataFrame
    pub fn apply<F>(&self, f: F, columns: Option<&[&str]>) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Send + Sync,
    {
        let mut result = Self::new();
        
        // 処理対象の列を決定
        let target_columns = if let Some(cols) = columns {
            // 指定された列のみを対象とする
            cols.iter()
                .map(|&name| {
                    self.column_indices.get(name)
                        .ok_or_else(|| Error::ColumnNotFound(name.to_string()))
                        .map(|&idx| (name, idx))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // すべての列を対象とする
            self.column_names.iter()
                .map(|name| {
                    let idx = self.column_indices[name];
                    (name.as_str(), idx)
                })
                .collect()
        };
        
        // 列ごとに関数を適用（パフォーマンス最適化のため並列処理を使用）
        use rayon::prelude::*;
        let processed_columns: Result<Vec<(String, Column)>> = target_columns
            .into_par_iter()  // 並列イテレーション
            .map(|(name, idx)| {
                // 列のビューを作成
                let view = ColumnView {
                    column: self.columns[idx].clone(),
                };
                
                // 関数を適用して新しい列を生成
                let new_column = f(&view)?;
                
                // 元の列と同じ行数であることを確認
                if new_column.len() != self.row_count {
                    return Err(Error::LengthMismatch {
                        expected: self.row_count,
                        actual: new_column.len(),
                    });
                }
                
                Ok((name.to_string(), new_column))
            })
            .collect();
            
        // 処理結果の列をDataFrameに追加
        for (name, column) in processed_columns? {
            result.add_column(name, column)?;
        }
            
        // 処理対象外の列をそのままコピー
        if columns.is_some() {
            for (name, idx) in self.column_names.iter().map(|name| (name, self.column_indices[name])) {
                if !result.column_indices.contains_key(name) {
                    result.add_column(name.clone(), self.columns[idx].clone_column())?;
                }
            }
        }
        
        // インデックスを新しいDataFrameにコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 要素ごとに関数を適用（applymap相当）
    ///
    /// # Arguments
    /// * `column_name` - 対象の列名
    /// * `f` - 適用する関数（列の型に応じた関数）
    /// # Returns
    /// * `Result<Self>` - 処理結果のDataFrame
    pub fn applymap<F, G, H, I>(&self, column_name: &str, f_str: F, f_int: G, f_float: H, f_bool: I) -> Result<Self>
    where
        F: Fn(&str) -> String + Send + Sync,
        G: Fn(&i64) -> i64 + Send + Sync,
        H: Fn(&f64) -> f64 + Send + Sync,
        I: Fn(&bool) -> bool + Send + Sync,
    {
        // 列の存在確認
        let col_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*col_idx];
        
        // 型に応じた処理
        let new_column = match column {
            Column::Int64(int_col) => {
                let mut new_data = Vec::with_capacity(int_col.len());
                
                for i in 0..int_col.len() {
                    if let Ok(Some(val)) = int_col.get(i) {
                        new_data.push(f_int(&val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(0);  // デフォルト値
                    }
                }
                
                Column::Int64(Int64Column::new(new_data))
            },
            Column::Float64(float_col) => {
                let mut new_data = Vec::with_capacity(float_col.len());
                
                for i in 0..float_col.len() {
                    if let Ok(Some(val)) = float_col.get(i) {
                        new_data.push(f_float(&val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(0.0);  // デフォルト値
                    }
                }
                
                Column::Float64(Float64Column::new(new_data))
            },
            Column::String(str_col) => {
                let mut new_data = Vec::with_capacity(str_col.len());
                
                for i in 0..str_col.len() {
                    if let Ok(Some(val)) = str_col.get(i) {
                        new_data.push(f_str(val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(String::new());  // デフォルト値
                    }
                }
                
                Column::String(StringColumn::new(new_data))
            },
            Column::Boolean(bool_col) => {
                let mut new_data = Vec::with_capacity(bool_col.len());
                
                for i in 0..bool_col.len() {
                    if let Ok(Some(val)) = bool_col.get(i) {
                        new_data.push(f_bool(&val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(false);  // デフォルト値
                    }
                }
                
                Column::Boolean(BooleanColumn::new(new_data))
            },
        };
        
        // 結果のDataFrameを作成
        let mut result = self.clone();
        
        // 既存の列を置き換え
        result.columns[*col_idx] = new_column;
        
        Ok(result)
    }
    
    /// DataFrameを「長形式」に変換する（melt操作）
    /// 
    /// 複数の列を単一の「変数」列と「値」列に変換します。
    /// パフォーマンスを最優先した実装です。
    ///
    /// # Arguments
    /// * `id_vars` - 変換せずに保持する列名（識別子列）
    /// * `value_vars` - 変換する列名（値列）。指定しない場合はid_vars以外のすべての列
    /// * `var_name` - 変数名の列名（デフォルト: "variable"）
    /// * `value_name` - 値の列名（デフォルト: "value"）
    ///
    /// # Returns
    /// * `Result<Self>` - 長形式に変換されたDataFrame
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<Self> {
        // split_dataframe/data_ops.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameに変換
        let mut split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのmeltを呼び出す
        let split_result = split_df.melt(id_vars, value_vars, var_name, value_name)?;
        
        // 結果を変換して返す
        let mut result = Self::new();
        
        // 列データをコピー
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }
        
        Ok(result)
    }
}

/// ビューされた列に対する操作
impl ColumnView {
    /// 列の型を取得
    pub fn column_type(&self) -> ColumnType {
        self.column.column_type()
    }
    
    /// 列の長さを取得
    pub fn len(&self) -> usize {
        self.column.len()
    }
    
    /// 列が空かどうかを確認
    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }
    
    /// 整数列としてアクセス
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        if let Column::Int64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// 浮動小数点列としてアクセス
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        if let Column::Float64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// 文字列列としてアクセス
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        if let Column::String(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// ブール列としてアクセス
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        if let Column::Boolean(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// 内部のColumnへの参照を取得
    pub fn column(&self) -> &Column {
        &self.column
    }
    
    /// 内部のColumnを取得（消費的）
    pub fn into_column(self) -> Column {
        self.column
    }
}

// IO関連のメソッドを追加
impl OptimizedDataFrame {

    /// Excelファイルからデータフレームを読み込む
    pub fn from_excel<P: AsRef<Path>>(
        path: P, 
        sheet_name: Option<&str>,
        header: bool,
        skip_rows: usize,
        use_cols: Option<&[&str]>,
    ) -> Result<Self> {
        // split_dataframe/io.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameのfrom_excelを呼び出す
        let split_df = SplitDataFrame::from_excel(path, sheet_name, header, skip_rows, use_cols)?;
        
        // StandardDataFrameに変換（互換性のため）
        let mut df = Self::new();
        
        // 列データをコピー
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                // 以下は元のコードと同じ
                df.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }
        
        Ok(df)
    }

    /// データフレームをExcelファイルに書き込む
    pub fn to_excel<P: AsRef<Path>>(
        &self,
        path: P,
        sheet_name: Option<&str>,
        index: bool,
    ) -> Result<()> {
        // split_dataframe/io.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameに変換
        let mut split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // SplitDataFrameのto_excelを呼び出す
        split_df.to_excel(path, sheet_name, index)
    }

    /// データフレームをParquetファイルに書き込む
    pub fn to_parquet<P: AsRef<Path>>(
        &self,
        path: P,
        compression: Option<ParquetCompression>,
    ) -> Result<()> {
        // split_dataframe/io.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::io::ParquetCompression as SplitParquetCompression;
        
        // SplitDataFrameに変換
        let mut split_df = SplitDataFrame::new();
        
        // 列データをコピー
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(ref index) = self.index {
            // DataFrameIndexからIndex<String>を取り出す
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: マルチインデックスの場合の処理
        }
        
        // 圧縮設定を変換
        let split_compression = compression.map(|c| match c {
            ParquetCompression::None => SplitParquetCompression::None,
            ParquetCompression::Snappy => SplitParquetCompression::Snappy,
            ParquetCompression::Gzip => SplitParquetCompression::Gzip,
            ParquetCompression::Lzo => SplitParquetCompression::Lzo,
            ParquetCompression::Brotli => SplitParquetCompression::Brotli,
            ParquetCompression::Lz4 => SplitParquetCompression::Lz4,
            ParquetCompression::Zstd => SplitParquetCompression::Zstd,
        });
        
        // SplitDataFrameのto_parquetを呼び出す
        split_df.to_parquet(path, split_compression)
    }

    /// Parquetファイルからデータフレームを読み込む
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        // split_dataframe/io.rsの実装を利用
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        
        // SplitDataFrameのfrom_parquetを呼び出す
        let split_df = SplitDataFrame::from_parquet(path)?;
        
        // StandardDataFrameに変換（互換性のため）
        let mut df = Self::new();
        
        // 列データをコピー
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                // 以下は元のコードと同じ
                df.add_column(name.to_string(), column.clone())?;
            }
        }
        
        // インデックスがあれば設定
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }
        
        Ok(df)
    }

    /// 標準のDataFrameからOptimizedDataFrameを作成する
    fn from_standard_dataframe(df: &crate::dataframe::DataFrame) -> Result<Self> {
        let mut opt_df = Self::new();
        
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
                            s == "true" || s == "1"
                        })
                        .collect();
                    opt_df.add_column(col_name.clone(), Column::Boolean(BooleanColumn::new(bool_values)))?;
                    continue;
                }
                
                // デフォルトは文字列型
                opt_df.add_column(col_name.clone(), Column::String(StringColumn::new(values)))?;
            }
        }
        
        Ok(opt_df)
    }
    
    /// OptimizedDataFrameを標準のDataFrameに変換する
    fn to_standard_dataframe(&self) -> Result<crate::dataframe::DataFrame> {
        let mut df = crate::dataframe::DataFrame::new();
        
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let col = &self.columns[col_idx];
            let mut series_values = Vec::with_capacity(self.row_count);
            
            for row_idx in 0..self.row_count {
                let value = match col {
                    Column::Int64(int_col) => {
                        match int_col.get(row_idx) {
                            Ok(Some(val)) => val.to_string(),
                            _ => String::new(),
                        }
                    },
                    Column::Float64(float_col) => {
                        match float_col.get(row_idx) {
                            Ok(Some(val)) => val.to_string(),
                            _ => String::new(),
                        }
                    },
                    Column::String(str_col) => {
                        match str_col.get(row_idx) {
                            Ok(Some(val)) => val.to_string(),
                            _ => String::new(),
                        }
                    },
                    Column::Boolean(bool_col) => {
                        match bool_col.get(row_idx) {
                            Ok(Some(val)) => val.to_string(),
                            _ => String::new(),
                        }
                    },
                };
                series_values.push(value);
            }
            
            // シリーズを作成して追加
            let series = crate::series::Series::new(series_values, Some(col_name.clone()))?;
            df.add_column(col_name.clone(), series)?;
        }
        
        Ok(df)
    }
}