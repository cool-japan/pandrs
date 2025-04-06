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
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(has_header)
            .from_reader(reader);
        
        // 列名を取得
        let headers = if has_header {
            csv_reader.headers()?.clone()
        } else {
            let record = csv_reader.records().next()
                .ok_or_else(|| Error::InvalidInput("CSVファイルが空です".to_string()))??;
            
            // 列番号を列名として使用
            csv::StringRecord::from(
                (0..record.len()).map(|i| format!("Column{}", i)).collect::<Vec<String>>()
            )
        };
        
        // 各列のデータを格納するベクター
        let mut column_data: Vec<Vec<String>> = vec![Vec::new(); headers.len()];
        
        // レコードを読み込み、列単位にデータを整理（パフォーマンス最適化）
        for result in csv_reader.records() {
            let record = result?;
            for (i, field) in record.iter().enumerate() {
                if i < column_data.len() {
                    column_data[i].push(field.to_string());
                }
            }
        }
        
        // データをプリアロケーションしてDataFrameを構築（メモリ効率向上）
        let mut df = Self::new();
        let row_count = if column_data.is_empty() { 0 } else { column_data[0].len() };
        
        // 並列処理で列を追加
        for (i, header) in headers.iter().enumerate() {
            if i < column_data.len() {
                // 列タイプを自動推定して最適な形式で格納
                let column = Self::infer_and_create_column(&column_data[i], header);
                df.add_column(header.to_string(), column)?;
            }
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
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        let mut csv_writer = csv::WriterBuilder::new()
            .has_headers(write_header)
            .from_writer(writer);
        
        // ヘッダ書き込み
        if write_header {
            csv_writer.write_record(&self.column_names)?;
        }
        
        // 行単位でデータを書き込む（パフォーマンス最適化）
        for row_idx in 0..self.row_count {
            let mut record = Vec::with_capacity(self.column_count());
            
            // 各列からこの行の値を取得
            for col_idx in 0..self.columns.len() {
                let col = &self.columns[col_idx];
                let value = match col {
                    Column::Int64(c) => {
                        match c.get(row_idx) {
                            Ok(Some(v)) => v.to_string(),
                            _ => String::new(),
                        }
                    },
                    Column::Float64(c) => {
                        match c.get(row_idx) {
                            Ok(Some(v)) => v.to_string(),
                            _ => String::new(),
                        }
                    },
                    Column::String(c) => {
                        match c.get(row_idx) {
                            Ok(Some(v)) => v.to_string(),
                            _ => String::new(),
                        }
                    },
                    Column::Boolean(c) => {
                        match c.get(row_idx) {
                            Ok(Some(v)) => v.to_string(),
                            _ => String::new(),
                        }
                    },
                };
                record.push(value);
            }
            
            csv_writer.write_record(&record)?;
        }
        
        csv_writer.flush()?;
        Ok(())
    }
    
    /// 列を追加
    pub fn add_column<C: Into<Column>>(&mut self, name: impl Into<String>, column: C) -> Result<()> {
        let name = name.into();
        let column = column.into();
        
        // 列名の重複チェック
        if self.column_indices.contains_key(&name) {
            return Err(Error::DuplicateColumnName(name));
        }
        
        // 行数の整合性チェック
        let column_len = column.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }
        
        // 列の追加
        let column_idx = self.columns.len();
        self.columns.push(column);
        self.column_indices.insert(name.clone(), column_idx);
        self.column_names.push(name);
        
        // 最初の列の場合は行数を設定
        if self.row_count == 0 {
            self.row_count = column_len;
        }
        
        Ok(())
    }
    
    /// 列の参照を取得
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
    }
    
    /// 列の型を取得
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
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
    
    /// DataFrameを縦方向に結合します。
    /// 互換性のある列を持つ2つのDataFrameを結合し、新しいDataFrameを作成します。
    pub fn append(&self, other: &OptimizedDataFrame) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }
        
        if other.columns.is_empty() {
            return Ok(self.clone());
        }
        
        // 新しいDataFrameの初期化
        let mut result = Self::new();
        let mut all_column_names = self.column_names.clone();
        
        // 他方にのみ存在する列名を追加
        for col_name in other.column_names() {
            if !all_column_names.contains(&col_name.to_string()) {
                all_column_names.push(col_name.to_string());
            }
        }
        
        // 各列を結合
        for col_name in &all_column_names {
            // 結合された列を作成
            let combined_column = match (self.contains_column(col_name), other.contains_column(col_name)) {
                (true, true) => {
                    // 両方のDataFrameに列が存在する場合
                    let self_col = self.column(col_name)?;
                    let other_col = other.column(col_name)?;
                    
                    // 列のデータ型が一致しているか確認
                    if self_col.column_type() != other_col.column_type() {
                        return Err(Error::InvalidInput(format!(
                            "列 {} のデータ型が一致しません", col_name
                        )));
                    }
                    
                    // 列型に応じて結合
                    match (self_col.column(), other_col.column()) {
                        (Column::Int64(self_int), Column::Int64(other_int)) => {
                            let mut new_data = Vec::with_capacity(self_int.len() + other_int.len());
                            
                            // 自身のデータをコピー
                            for i in 0..self_int.len() {
                                if let Ok(Some(val)) = self_int.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のデータを追加
                            for i in 0..other_int.len() {
                                if let Ok(Some(val)) = other_int.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::Int64(Int64Column::new(new_data))
                        },
                        (Column::Float64(self_float), Column::Float64(other_float)) => {
                            let mut new_data = Vec::with_capacity(self_float.len() + other_float.len());
                            
                            // 自身のデータをコピー
                            for i in 0..self_float.len() {
                                if let Ok(Some(val)) = self_float.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0.0); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のデータを追加
                            for i in 0..other_float.len() {
                                if let Ok(Some(val)) = other_float.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0.0); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::Float64(Float64Column::new(new_data))
                        },
                        (Column::String(self_str), Column::String(other_str)) => {
                            let mut new_data = Vec::with_capacity(self_str.len() + other_str.len());
                            
                            // 自身のデータをコピー
                            for i in 0..self_str.len() {
                                if let Ok(Some(val)) = self_str.get(i) {
                                    new_data.push(val.to_string());
                                } else {
                                    new_data.push("".to_string()); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のデータを追加
                            for i in 0..other_str.len() {
                                if let Ok(Some(val)) = other_str.get(i) {
                                    new_data.push(val.to_string());
                                } else {
                                    new_data.push("".to_string()); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::String(StringColumn::new(new_data))
                        },
                        (Column::Boolean(self_bool), Column::Boolean(other_bool)) => {
                            let mut new_data = Vec::with_capacity(self_bool.len() + other_bool.len());
                            
                            // 自身のデータをコピー
                            for i in 0..self_bool.len() {
                                if let Ok(Some(val)) = self_bool.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(false); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のデータを追加
                            for i in 0..other_bool.len() {
                                if let Ok(Some(val)) = other_bool.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(false); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::Boolean(BooleanColumn::new(new_data))
                        },
                        _ => {
                            // 型の組み合わせが一致しない場合（前の条件でチェック済みなのでここには来ないはず）
                            return Err(Error::InvalidInput(format!(
                                "互換性のない列型の組み合わせ: {}", col_name
                            )));
                        }
                    }
                },
                (true, false) => {
                    // 自身にのみ列が存在する場合、他方のDataFrame分のNULL値を追加
                    let self_col = self.column(col_name)?;
                    
                    match self_col.column() {
                        Column::Int64(self_int) => {
                            let mut new_data = Vec::with_capacity(self_int.len() + other.row_count());
                            
                            // 自身のデータをコピー
                            for i in 0..self_int.len() {
                                if let Ok(Some(val)) = self_int.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のDataFrame分のNULL値を追加
                            for _ in 0..other.row_count() {
                                new_data.push(0); // NAとして0を追加
                            }
                            
                            Column::Int64(Int64Column::new(new_data))
                        },
                        Column::Float64(self_float) => {
                            let mut new_data = Vec::with_capacity(self_float.len() + other.row_count());
                            
                            // 自身のデータをコピー
                            for i in 0..self_float.len() {
                                if let Ok(Some(val)) = self_float.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0.0); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のDataFrame分のNULL値を追加
                            for _ in 0..other.row_count() {
                                new_data.push(0.0); // NAとして0.0を追加
                            }
                            
                            Column::Float64(Float64Column::new(new_data))
                        },
                        Column::String(self_str) => {
                            let mut new_data = Vec::with_capacity(self_str.len() + other.row_count());
                            
                            // 自身のデータをコピー
                            for i in 0..self_str.len() {
                                if let Ok(Some(val)) = self_str.get(i) {
                                    new_data.push(val.to_string());
                                } else {
                                    new_data.push("".to_string()); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のDataFrame分のNULL値を追加
                            for _ in 0..other.row_count() {
                                new_data.push("".to_string()); // NAとして空文字列を追加
                            }
                            
                            Column::String(StringColumn::new(new_data))
                        },
                        Column::Boolean(self_bool) => {
                            let mut new_data = Vec::with_capacity(self_bool.len() + other.row_count());
                            
                            // 自身のデータをコピー
                            for i in 0..self_bool.len() {
                                if let Ok(Some(val)) = self_bool.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(false); // NAの場合はデフォルト値
                                }
                            }
                            
                            // 他方のDataFrame分のNULL値を追加
                            for _ in 0..other.row_count() {
                                new_data.push(false); // NAとしてfalseを追加
                            }
                            
                            Column::Boolean(BooleanColumn::new(new_data))
                        }
                    }
                },
                (false, true) => {
                    // 他方にのみ列が存在する場合、自身のDataFrame分のNULL値を追加
                    let other_col = other.column(col_name)?;
                    
                    match other_col.column() {
                        Column::Int64(other_int) => {
                            let mut new_data = Vec::with_capacity(self.row_count() + other_int.len());
                            
                            // 自身のDataFrame分のNULL値を追加
                            for _ in 0..self.row_count() {
                                new_data.push(0); // NAとして0を追加
                            }
                            
                            // 他方のデータをコピー
                            for i in 0..other_int.len() {
                                if let Ok(Some(val)) = other_int.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::Int64(Int64Column::new(new_data))
                        },
                        Column::Float64(other_float) => {
                            let mut new_data = Vec::with_capacity(self.row_count() + other_float.len());
                            
                            // 自身のDataFrame分のNULL値を追加
                            for _ in 0..self.row_count() {
                                new_data.push(0.0); // NAとして0.0を追加
                            }
                            
                            // 他方のデータをコピー
                            for i in 0..other_float.len() {
                                if let Ok(Some(val)) = other_float.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(0.0); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::Float64(Float64Column::new(new_data))
                        },
                        Column::String(other_str) => {
                            let mut new_data = Vec::with_capacity(self.row_count() + other_str.len());
                            
                            // 自身のDataFrame分のNULL値を追加
                            for _ in 0..self.row_count() {
                                new_data.push("".to_string()); // NAとして空文字列を追加
                            }
                            
                            // 他方のデータをコピー
                            for i in 0..other_str.len() {
                                if let Ok(Some(val)) = other_str.get(i) {
                                    new_data.push(val.to_string());
                                } else {
                                    new_data.push("".to_string()); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::String(StringColumn::new(new_data))
                        },
                        Column::Boolean(other_bool) => {
                            let mut new_data = Vec::with_capacity(self.row_count() + other_bool.len());
                            
                            // 自身のDataFrame分のNULL値を追加
                            for _ in 0..self.row_count() {
                                new_data.push(false); // NAとしてfalseを追加
                            }
                            
                            // 他方のデータをコピー
                            for i in 0..other_bool.len() {
                                if let Ok(Some(val)) = other_bool.get(i) {
                                    new_data.push(val);
                                } else {
                                    new_data.push(false); // NAの場合はデフォルト値
                                }
                            }
                            
                            Column::Boolean(BooleanColumn::new(new_data))
                        }
                    }
                },
                (false, false) => {
                    // どちらのDataFrameにも列が存在しない場合（通常はここには来ないはず）
                    continue;
                }
            };
            
            // 結合された列を新しいDataFrameに追加
            result.add_column(col_name, combined_column)?;
        }
        
        Ok(result)
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
        use rayon::prelude::*;
        use std::collections::hash_map::Entry;
        use std::sync::{Arc, Mutex};
        
        // データサイズに基づく最適化閾値
        const PARALLEL_THRESHOLD: usize = 50_000;
        
        // グループ化キーのカラムインデックスを取得
        let mut group_col_indices = Vec::with_capacity(group_by_columns.len());
        for &col_name in group_by_columns {
            let col_idx = self.column_indices.get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            group_col_indices.push(*col_idx);
        }
        
        // グループキーを生成し、各行のインデックスをグループ化
        let groups: HashMap<String, Vec<usize>> = if self.row_count < PARALLEL_THRESHOLD {
            // 小規模データでは直列処理の方が効率的
            let mut groups = HashMap::new();
            
            for row_idx in 0..self.row_count {
                // この行のグループキーを生成
                let mut key_parts = Vec::with_capacity(group_col_indices.len());
                
                for &col_idx in &group_col_indices {
                    let column = &self.columns[col_idx];
                    let part = match column {
                        Column::Int64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::Float64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::String(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::Boolean(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                    };
                    key_parts.push(part);
                }
                
                let group_key = key_parts.join("_");
                
                match groups.entry(group_key) {
                    Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                    Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                }
            }
            
            groups
        } else {
            // 大規模データでは並列処理+ロックフリーアプローチ
            // 1. 並列でローカルなグループマップを作成
            // 2. それらをマージ
            let chunk_size = (self.row_count / rayon::current_num_threads()).max(1000);
            
            // ステップ1: 並列でローカルな中間グループマップを作成
            let local_maps: Vec<HashMap<String, Vec<usize>>> = 
                (0..self.row_count).collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_groups = HashMap::new();
                    
                    for &row_idx in chunk {
                        // この行のグループキーを生成
                        let mut key_parts = Vec::with_capacity(group_col_indices.len());
                        
                        for &col_idx in &group_col_indices {
                            let column = &self.columns[col_idx];
                            let part = match column {
                                Column::Int64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::Float64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::String(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::Boolean(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                            };
                            key_parts.push(part);
                        }
                        
                        let group_key = key_parts.join("_");
                        
                        match local_groups.entry(group_key) {
                            Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                            Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                        }
                    }
                    
                    local_groups
                })
                .collect();
            
            // ステップ2: 中間マップをマージ
            let mut merged_groups = HashMap::new();
            for local_map in local_maps {
                for (key, indices) in local_map {
                    match merged_groups.entry(key) {
                        Entry::Vacant(e) => { e.insert(indices); },
                        Entry::Occupied(mut e) => { e.get_mut().extend(indices); }
                    }
                }
            }
            
            merged_groups
        };
        
        // 各グループに対してDataFrameを効率的に作成
        let result = if groups.len() < 100 || self.row_count < PARALLEL_THRESHOLD {
            // グループ数が少ない場合や小規模データでは直列処理
            let mut result = HashMap::with_capacity(groups.len());
            for (key, indices) in groups {
                let group_df = self.filter_by_indices(&indices)?;
                result.insert(key, group_df);
            }
            result
        } else {
            // 大規模データでのグループ処理は並列化
            // 各グループを並列処理し、スレッドセーフに結果を集約
            let result_mutex = Arc::new(Mutex::new(HashMap::with_capacity(groups.len())));
            
            // チャンクサイズを調整して、オーバーヘッドを最小化
            let chunk_size = (groups.len() / rayon::current_num_threads()).max(10);
            
            // グループのリストを作成し、チャンクに分割して並列処理
            let group_items: Vec<(String, Vec<usize>)> = groups.into_iter().collect();
            
            group_items.par_chunks(chunk_size)
                .for_each(|chunk| {
                    // 各チャンクの処理結果を一時保存
                    let mut local_results = HashMap::new();
                    
                    for (key, indices) in chunk {
                        if let Ok(group_df) = self.filter_by_indices(indices) {
                            local_results.insert(key.clone(), group_df);
                        }
                    }
                    
                    // 結果をメインのHashMapにマージ
                    if let Ok(mut result_map) = result_mutex.lock() {
                        for (key, df) in local_results {
                            result_map.insert(key, df);
                        }
                    }
                });
            
            // 最終結果を取得
            match Arc::try_unwrap(result_mutex) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(_) => HashMap::new(), // アークの解除に失敗した場合
            }
        };
        
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
        self.join_impl(other, left_on, right_on, JoinType::Inner)
    }
    
    /// 左結合
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Left)
    }
    
    /// 右結合
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Right)
    }
    
    /// 外部結合
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Outer)
    }
    
    // 結合の実装（内部メソッド）
    fn join_impl(&self, other: &Self, left_on: &str, right_on: &str, join_type: JoinType) -> Result<Self> {
        // 結合キー列の取得
        let left_col_idx = self.column_indices.get(left_on)
            .ok_or_else(|| Error::ColumnNotFound(left_on.to_string()))?;
        
        let right_col_idx = other.column_indices.get(right_on)
            .ok_or_else(|| Error::ColumnNotFound(right_on.to_string()))?;
        
        let left_col = &self.columns[*left_col_idx];
        let right_col = &other.columns[*right_col_idx];
        
        // 両方の列が同じ型であることを確認
        if left_col.column_type() != right_col.column_type() {
            return Err(Error::ColumnTypeMismatch {
                name: format!("{} と {}", left_on, right_on),
                expected: left_col.column_type(),
                found: right_col.column_type(),
            });
        }
        
        // 結合キーのマッピングを構築
        let mut right_key_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        
        match right_col {
            Column::Int64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
            Column::Float64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
            Column::String(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
            Column::Boolean(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
        }
        
        // 結合行の作成
        let mut result = Self::new();
        let mut join_indices: Vec<(Option<usize>, Option<usize>)> = Vec::new();
        
        // 左側のDataFrameを処理
        match left_col {
            Column::Int64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            // マッチがある場合、各右側インデックスと結合
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            // 左結合または外部結合では、マッチがなくても左側の行を含める
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
            Column::Float64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
            Column::String(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
            Column::Boolean(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
        }
        
        // 右結合または外部結合の場合、右側で未マッチの行を追加
        if join_type == JoinType::Right || join_type == JoinType::Outer {
            let mut right_matched = vec![false; other.row_count];
            for (_, right_idx) in &join_indices {
                if let Some(idx) = right_idx {
                    right_matched[*idx] = true;
                }
            }
            
            for (i, matched) in right_matched.iter().enumerate() {
                if !matched {
                    join_indices.push((None, Some(i)));
                }
            }
        }
        
        // 結果が空の場合、早期リターン
        if join_indices.is_empty() {
            // 空のデータフレームを返す（列だけ設定）
            let mut result = Self::new();
            for name in &self.column_names {
                if name != left_on {
                    let col_idx = self.column_indices[name];
                    let col_type = self.columns[col_idx].column_type();
                    
                    // 型に応じた空の列を作成
                    let empty_col = match col_type {
                        ColumnType::Int64 => Column::Int64(Int64Column::new(Vec::new())),
                        ColumnType::Float64 => Column::Float64(crate::column::Float64Column::new(Vec::new())),
                        ColumnType::String => Column::String(crate::column::StringColumn::new(Vec::new())),
                        ColumnType::Boolean => Column::Boolean(crate::column::BooleanColumn::new(Vec::new())),
                    };
                    
                    result.add_column(name.clone(), empty_col)?;
                }
            }
            
            for name in &other.column_names {
                if name != right_on {
                    let suffix = "_right";
                    let new_name = if self.column_indices.contains_key(name) {
                        format!("{}{}", name, suffix)
                    } else {
                        name.clone()
                    };
                    
                    let col_idx = other.column_indices[name];
                    let col_type = other.columns[col_idx].column_type();
                    
                    // 型に応じた空の列を作成
                    let empty_col = match col_type {
                        ColumnType::Int64 => Column::Int64(Int64Column::new(Vec::new())),
                        ColumnType::Float64 => Column::Float64(crate::column::Float64Column::new(Vec::new())),
                        ColumnType::String => Column::String(crate::column::StringColumn::new(Vec::new())),
                        ColumnType::Boolean => Column::Boolean(crate::column::BooleanColumn::new(Vec::new())),
                    };
                    
                    result.add_column(new_name, empty_col)?;
                }
            }
            
            return Ok(result);
        }
        
        // 結果用の各列データを準備
        let row_count = join_indices.len();
        
        // 左側の列を追加
        for name in &self.column_names {
            if name != left_on { // 結合キー列は1つだけ追加する
                let col_idx = self.column_indices[name];
                let col = &self.columns[col_idx];
                
                let joined_col = match col {
                    Column::Int64(int_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = int_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0); // デフォルト値
                                }
                            } else {
                                data.push(0); // デフォルト値（右側のみ）
                            }
                        }
                        Column::Int64(Int64Column::new(data))
                    },
                    Column::Float64(float_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = float_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0.0); // デフォルト値
                                }
                            } else {
                                data.push(0.0); // デフォルト値（右側のみ）
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(data))
                    },
                    Column::String(str_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = str_col.get(*idx) {
                                    data.push(val.to_string());
                                } else {
                                    data.push(String::new()); // デフォルト値
                                }
                            } else {
                                data.push(String::new()); // デフォルト値（右側のみ）
                            }
                        }
                        Column::String(crate::column::StringColumn::new(data))
                    },
                    Column::Boolean(bool_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = bool_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(false); // デフォルト値
                                }
                            } else {
                                data.push(false); // デフォルト値（右側のみ）
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(data))
                    },
                };
                
                result.add_column(name.clone(), joined_col)?;
            }
        }
        
        // 結合キー列を追加（左側から）
        let left_key_col = &self.columns[*left_col_idx];
        let joined_key_col = match left_key_col {
            Column::Int64(int_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = int_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(0); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::Int64(right_int_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_int_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(0); // デフォルト値
                            }
                        } else {
                            data.push(0); // デフォルト値
                        }
                    } else {
                        data.push(0); // デフォルト値
                    }
                }
                Column::Int64(Int64Column::new(data))
            },
            Column::Float64(float_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = float_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(0.0); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::Float64(right_float_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_float_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(0.0); // デフォルト値
                            }
                        } else {
                            data.push(0.0); // デフォルト値
                        }
                    } else {
                        data.push(0.0); // デフォルト値
                    }
                }
                Column::Float64(crate::column::Float64Column::new(data))
            },
            Column::String(str_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = str_col.get(*idx) {
                            data.push(val.to_string());
                        } else {
                            data.push(String::new()); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::String(right_str_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_str_col.get(*idx) {
                                data.push(val.to_string());
                            } else {
                                data.push(String::new()); // デフォルト値
                            }
                        } else {
                            data.push(String::new()); // デフォルト値
                        }
                    } else {
                        data.push(String::new()); // デフォルト値
                    }
                }
                Column::String(crate::column::StringColumn::new(data))
            },
            Column::Boolean(bool_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = bool_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(false); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::Boolean(right_bool_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_bool_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(false); // デフォルト値
                            }
                        } else {
                            data.push(false); // デフォルト値
                        }
                    } else {
                        data.push(false); // デフォルト値
                    }
                }
                Column::Boolean(crate::column::BooleanColumn::new(data))
            },
        };
        
        result.add_column(left_on.to_string(), joined_key_col)?;
        
        // 右側の列を追加（結合キー以外）
        for name in &other.column_names {
            if name != right_on {
                let suffix = "_right";
                let new_name = if result.column_indices.contains_key(name) {
                    format!("{}{}", name, suffix)
                } else {
                    name.clone()
                };
                
                let col_idx = other.column_indices[name];
                let col = &other.columns[col_idx];
                
                let joined_col = match col {
                    Column::Int64(int_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = int_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0); // デフォルト値
                                }
                            } else {
                                data.push(0); // デフォルト値（左側のみ）
                            }
                        }
                        Column::Int64(Int64Column::new(data))
                    },
                    Column::Float64(float_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = float_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0.0); // デフォルト値
                                }
                            } else {
                                data.push(0.0); // デフォルト値（左側のみ）
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(data))
                    },
                    Column::String(str_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = str_col.get(*idx) {
                                    data.push(val.to_string());
                                } else {
                                    data.push(String::new()); // デフォルト値
                                }
                            } else {
                                data.push(String::new()); // デフォルト値（左側のみ）
                            }
                        }
                        Column::String(crate::column::StringColumn::new(data))
                    },
                    Column::Boolean(bool_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = bool_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(false); // デフォルト値
                                }
                            } else {
                                data.push(false); // デフォルト値（左側のみ）
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(data))
                    },
                };
                
                result.add_column(new_name, joined_col)?;
            }
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
        // 引数のデフォルト値を設定
        let var_name = var_name.unwrap_or("variable");
        let value_name = value_name.unwrap_or("value");
        
        // value_varsが指定されていない場合は、id_vars以外のすべての列を使用
        let value_vars = if let Some(vars) = value_vars {
            vars.to_vec()
        } else {
            self.column_names
                .iter()
                .filter(|name| !id_vars.contains(&name.as_str()))
                .map(|s| s.as_str())
                .collect()
        };
        
        // 存在しない列名をチェック
        for col in id_vars.iter().chain(value_vars.iter()) {
            if !self.column_indices.contains_key(*col) {
                return Err(Error::ColumnNotFound((*col).to_string()));
            }
        }
        
        // 結果のサイズを事前計算（パフォーマンス最適化）
        let result_rows = self.row_count * value_vars.len();
        
        // ID列のデータを抽出
        let mut id_columns = Vec::with_capacity(id_vars.len());
        for &id_col in id_vars {
            let idx = self.column_indices[id_col];
            id_columns.push((id_col, &self.columns[idx]));
        }
        
        // 値列のデータを抽出
        let mut value_columns = Vec::with_capacity(value_vars.len());
        for &val_col in &value_vars {
            let idx = self.column_indices[val_col];
            value_columns.push((val_col, &self.columns[idx]));
        }
        
        // 結果のDataFrameを生成
        let mut result = Self::new();
        
        // 変数名の列を作成
        let mut var_col_data = Vec::with_capacity(result_rows);
        for &value_col_name in &value_vars {
            for _ in 0..self.row_count {
                var_col_data.push(value_col_name.to_string());
            }
        }
        result.add_column(var_name.to_string(), Column::String(StringColumn::new(var_col_data)))?;
        
        // ID列をレプリケートして追加
        for &(id_col_name, col) in &id_columns {
            match col {
                Column::Int64(int_col) => {
                    // 整数型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = int_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // NULL値の場合はデフォルト値を使用
                                repeated_data.push(0);
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::Int64(Int64Column::new(repeated_data)))?;
                },
                Column::Float64(float_col) => {
                    // 浮動小数点型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = float_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // NULL値の場合はデフォルト値を使用
                                repeated_data.push(0.0);
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::Float64(Float64Column::new(repeated_data)))?;
                },
                Column::String(str_col) => {
                    // 文字列型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = str_col.get(i) {
                                repeated_data.push(val.to_string());
                            } else {
                                // NULL値の場合は空文字列を使用
                                repeated_data.push(String::new());
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::String(StringColumn::new(repeated_data)))?;
                },
                Column::Boolean(bool_col) => {
                    // ブール型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = bool_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // NULL値の場合はデフォルト値を使用
                                repeated_data.push(false);
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::Boolean(BooleanColumn::new(repeated_data)))?;
                },
            }
        }
        
        // 値列を作成（型に応じた最適な方法で）
        // 最適化のため、最終的な型を推測してからデータを追加する
        let mut all_values = Vec::with_capacity(result_rows);
        
        // まず全ての値を文字列として収集
        for (_, col) in value_columns {
            match col {
                Column::Int64(int_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = int_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
                Column::Float64(float_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = float_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
                Column::String(str_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = str_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
                Column::Boolean(bool_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = bool_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
            }
        }
        
        // 適切な型を推測してデータ追加（すべて数値なら数値型、等）
        let is_all_int = all_values.iter()
            .all(|s| s.parse::<i64>().is_ok() || s.is_empty());
        
        let is_all_float = !is_all_int && all_values.iter()
            .all(|s| s.parse::<f64>().is_ok() || s.is_empty());
        
        let is_all_bool = !is_all_int && !is_all_float && all_values.iter()
            .all(|s| {
                let lower = s.to_lowercase();
                lower.is_empty() || lower == "true" || lower == "false" || 
                lower == "1" || lower == "0" || lower == "yes" || lower == "no"
            });
        
        // 型に合わせて列を追加
        if is_all_int {
            let int_values: Vec<i64> = all_values.iter()
                .map(|s| s.parse::<i64>().unwrap_or(0))
                .collect();
            result.add_column(value_name.to_string(), Column::Int64(Int64Column::new(int_values)))?;
        } else if is_all_float {
            let float_values: Vec<f64> = all_values.iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            result.add_column(value_name.to_string(), Column::Float64(Float64Column::new(float_values)))?;
        } else if is_all_bool {
            let bool_values: Vec<bool> = all_values.iter()
                .map(|s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "1" || lower == "yes"
                })
                .collect();
            result.add_column(value_name.to_string(), Column::Boolean(BooleanColumn::new(bool_values)))?;
        } else {
            // デフォルトは文字列型
            result.add_column(value_name.to_string(), Column::String(StringColumn::new(all_values)))?;
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
        let df = crate::io::read_excel(path, sheet_name, header, skip_rows, use_cols)?;
        Self::from_standard_dataframe(&df)
    }

    /// データフレームをExcelファイルに書き込む
    pub fn to_excel<P: AsRef<Path>>(
        &self,
        path: P,
        sheet_name: Option<&str>,
        index: bool,
    ) -> Result<()> {
        // 新しいExcelファイルを作成
        let mut workbook = simple_excel_writer::Workbook::create(path.as_ref()
            .to_str()
            .ok_or_else(|| Error::IoError("ファイルパスを文字列に変換できませんでした".to_string()))?);
        
        let sheet_name = sheet_name.unwrap_or("Sheet1");
        
        // シートを作成
        let mut sheet = workbook.create_sheet(sheet_name);
        
        // ヘッダー行を作成
        let mut headers = Vec::new();
        
        // インデックスを含める場合
        if index {
            headers.push("Index".to_string());
        }
        
        // 列名を追加
        for col_name in &self.column_names {
            headers.push(col_name.clone());
        }
        
        // データを書き込む
        workbook.write_sheet(&mut sheet, |sheet_writer| {
            // ヘッダー行を追加
            if !headers.is_empty() {
                let header_row: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
                // Rowを直接作成
                let row = simple_excel_writer::Row::from_iter(header_row.iter().cloned());
                sheet_writer.append_row(row)?;
            }
            
            // データ行を書き込む
            for row_idx in 0..self.row_count {
                let mut row_values = Vec::new();
                
                // インデックスを含める場合
                if index {
                    row_values.push(row_idx.to_string());
                }
                
                // 各列のデータを追加
                for col_idx in 0..self.columns.len() {
                    let col = &self.columns[col_idx];
                    let value = match col {
                        Column::Int64(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        },
                        Column::Float64(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        },
                        Column::String(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        },
                        Column::Boolean(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        },
                    };
                    
                    row_values.push(value);
                }
                
                // 行をExcelに追加（文字列参照のスライスに変換）
                let row_str_refs: Vec<&str> = row_values.iter().map(|s| s.as_str()).collect();
                // Rowを直接作成
                let row = simple_excel_writer::Row::from_iter(row_str_refs.iter().cloned());
                sheet_writer.append_row(row)?;
            }
            
            Ok(())
        })?;
        
        // ワークブックを閉じて保存
        workbook.close()
            .map_err(|e| Error::IoError(format!("Excelファイルを保存できませんでした: {}", e)))?;
        
        Ok(())
    }

    /// データフレームをParquetファイルに書き込む
    pub fn to_parquet<P: AsRef<Path>>(
        &self,
        path: P,
        compression: Option<ParquetCompression>,
    ) -> Result<()> {
        // まず標準のDataFrameに変換
        let df = self.to_standard_dataframe()?;
        // 圧縮設定を変換
        let io_compression = compression.map(|c| match c {
            ParquetCompression::None => crate::io::parquet::ParquetCompression::None,
            ParquetCompression::Snappy => crate::io::parquet::ParquetCompression::Snappy,
            ParquetCompression::Gzip => crate::io::parquet::ParquetCompression::Gzip,
            ParquetCompression::Lzo => crate::io::parquet::ParquetCompression::Lzo,
            ParquetCompression::Brotli => crate::io::parquet::ParquetCompression::Brotli,
            ParquetCompression::Lz4 => crate::io::parquet::ParquetCompression::Lz4,
            ParquetCompression::Zstd => crate::io::parquet::ParquetCompression::Zstd,
        });
        // 標準APIを使用して書き込む
        // 互換性のためにself参照を使う
        crate::io::write_parquet(self, path, io_compression)
    }

    /// Parquetファイルからデータフレームを読み込む
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        let df = crate::io::read_parquet(path)?;
        Self::from_standard_dataframe(&df)
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