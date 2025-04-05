//! OptimizedDataFrameの入出力関連機能

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use csv::{ReaderBuilder, Writer};
use serde_json::{Map, Value};
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::core::OptimizedDataFrame;
use crate::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};

/// JSON出力形式
pub enum JsonOrient {
    /// レコード形式 [{col1:val1, col2:val2}, ...]
    Records,
    /// 列形式 {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

/// Parquet圧縮オプションの列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetCompression {
    None,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
}

impl From<ParquetCompression> for Compression {
    fn from(comp: ParquetCompression) -> Self {
        match comp {
            ParquetCompression::None => Compression::UNCOMPRESSED,
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Gzip => Compression::GZIP(Default::default()),
            ParquetCompression::Lzo => Compression::LZO,
            ParquetCompression::Brotli => Compression::BROTLI(Default::default()),
            ParquetCompression::Lz4 => Compression::LZ4,
            ParquetCompression::Zstd => Compression::ZSTD(Default::default()),
        }
    }
}

impl OptimizedDataFrame {
    /// CSVファイルからDataFrameを読み込む
    ///
    /// # Arguments
    /// * `path` - CSVファイルのパス
    /// * `has_header` - ヘッダー行があるかどうか
    ///
    /// # Returns
    /// * `Result<Self>` - 読み込んだDataFrame
    pub fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::IO(e.to_string()))?;

        // CSVリーダーを設定
        let mut rdr = ReaderBuilder::new()
            .has_headers(has_header)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_reader(file);

        let mut df = Self::new();

        // ヘッダー行を取得
        let headers: Vec<String> = if has_header {
            rdr.headers()
                .map_err(|e| Error::CSV(e.to_string()))?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // ヘッダーがない場合は、列名を生成
            if let Some(first_record_result) = rdr.records().next() {
                let first_record = first_record_result.map_err(|e| Error::CSV(e.to_string()))?;
                (0..first_record.len())
                    .map(|i| format!("column_{}", i))
                    .collect()
            } else {
                // ファイルが空の場合
                return Ok(Self::new());
            }
        };

        // 列データの収集用バッファ
        let mut str_buffers: Vec<Vec<String>> = headers.iter().map(|_| Vec::new()).collect();

        // すべての行を読み込み
        for result in rdr.records() {
            let record = result.map_err(|e| Error::CSV(e.to_string()))?;
            for (i, field) in record.iter().enumerate() {
                if i < str_buffers.len() {
                    str_buffers[i].push(field.to_string());
                }
            }
            // 不足分をNULLとして追加
            for buffer in &mut str_buffers {
                if buffer.len() < str_buffers[0].len() {
                    buffer.push(String::new());
                }
            }
        }

        // 文字列データを型推論して列を追加
        for (i, header) in headers.into_iter().enumerate() {
            if i < str_buffers.len() {
                // 型推論を行う
                let values = &str_buffers[i];
                
                // 空でない値のチェック
                let non_empty_values: Vec<&String> = values.iter().filter(|s| !s.is_empty()).collect();
                
                if non_empty_values.is_empty() {
                    // すべて空の場合は文字列型
                    df.add_column(header, Column::String(StringColumn::new(
                        values.iter().map(|s| s.clone()).collect()
                    )))?;
                    continue;
                }
                
                // 整数型として解析を試みる
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = values.iter()
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                        .collect();
                    df.add_column(header, Column::Int64(Int64Column::new(int_values)))?;
                    continue;
                }
                
                // 浮動小数点型として解析を試みる
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = values.iter()
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                        .collect();
                    df.add_column(header, Column::Float64(Float64Column::new(float_values)))?;
                    continue;
                }
                
                // ブール型として解析を試みる
                let all_bools = non_empty_values.iter().all(|&s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "false" || lower == "1" || lower == "0" || 
                    lower == "yes" || lower == "no" || lower == "t" || lower == "f"
                });
                
                if all_bools {
                    let bool_values: Vec<bool> = values.iter()
                        .map(|s| {
                            let lower = s.to_lowercase();
                            lower == "true" || lower == "1" || lower == "yes" || lower == "t"
                        })
                        .collect();
                    df.add_column(header, Column::Boolean(BooleanColumn::new(bool_values)))?;
                } else {
                    // デフォルトは文字列型
                    df.add_column(header, Column::String(StringColumn::new(
                        values.iter().map(|s| s.clone()).collect()
                    )))?;
                }
            }
        }

        Ok(df)
    }

    /// DataFrameをCSVファイルに書き込む
    ///
    /// # Arguments
    /// * `path` - 書き込み先のCSVファイルパス
    /// * `has_header` - ヘッダー行を書き込むかどうか
    ///
    /// # Returns
    /// * `Result<()>` - 成功時はOk
    pub fn to_csv<P: AsRef<Path>>(&self, path: P, has_header: bool) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| Error::IO(e.to_string()))?;
        let mut wtr = Writer::from_writer(file);

        // ヘッダー行を書き込む
        if has_header {
            wtr.write_record(&self.column_names)
                .map_err(|e| Error::CSV(e.to_string()))?;
        }

        // 行がない場合は何もせず終了
        if self.row_count == 0 {
            wtr.flush().map_err(|e| Error::IO(e.to_string()))?;
            return Ok(());
        }

        // 各行を書き込む
        for i in 0..self.row_count {
            let mut row = Vec::new();
            
            for col_idx in 0..self.columns.len() {
                let value = match &self.columns[col_idx] {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    },
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    },
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    },
                };
                
                row.push(value);
            }
            
            wtr.write_record(&row).map_err(|e| Error::CSV(e.to_string()))?;
        }

        wtr.flush().map_err(|e| Error::IO(e.to_string()))?;
        Ok(())
    }

    /// JSONファイルからDataFrameを読み込む
    ///
    /// # Arguments
    /// * `path` - JSONファイルのパス
    ///
    /// # Returns
    /// * `Result<Self>` - 読み込んだDataFrame
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::IO(e.to_string()))?;
        let reader = BufReader::new(file);

        // JSONを解析
        let json_value: Value = serde_json::from_reader(reader).map_err(|e| Error::Json(e.to_string()))?;

        match json_value {
            Value::Array(array) => Self::from_records_array(array),
            Value::Object(map) => Self::from_column_oriented(map),
            _ => Err(Error::Format(
                "JSONはオブジェクトまたは配列である必要があります".to_string(),
            )),
        }
    }

    // レコード指向JSONから読み込む
    fn from_records_array(array: Vec<Value>) -> Result<Self> {
        let mut df = Self::new();

        // 空配列の場合は空のDataFrameを返す
        if array.is_empty() {
            return Ok(df);
        }

        // 全てのキーを収集
        let mut all_keys = std::collections::HashSet::new();
        for item in &array {
            if let Value::Object(map) = item {
                for key in map.keys() {
                    all_keys.insert(key.clone());
                }
            } else {
                return Err(Error::Format(
                    "配列の各要素はオブジェクトである必要があります".to_string(),
                ));
            }
        }

        // 列データを収集
        let mut string_columns: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for key in &all_keys {
            let mut values = Vec::with_capacity(array.len());

            for item in &array {
                if let Value::Object(map) = item {
                    if let Some(value) = map.get(key) {
                        values.push(value.to_string());
                    } else {
                        // キーが存在しない場合は空文字列
                        values.push(String::new());
                    }
                }
            }

            string_columns.insert(key.clone(), values);
        }

        // 列をDataFrameに追加し、適切な型に変換
        for (key, values) in string_columns {
            // 空でない値のチェック
            let non_empty_values: Vec<&String> = values.iter().filter(|s| !s.is_empty()).collect();
            
            if non_empty_values.is_empty() {
                // すべて空の場合は文字列型
                df.add_column(key, Column::String(StringColumn::new(
                    values.iter().map(|s| s.clone()).collect()
                )))?;
                continue;
            }
            
            // 整数型として解析を試みる
            let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
            if all_ints {
                let int_values: Vec<i64> = values.iter()
                    .map(|s| s.parse::<i64>().unwrap_or(0))
                    .collect();
                df.add_column(key, Column::Int64(Int64Column::new(int_values)))?;
                continue;
            }
            
            // 浮動小数点型として解析を試みる
            let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = values.iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                df.add_column(key, Column::Float64(Float64Column::new(float_values)))?;
                continue;
            }
            
            // ブール型として解析を試みる
            let all_bools = non_empty_values.iter().all(|&s| {
                let lower = s.to_lowercase();
                lower == "true" || lower == "false" || lower == "1" || lower == "0" || 
                lower == "yes" || lower == "no" || lower == "t" || lower == "f"
            });
            
            if all_bools {
                let bool_values: Vec<bool> = values.iter()
                    .map(|s| {
                        let lower = s.to_lowercase();
                        lower == "true" || lower == "1" || lower == "yes" || lower == "t"
                    })
                    .collect();
                df.add_column(key, Column::Boolean(BooleanColumn::new(bool_values)))?;
            } else {
                // デフォルトは文字列型
                df.add_column(key, Column::String(StringColumn::new(
                    values.iter().map(|s| s.clone()).collect()
                )))?;
            }
        }

        Ok(df)
    }

    // 列指向JSONから読み込む
    fn from_column_oriented(map: Map<String, Value>) -> Result<Self> {
        let mut df = Self::new();

        // 各列を処理
        for (key, value) in map {
            if let Value::Array(array) = value {
                let string_values: Vec<String> = array.iter().map(|v| v.to_string()).collect();
                
                // 空でない値のチェック
                let non_empty_values: Vec<&String> = string_values.iter().filter(|s| !s.is_empty()).collect();
                
                if non_empty_values.is_empty() {
                    // すべて空の場合は文字列型
                    df.add_column(key, Column::String(StringColumn::new(
                        string_values.iter().map(|s| s.clone()).collect()
                    )))?;
                    continue;
                }
                
                // 整数型として解析を試みる
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = string_values.iter()
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                        .collect();
                    df.add_column(key, Column::Int64(Int64Column::new(int_values)))?;
                    continue;
                }
                
                // 浮動小数点型として解析を試みる
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = string_values.iter()
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                        .collect();
                    df.add_column(key, Column::Float64(Float64Column::new(float_values)))?;
                    continue;
                }
                
                // ブール型として解析を試みる
                let all_bools = non_empty_values.iter().all(|&s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "false" || lower == "1" || lower == "0" || 
                    lower == "yes" || lower == "no" || lower == "t" || lower == "f"
                });
                
                if all_bools {
                    let bool_values: Vec<bool> = string_values.iter()
                        .map(|s| {
                            let lower = s.to_lowercase();
                            lower == "true" || lower == "1" || lower == "yes" || lower == "t"
                        })
                        .collect();
                    df.add_column(key, Column::Boolean(BooleanColumn::new(bool_values)))?;
                } else {
                    // デフォルトは文字列型
                    df.add_column(key, Column::String(StringColumn::new(
                        string_values.iter().map(|s| s.clone()).collect()
                    )))?;
                }
            } else {
                return Err(Error::Format(format!(
                    "列 '{}' は配列である必要があります",
                    key
                )));
            }
        }

        Ok(df)
    }

    /// DataFrameをJSONファイルに書き込む
    ///
    /// # Arguments
    /// * `path` - 書き込み先のJSONファイルパス
    /// * `orient` - JSON出力形式（Records または Columns）
    ///
    /// # Returns
    /// * `Result<()>` - 成功時はOk
    pub fn to_json<P: AsRef<Path>>(&self, path: P, orient: JsonOrient) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| Error::IO(e.to_string()))?;
        let writer = BufWriter::new(file);

        let json_value = match orient {
            JsonOrient::Records => self.to_records_json()?,
            JsonOrient::Columns => self.to_column_json()?,
        };

        serde_json::to_writer_pretty(writer, &json_value).map_err(|e| Error::Json(e.to_string()))?;

        Ok(())
    }

    // DataFrameをレコード指向JSONに変換
    fn to_records_json(&self) -> Result<Value> {
        let mut records = Vec::new();

        // 行がない場合は空の配列を返す
        if self.row_count == 0 {
            return Ok(Value::Array(records));
        }

        // 各行のデータを処理
        for row_idx in 0..self.row_count {
            let mut record = serde_json::Map::new();

            // 各列の値を取得して追加
            for (col_idx, col_name) in self.column_names.iter().enumerate() {
                let value = match &self.columns[col_idx] {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Number(serde_json::Number::from(val))
                        } else {
                            Value::Null
                        }
                    },
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            // f64をNumberに変換（NaNやInfinityは処理できないのでその場合はNull）
                            if val.is_finite() {
                                serde_json::Number::from_f64(val)
                                    .map_or(Value::Null, Value::Number)
                            } else {
                                Value::Null
                            }
                        } else {
                            Value::Null
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::String(val.to_string())
                        } else {
                            Value::Null
                        }
                    },
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Bool(val)
                        } else {
                            Value::Null
                        }
                    },
                };
                
                record.insert(col_name.clone(), value);
            }
            
            records.push(Value::Object(record));
        }

        Ok(Value::Array(records))
    }

    // DataFrameを列指向JSONに変換
    fn to_column_json(&self) -> Result<Value> {
        let mut columns = serde_json::Map::new();

        // 行がない場合は空のオブジェクトを返す
        if self.row_count == 0 {
            return Ok(Value::Object(columns));
        }

        // 各列を処理
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let mut values = Vec::new();

            // 列の全ての値を取得
            for row_idx in 0..self.row_count {
                let value = match &self.columns[col_idx] {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Number(serde_json::Number::from(val))
                        } else {
                            Value::Null
                        }
                    },
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            // f64をNumberに変換（NaNやInfinityは処理できないのでその場合はNull）
                            if val.is_finite() {
                                serde_json::Number::from_f64(val)
                                    .map_or(Value::Null, Value::Number)
                            } else {
                                Value::Null
                            }
                        } else {
                            Value::Null
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::String(val.to_string())
                        } else {
                            Value::Null
                        }
                    },
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Bool(val)
                        } else {
                            Value::Null
                        }
                    },
                };
                
                values.push(value);
            }
            
            columns.insert(col_name.clone(), Value::Array(values));
        }

        Ok(Value::Object(columns))
    }

    /// DataFrameをParquetファイルに書き込む
    ///
    /// # Arguments
    /// * `path` - 書き込み先のParquetファイルパス
    /// * `compression` - 圧縮方式（オプション、Noneの場合はZSTDが使用される）
    ///
    /// # Returns
    /// * `Result<()>` - 成功時はOk
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P, compression: Option<ParquetCompression>) -> Result<()> {
        // 現在は実装されていないので、未実装エラーを返す
        Err(Error::NotImplemented(
            "Parquet書き込み機能は現在実装中です。将来のバージョンで利用可能になります。".to_string()
        ))
    }

    /// ParquetファイルからDataFrameを読み込む
    ///
    /// # Arguments
    /// * `path` - Parquetファイルのパス
    ///
    /// # Returns
    /// * `Result<Self>` - 読み込んだDataFrame
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        // 現在は実装されていないので、未実装エラーを返す
        Err(Error::NotImplemented(
            "Parquet読み込み機能は現在実装中です。将来のバージョンで利用可能になります。".to_string()
        ))
    }
}
