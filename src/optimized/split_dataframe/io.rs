//! OptimizedDataFrameの入出力関連機能

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use calamine::{open_workbook, Reader, Xlsx};
use csv::{ReaderBuilder, Writer};
use rusqlite::{Connection, params};
use serde_json::{Map, Value};
use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use simple_excel_writer::{Workbook, Sheet};

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
        let file = File::open(path.as_ref()).map_err(|e| Error::Io(e))?;

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
                .map_err(|e| Error::Csv(e))?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // ヘッダーがない場合は、列名を生成
            if let Some(first_record_result) = rdr.records().next() {
                let first_record = first_record_result.map_err(|e| Error::Csv(e))?;
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
            let record = result.map_err(|e| Error::Csv(e))?;
            for (i, field) in record.iter().enumerate() {
                if i < str_buffers.len() {
                    str_buffers[i].push(field.to_string());
                }
            }
            // 不足分をNULLとして追加
            let max_len = str_buffers.get(0).map_or(0, |b| b.len());
            for buffer in &mut str_buffers {
                if buffer.len() < max_len {
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
        let file = File::create(path.as_ref()).map_err(|e| Error::Io(e))?;
        let mut wtr = Writer::from_writer(file);

        // ヘッダー行を書き込む
        if has_header {
            wtr.write_record(&self.column_names)
                .map_err(|e| Error::Csv(e))?;
        }

        // 行がない場合は何もせず終了
        if self.row_count == 0 {
            wtr.flush().map_err(|e| Error::Io(e))?;
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
            
            wtr.write_record(&row).map_err(|e| Error::Csv(e))?;
        }

        wtr.flush().map_err(|e| Error::Io(e))?;
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
        let file = File::open(path.as_ref()).map_err(|e| Error::Io(e))?;
        let reader = BufReader::new(file);

        // JSONを解析
        let json_value: Value = serde_json::from_reader(reader).map_err(|e| Error::Json(e))?;

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
        let file = File::create(path.as_ref()).map_err(|e| Error::Io(e))?;
        let writer = BufWriter::new(file);

        let json_value = match orient {
            JsonOrient::Records => self.to_records_json()?,
            JsonOrient::Columns => self.to_column_json()?,
        };

        serde_json::to_writer_pretty(writer, &json_value).map_err(|e| Error::Json(e))?;

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
    /// * `compression` - 圧縮方式（オプション、Noneの場合はSnappyが使用される）
    ///
    /// # Returns
    /// * `Result<()>` - 成功時はOk
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P, compression: Option<ParquetCompression>) -> Result<()> {
        // 行がない場合でも空のデータフレームとして書き込む
        
        // Arrowスキーマを作成
        let schema_fields: Vec<Field> = self.column_names.iter()
            .enumerate()
            .map(|(idx, col_name)| {
                match &self.columns[idx] {
                    Column::Int64(_) => Field::new(col_name, DataType::Int64, true),
                    Column::Float64(_) => Field::new(col_name, DataType::Float64, true),
                    Column::Boolean(_) => Field::new(col_name, DataType::Boolean, true),
                    Column::String(_) => Field::new(col_name, DataType::Utf8, true),
                }
            })
            .collect();
        
        let schema = Schema::new(schema_fields);
        let schema_ref = Arc::new(schema);
        
        // 列データをArrow配列に変換
        let arrays: Vec<ArrayRef> = self.column_names.iter()
            .enumerate()
            .map(|(idx, _)| {
                match &self.columns[idx] {
                    Column::Int64(col) => {
                        let values: Vec<i64> = (0..self.row_count)
                            .map(|i| match col.get(i) {
                                Ok(Some(v)) => v,
                                _ => 0
                            })
                            .collect();
                        Arc::new(Int64Array::from(values)) as ArrayRef
                    },
                    Column::Float64(col) => {
                        let values: Vec<f64> = (0..self.row_count)
                            .map(|i| match col.get(i) {
                                Ok(Some(v)) => v,
                                _ => f64::NAN
                            })
                            .collect();
                        Arc::new(Float64Array::from(values)) as ArrayRef
                    },
                    Column::Boolean(col) => {
                        let values: Vec<bool> = (0..self.row_count)
                            .map(|i| match col.get(i) {
                                Ok(Some(v)) => v,
                                _ => false
                            })
                            .collect();
                        Arc::new(BooleanArray::from(values)) as ArrayRef
                    },
                    Column::String(col) => {
                        let values: Vec<String> = (0..self.row_count)
                            .map(|i| {
                                if let Ok(Some(v)) = col.get(i) {
                                    v.to_string()
                                } else {
                                    String::new()
                                }
                            })
                            .collect();
                        Arc::new(StringArray::from(values)) as ArrayRef
                    },
                }
            })
            .collect();
        
        // レコードバッチを作成
        let batch = RecordBatch::try_new(schema_ref.clone(), arrays)
            .map_err(|e| Error::Cast(format!("レコードバッチの作成に失敗しました: {}", e)))?;
        
        // 圧縮オプションを設定
        let compression_type = compression.unwrap_or(ParquetCompression::Snappy);
        let props = WriterProperties::builder()
            .set_compression(Compression::from(compression_type))
            .build();
        
        // ファイルを作成
        let file = File::create(path.as_ref())
            .map_err(|e| Error::Io(crate::error::io_error(format!("Parquetファイルを作成できませんでした: {}", e))))?;
        
        // Arrowライターを作成して書き込み
        let mut writer = ArrowWriter::try_new(file, schema_ref, Some(props))
            .map_err(|e| Error::Io(crate::error::io_error(format!("Parquetライターの作成に失敗しました: {}", e))))?;
        
        // レコードバッチを書き込む
        writer.write(&batch)
            .map_err(|e| Error::Io(crate::error::io_error(format!("レコードバッチの書き込みに失敗しました: {}", e))))?;
        
        // ファイルを閉じる
        writer.close()
            .map_err(|e| Error::Io(crate::error::io_error(format!("Parquetファイルの閉じる操作に失敗しました: {}", e))))?;
        
        Ok(())
    }

    /// ParquetファイルからDataFrameを読み込む
    ///
    /// # Arguments
    /// * `path` - Parquetファイルのパス
    ///
    /// # Returns
    /// * `Result<Self>` - 読み込んだDataFrame
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        // ファイルを開く
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Io(crate::error::io_error(format!("Parquetファイルを開けませんでした: {}", e))))?;
        
        // ArrowのParquetReaderを作成
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::Io(crate::error::io_error(format!("Parquetファイルの解析に失敗しました: {}", e))))?;
        
        // スキーマ情報を取得（クローンしておく）
        let schema = builder.schema().clone();
        
        // レコードバッチリーダーを作成
        let mut reader = builder.build()
            .map_err(|e| Error::Io(crate::error::io_error(format!("Parquetファイルの読み込みに失敗しました: {}", e))))?;
        
        // 全てのレコードバッチを読み込む
        let mut all_batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| Error::Io(crate::error::io_error(format!("レコードバッチの読み込みに失敗しました: {}", e))))?;
            all_batches.push(batch);
        }
        
        // レコードバッチがない場合は空のデータフレームを返す
        if all_batches.is_empty() {
            return Ok(Self::new());
        }
        
        // データフレームに変換
        let mut df = Self::new();
        
        // スキーマから列情報を取得
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col_name = field.name().clone();
            let col_type = field.data_type();
            
            // 全てのバッチから列データを収集
            match col_type {
                DataType::Int64 => {
                    let mut values = Vec::new();
                    
                    for batch in &all_batches {
                        let array = batch.column(col_idx).as_any().downcast_ref::<Int64Array>()
                            .ok_or_else(|| Error::Cast(format!("列 '{}' をInt64Arrayに変換できませんでした", col_name)))?;
                        
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(0);  // NULLの場合はデフォルト値として0を使用
                            } else {
                                values.push(array.value(i));
                            }
                        }
                    }
                    
                    df.add_column(col_name, Column::Int64(Int64Column::new(values)))?;
                },
                DataType::Float64 => {
                    let mut values = Vec::new();
                    
                    for batch in &all_batches {
                        let array = batch.column(col_idx).as_any().downcast_ref::<Float64Array>()
                            .ok_or_else(|| Error::Cast(format!("列 '{}' をFloat64Arrayに変換できませんでした", col_name)))?;
                        
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(f64::NAN);  // NULLの場合はNaNを使用
                            } else {
                                values.push(array.value(i));
                            }
                        }
                    }
                    
                    df.add_column(col_name, Column::Float64(Float64Column::new(values)))?;
                },
                DataType::Boolean => {
                    let mut values = Vec::new();
                    
                    for batch in &all_batches {
                        let array = batch.column(col_idx).as_any().downcast_ref::<BooleanArray>()
                            .ok_or_else(|| Error::Cast(format!("列 '{}' をBooleanArrayに変換できませんでした", col_name)))?;
                        
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(false);  // NULLの場合はデフォルト値としてfalseを使用
                            } else {
                                values.push(array.value(i));
                            }
                        }
                    }
                    
                    df.add_column(col_name, Column::Boolean(BooleanColumn::new(values)))?;
                },
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let mut values = Vec::new();
                    
                    for batch in &all_batches {
                        let array = batch.column(col_idx).as_any().downcast_ref::<StringArray>()
                            .ok_or_else(|| Error::Cast(format!("列 '{}' をStringArrayに変換できませんでした", col_name)))?;
                        
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push("".to_string());  // NULLの場合は空文字列を使用
                            } else {
                                values.push(array.value(i).to_string());
                            }
                        }
                    }
                    
                    df.add_column(col_name, Column::String(StringColumn::new(values)))?;
                },
                _ => {
                    // サポートされていないデータ型は文字列として扱う
                    let mut values = Vec::new();
                    
                    for batch in &all_batches {
                        let array = batch.column(col_idx);
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push("".to_string());
                            } else {
                                // ArrayのvalueメソッドにアクセスできないのでStringArrayにダウンキャストしてからvalueを呼び出す
                                if let Some(str_array) = array.as_any().downcast_ref::<StringArray>() {
                                    values.push(str_array.value(i).to_string());
                                } else {
                                    values.push(format!("{:?}", array));
                                }
                            }
                        }
                    }
                    
                    df.add_column(col_name, Column::String(StringColumn::new(values)))?;
                },
            }
        }
        
        Ok(df)
    }
    
    /// Excelファイル (.xlsx) からDataFrameを読み込む
    ///
    /// # Arguments
    /// * `path` - Excelファイルのパス
    /// * `sheet_name` - 読み込むシート名 (Noneの場合、最初のシートを読み込む)
    /// * `header` - ヘッダー行があるかどうか
    /// * `skip_rows` - 読み込み開始前にスキップする行数
    /// * `use_cols` - 読み込む列名または列番号のリスト (Noneの場合、すべての列を読み込む)
    ///
    /// # Returns
    /// * `Result<Self>` - 読み込んだDataFrame
    pub fn from_excel<P: AsRef<Path>>(
        path: P, 
        sheet_name: Option<&str>,
        header: bool,
        skip_rows: usize,
        use_cols: Option<&[&str]>,
    ) -> Result<Self> {
        // ファイルを開く
        let mut workbook: Xlsx<BufReader<File>> = open_workbook(path.as_ref())
            .map_err(|e| Error::Io(crate::error::io_error(format!("Excelファイルを開けませんでした: {}", e))))?;
        
        // シート名を取得（指定がなければ最初のシート）
        let sheet_name = match sheet_name {
            Some(name) => name.to_string(),
            None => workbook.sheet_names().get(0)
                .ok_or_else(|| Error::Io(crate::error::io_error("Excelファイルにシートがありません")))?
                .clone(),
        };
        
        // シートを取得
        let range = workbook.worksheet_range(&sheet_name)
            .map_err(|e| Error::Io(crate::error::io_error(format!("シート '{}' を読み込めませんでした: {}", sheet_name, e))))?;
        
        // 列名（ヘッダー）を取得
        let mut column_names: Vec<String> = Vec::new();
        if header && !range.is_empty() && skip_rows < range.rows().len() {
            // ヘッダー行を取得
            let header_row = range.rows().nth(skip_rows).unwrap();
            
            // 列名を文字列に変換
            for cell in header_row {
                column_names.push(cell.to_string());
            }
        } else {
            // ヘッダーがない場合、列番号を列名として使用
            if !range.is_empty() {
                let first_row = range.rows().next().unwrap();
                for i in 0..first_row.len() {
                    column_names.push(format!("Column{}", i+1));
                }
            }
        }
        
        // 読み込む列を決定
        let use_cols_indices = if let Some(cols) = use_cols {
            // 指定された列のインデックスを取得
            let mut indices = Vec::new();
            for col_name in cols {
                if let Some(pos) = column_names.iter().position(|name| name == col_name) {
                    indices.push(pos);
                }
            }
            Some(indices)
        } else {
            None
        };
        
        // データフレームを作成
        let mut df = Self::new();
        
        // 列ごとにデータを収集
        let mut column_data: HashMap<usize, Vec<String>> = HashMap::new();
        let start_row = if header { skip_rows + 1 } else { skip_rows };
        
        for (row_idx, row) in range.rows().enumerate().skip(start_row) {
            for (col_idx, cell) in row.iter().enumerate() {
                // 使用する列のみ処理
                if let Some(ref indices) = use_cols_indices {
                    if !indices.contains(&col_idx) {
                        continue;
                    }
                }
                
                // 列データに追加
                column_data.entry(col_idx)
                    .or_insert_with(Vec::new)
                    .push(cell.to_string());
            }
        }
        
        // 列データをDataFrameに追加
        for col_idx in 0..column_names.len() {
            // 使用する列のみ処理
            if let Some(ref indices) = use_cols_indices {
                if !indices.contains(&col_idx) {
                    continue;
                }
            }
            
            let col_name = column_names.get(col_idx)
                .unwrap_or(&format!("Column{}", col_idx+1))
                .clone();
            
            // 列データを取得
            let data = column_data.get(&col_idx).cloned().unwrap_or_default();
            
            // 空の列はスキップ
            if data.is_empty() {
                continue;
            }
            
            // データ型を推測して適切な列を作成
            let non_empty_values: Vec<&String> = data.iter().filter(|s| !s.is_empty()).collect();
            
            if non_empty_values.is_empty() {
                // すべて空の場合は文字列型
                df.add_column(col_name, Column::String(StringColumn::new(data)))?;
                continue;
            }
            
            // 整数型として解析を試みる
            let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
            if all_ints {
                let int_values: Vec<i64> = data.iter()
                    .map(|s| s.parse::<i64>().unwrap_or(0))
                    .collect();
                df.add_column(col_name, Column::Int64(Int64Column::new(int_values)))?;
                continue;
            }
            
            // 浮動小数点型として解析を試みる
            let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = data.iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                df.add_column(col_name, Column::Float64(Float64Column::new(float_values)))?;
                continue;
            }
            
            // ブール型として解析を試みる
            let all_bools = non_empty_values.iter().all(|&s| {
                let s = s.trim().to_lowercase();
                s == "true" || s == "false" || s == "1" || s == "0" || 
                s == "yes" || s == "no" || s == "t" || s == "f"
            });
            
            if all_bools {
                let bool_values: Vec<bool> = data.iter()
                    .map(|s| {
                        let s = s.trim().to_lowercase();
                        s == "true" || s == "1" || s == "yes" || s == "t"
                    })
                    .collect();
                df.add_column(col_name, Column::Boolean(BooleanColumn::new(bool_values)))?;
            } else {
                // デフォルトは文字列型
                df.add_column(col_name, Column::String(StringColumn::new(data)))?;
            }
        }
        
        Ok(df)
    }
    
    /// DataFrameをExcelファイル (.xlsx) に書き込む
    ///
    /// # Arguments
    /// * `path` - 出力するExcelファイルのパス
    /// * `sheet_name` - シート名 (Noneの場合、"Sheet1"が使用される)
    /// * `index` - インデックスを含めるかどうか
    ///
    /// # Returns
    /// * `Result<()>` - 成功時はOk
    pub fn to_excel<P: AsRef<Path>>(
        &self,
        path: P,
        sheet_name: Option<&str>,
        index: bool,
    ) -> Result<()> {
        // 新しいExcelファイルを作成
        let mut workbook = Workbook::create(path.as_ref()
            .to_str()
            .ok_or_else(|| Error::Io(crate::error::io_error("ファイルパスを文字列に変換できませんでした")))?);
        
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
                for col in &self.columns {
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
            .map_err(|e| Error::Io(crate::error::io_error(format!("Excelファイルを保存できませんでした: {}", e))))?;
        
        Ok(())
    }
    
    /// SQLクエリの実行結果からDataFrameを作成する
    ///
    /// # Arguments
    /// * `query` - 実行するSQLクエリ
    /// * `db_path` - SQLiteデータベースファイルのパス
    ///
    /// # Returns
    /// * `Result<Self>` - クエリ結果を含むDataFrame
    pub fn from_sql<P: AsRef<Path>>(query: &str, db_path: P) -> Result<Self> {
        // データベース接続
        let mut conn = Connection::open(db_path)
            .map_err(|e| Error::Io(crate::error::io_error(format!("データベースに接続できませんでした: {}", e))))?;
        
        // クエリを準備
        let mut stmt = conn.prepare(query)
            .map_err(|e| Error::Io(crate::error::io_error(format!("SQLクエリの準備に失敗しました: {}", e))))?;
        
        // 列名を取得
        let column_names: Vec<String> = stmt.column_names().iter()
            .map(|&name| name.to_string())
            .collect();
        
        // 列ごとのデータを格納するためのマップ
        let mut column_data: HashMap<String, Vec<String>> = HashMap::new();
        for name in &column_names {
            column_data.insert(name.clone(), Vec::new());
        }
        
        // クエリを実行して結果を取得
        let mut rows = stmt.query([])
            .map_err(|e| Error::Io(crate::error::io_error(format!("SQLクエリの実行に失敗しました: {}", e))))?;
        
        // 各行のデータを処理
        while let Some(row) = rows.next()
            .map_err(|e| Error::Io(crate::error::io_error(format!("SQLクエリの結果取得に失敗しました: {}", e))))? {
            
            for (idx, name) in column_names.iter().enumerate() {
                let value: Option<String> = row.get(idx)
                    .map_err(|e| Error::Io(crate::error::io_error(format!("行データの取得に失敗しました: {}", e))))?;
                
                if let Some(data) = column_data.get_mut(name) {
                    data.push(value.unwrap_or_else(|| "NULL".to_string()));
                }
            }
        }
        
        // DataFrameを作成
        let mut df = Self::new();
        
        // 列データからデータフレームを作成
        for name in column_names {
            if let Some(data) = column_data.get(&name) {
                // 空でない値のチェック
                let non_empty_values: Vec<&String> = data.iter()
                    .filter(|s| !s.is_empty() && *s != "NULL")
                    .collect();
                
                if non_empty_values.is_empty() {
                    // すべて空の場合は文字列型
                    df.add_column(name, Column::String(StringColumn::new(
                        data.iter().map(|s| s.clone()).collect()
                    )))?;
                    continue;
                }
                
                // 整数型として解析を試みる
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = data.iter()
                        .map(|s| {
                            if s.is_empty() || s == "NULL" {
                                0
                            } else {
                                s.parse::<i64>().unwrap_or(0)
                            }
                        })
                        .collect();
                    df.add_column(name, Column::Int64(Int64Column::new(int_values)))?;
                    continue;
                }
                
                // 浮動小数点型として解析を試みる
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = data.iter()
                        .map(|s| {
                            if s.is_empty() || s == "NULL" {
                                0.0
                            } else {
                                s.parse::<f64>().unwrap_or(0.0)
                            }
                        })
                        .collect();
                    df.add_column(name, Column::Float64(Float64Column::new(float_values)))?;
                    continue;
                }
                
                // ブール型として解析を試みる
                let all_bools = non_empty_values.iter().all(|&s| {
                    let s = s.trim().to_lowercase();
                    s == "true" || s == "false" || s == "1" || s == "0" || 
                    s == "yes" || s == "no" || s == "t" || s == "f"
                });
                
                if all_bools {
                    let bool_values: Vec<bool> = data.iter()
                        .map(|s| {
                            let s = s.trim().to_lowercase();
                            s == "true" || s == "1" || s == "yes" || s == "t"
                        })
                        .collect();
                    df.add_column(name, Column::Boolean(BooleanColumn::new(bool_values)))?;
                } else {
                    // デフォルトは文字列型
                    df.add_column(name, Column::String(StringColumn::new(
                        data.iter().map(|s| if s == "NULL" { String::new() } else { s.clone() }).collect()
                    )))?;
                }
            }
        }
        
        Ok(df)
    }
    
    /// DataFrameをSQLiteテーブルに書き込む
    ///
    /// # Arguments
    /// * `table_name` - テーブル名
    /// * `db_path` - SQLiteデータベースファイルのパス
    /// * `if_exists` - テーブルが存在する場合の動作 ("fail", "replace", "append")
    ///
    /// # Returns
    /// * `Result<()>` - 成功時はOk
    pub fn to_sql<P: AsRef<Path>>(&self, table_name: &str, db_path: P, if_exists: &str) -> Result<()> {
        // データベース接続
        let mut conn = Connection::open(db_path)
            .map_err(|e| Error::Io(crate::error::io_error(format!("データベースに接続できませんでした: {}", e))))?;
        
        // テーブルが存在するか確認
        let table_exists = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
            .map_err(|e| Error::Io(crate::error::io_error(format!("テーブル確認クエリの準備に失敗しました: {}", e))))?
            .exists(params![table_name])
            .map_err(|e| Error::Io(crate::error::io_error(format!("テーブル存在確認に失敗しました: {}", e))))?;
        
        // if_exists に基づいてテーブルを処理
        if table_exists {
            match if_exists {
                "fail" => {
                    return Err(Error::Io(crate::error::io_error(format!("テーブル '{}' は既に存在します", table_name))));
                },
                "replace" => {
                    // テーブルを削除して新規作成
                    conn.execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
                        .map_err(|e| Error::Io(crate::error::io_error(format!("テーブルの削除に失敗しました: {}", e))))?;
                    
                    // 新しいテーブルを作成
                    self.create_table_from_df(&conn, table_name)?;
                },
                "append" => {
                    // テーブルは既に存在するので、このままデータを追加
                },
                _ => {
                    return Err(Error::Io(crate::error::io_error(format!("不明なif_exists値: {}", if_exists))));
                }
            }
        } else {
            // テーブルが存在しない場合は新規作成
            self.create_table_from_df(&conn, table_name)?;
        }
        
        // データの挿入
        // カラム名のリスト
        let columns = self.column_names.join(", ");
        
        // プレースホルダーのリスト
        let placeholders: Vec<String> = (0..self.column_names.len())
            .map(|_| "?".to_string())
            .collect();
        let placeholders = placeholders.join(", ");
        
        // INSERT文を準備
        let insert_sql = format!("INSERT INTO {} ({}) VALUES ({})", table_name, columns, placeholders);
        
        // トランザクション開始
        let tx = conn.transaction()
            .map_err(|e| Error::Io(crate::error::io_error(format!("トランザクションの開始に失敗しました: {}", e))))?;
        
        // 各行のデータを挿入
        for row_idx in 0..self.row_count {
            // 行データを取得
            let mut row_values: Vec<String> = Vec::new();
            
            for col in &self.columns {
                let value = match col {
                    Column::Int64(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Float64(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Boolean(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                };
                
                row_values.push(value);
            }
            
            // INSERT実行
            let mut stmt = tx.prepare(&insert_sql)
                .map_err(|e| Error::Io(crate::error::io_error(format!("INSERT文の準備に失敗しました: {}", e))))?;
            
            let params: Vec<&dyn rusqlite::ToSql> = row_values.iter()
                .map(|s| s as &dyn rusqlite::ToSql)
                .collect();
            
            stmt.execute(params.as_slice())
                .map_err(|e| Error::Io(crate::error::io_error(format!("データの挿入に失敗しました: {}", e))))?;
        }
        
        // トランザクションをコミット
        tx.commit()
            .map_err(|e| Error::Io(crate::error::io_error(format!("トランザクションのコミットに失敗しました: {}", e))))?;
        
        Ok(())
    }
    
    // DataFrameからSQLiteテーブルを作成するヘルパーメソッド
    fn create_table_from_df(&self, conn: &Connection, table_name: &str) -> Result<()> {
        // 列名と型のリストを作成
        let mut columns = Vec::new();
        
        for (idx, col_name) in self.column_names.iter().enumerate() {
            let sql_type = match &self.columns[idx] {
                Column::Int64(_) => "INTEGER",
                Column::Float64(_) => "REAL",
                Column::Boolean(_) => "INTEGER", // SQLiteではブール値は整数として保存
                Column::String(_) => "TEXT",
            };
            
            columns.push(format!("{} {}", col_name, sql_type));
        }
        
        // CREATE TABLE文を作成して実行
        let create_sql = format!("CREATE TABLE {} ({})", table_name, columns.join(", "));
        conn.execute(&create_sql, [])
            .map_err(|e| Error::Io(crate::error::io_error(format!("テーブルの作成に失敗しました: {}", e))))?;
        
        Ok(())
    }
}
