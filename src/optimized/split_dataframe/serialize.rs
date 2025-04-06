use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::column::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

/// JSON出力形式
pub enum JsonOrient {
    /// レコード形式 [{col1:val1, col2:val2}, ...]
    Records,
    /// 列形式 {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

impl OptimizedDataFrame {
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

        // キーをソートして順序を安定化
        let keys: Vec<String> = all_keys.into_iter().collect();

        // 列データを収集
        let mut string_values: HashMap<String, Vec<String>> = HashMap::new();

        for key in &keys {
            string_values.insert(key.clone(), Vec::with_capacity(array.len()));
        }

        for item in &array {
            if let Value::Object(map) = item {
                for key in &keys {
                    let value_str = if let Some(value) = map.get(key) {
                        match value {
                            Value::Null => String::new(),
                            Value::Bool(b) => b.to_string(),
                            Value::Number(n) => n.to_string(),
                            Value::String(s) => s.clone(),
                            _ => serde_json::to_string(value).unwrap_or_default(),
                        }
                    } else {
                        String::new()
                    };
                    string_values.get_mut(key).unwrap().push(value_str);
                }
            }
        }

        // 型推論して列を追加
        for key in &keys {
            let values = &string_values[key];
            
            // 空でない値のチェック
            let non_empty_values: Vec<&String> = values.iter().filter(|s| !s.is_empty()).collect();
            
            if non_empty_values.is_empty() {
                // すべて空の場合は文字列型
                df.add_column(key.clone(), Column::String(crate::column::StringColumn::new(
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
                df.add_column(key.clone(), Column::Int64(crate::column::Int64Column::new(int_values)))?;
                continue;
            }
            
            // 浮動小数点型として解析を試みる
            let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = values.iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                df.add_column(key.clone(), Column::Float64(crate::column::Float64Column::new(float_values)))?;
                continue;
            }
            
            // ブール型として解析を試みる
            let all_bools = non_empty_values.iter().all(|&s| {
                let lower = s.to_lowercase();
                lower == "true" || lower == "false"
            });
            
            if all_bools {
                let bool_values: Vec<bool> = values.iter()
                    .map(|s| s.to_lowercase() == "true")
                    .collect();
                df.add_column(key.clone(), Column::Boolean(crate::column::BooleanColumn::new(bool_values)))?;
            } else {
                // デフォルトは文字列型
                df.add_column(key.clone(), Column::String(crate::column::StringColumn::new(
                    values.iter().map(|s| s.clone()).collect()
                )))?;
            }
        }

        Ok(df)
    }

    // 列指向JSONから読み込む
    fn from_column_oriented(map: Map<String, Value>) -> Result<Self> {
        let mut df = Self::new();

        // 空オブジェクトの場合は空のDataFrameを返す
        if map.is_empty() {
            return Ok(df);
        }

        // 列の長さを確認
        let mut column_length = 0;
        for (_, value) in &map {
            if let Value::Array(array) = value {
                if column_length == 0 {
                    column_length = array.len();
                } else if array.len() != column_length {
                    return Err(Error::Format(
                        "全ての列の長さが一致する必要があります".to_string(),
                    ));
                }
            } else {
                return Err(Error::Format(
                    "JSONの値は配列である必要があります".to_string(),
                ));
            }
        }

        // 列データを処理
        for (key, value) in map {
            if let Value::Array(array) = value {
                // 値を文字列に変換
                let str_values: Vec<String> = array.iter()
                    .map(|v| match v {
                        Value::Null => String::new(),
                        Value::Bool(b) => b.to_string(),
                        Value::Number(n) => n.to_string(),
                        Value::String(s) => s.clone(),
                        _ => serde_json::to_string(v).unwrap_or_default(),
                    })
                    .collect();
                
                // 空でない値のチェック
                let non_empty_values: Vec<&String> = str_values.iter().filter(|s| !s.is_empty()).collect();
                
                if non_empty_values.is_empty() {
                    // すべて空の場合は文字列型
                    df.add_column(key.clone(), Column::String(crate::column::StringColumn::new(
                        str_values.iter().map(|s| s.clone()).collect()
                    )))?;
                    continue;
                }
                
                // 整数型として解析を試みる
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = str_values.iter()
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                        .collect();
                    df.add_column(key.clone(), Column::Int64(crate::column::Int64Column::new(int_values)))?;
                    continue;
                }
                
                // 浮動小数点型として解析を試みる
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = str_values.iter()
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                        .collect();
                    df.add_column(key.clone(), Column::Float64(crate::column::Float64Column::new(float_values)))?;
                    continue;
                }
                
                // ブール型として解析を試みる
                let all_bools = non_empty_values.iter().all(|&s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "false"
                });
                
                if all_bools {
                    let bool_values: Vec<bool> = str_values.iter()
                        .map(|s| s.to_lowercase() == "true")
                        .collect();
                    df.add_column(key.clone(), Column::Boolean(crate::column::BooleanColumn::new(bool_values)))?;
                } else {
                    // デフォルトは文字列型
                    df.add_column(key.clone(), Column::String(crate::column::StringColumn::new(
                        str_values.iter().map(|s| s.clone()).collect()
                    )))?;
                }
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
        
        // JSON形式に変換
        let json_value = match orient {
            JsonOrient::Records => self.to_records_json()?,
            JsonOrient::Columns => self.to_column_json()?,
        };
        
        // JSONを書き込む
        serde_json::to_writer_pretty(writer, &json_value).map_err(|e| Error::Json(e))?;
        
        Ok(())
    }
    
    // DataFrameをレコード指向JSONに変換
    fn to_records_json(&self) -> Result<Value> {
        let mut records = Vec::with_capacity(self.row_count());
        
        // 行がない場合は空の配列を返す
        if self.row_count() == 0 {
            return Ok(Value::Array(records));
        }
        
        // 各行のデータを処理
        for row_idx in 0..self.row_count() {
            let mut record = Map::new();
            
            // 各列の値を取得
            for (col_idx, col_name) in self.column_names().iter().enumerate() {
                let column = &self.columns[col_idx];
                
                // 列の値をJSON値に変換
                let value = match column {
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
        if self.row_count() == 0 {
            return Ok(Value::Object(columns));
        }
        
        // 各列を処理
        for (col_idx, col_name) in self.column_names().iter().enumerate() {
            let mut values = Vec::new();
            
            // 列の全ての値を取得
            for row_idx in 0..self.row_count() {
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
}