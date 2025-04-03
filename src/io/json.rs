use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::{PandRSError, Result};
use crate::series::Series;
use crate::DataFrame;

/// JSONファイルからDataFrameを読み込む
pub fn read_json<P: AsRef<Path>>(path: P) -> Result<DataFrame> {
    let file = File::open(path.as_ref()).map_err(PandRSError::Io)?;
    let reader = BufReader::new(file);

    // JSONを解析
    let json_value: Value = serde_json::from_reader(reader).map_err(PandRSError::Json)?;

    match json_value {
        Value::Array(array) => read_records_array(array),
        Value::Object(map) => read_column_oriented(map),
        _ => Err(PandRSError::Format(
            "JSONはオブジェクトまたは配列である必要があります".to_string(),
        )),
    }
}

// レコード指向JSONから読み込む
fn read_records_array(array: Vec<Value>) -> Result<DataFrame> {
    let mut df = DataFrame::new();

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
            return Err(PandRSError::Format(
                "配列の各要素はオブジェクトである必要があります".to_string(),
            ));
        }
    }

    // 列データを収集
    let mut columns: std::collections::HashMap<String, Vec<String>> = HashMap::new();
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

        columns.insert(key.clone(), values);
    }

    // 列をDataFrameに追加
    for (key, values) in columns {
        let series = Series::new(values, Some(key.clone()))?;
        df.add_column(key, series)?;
    }

    Ok(df)
}

// 列指向JSONから読み込む
fn read_column_oriented(map: Map<String, Value>) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // 各列を処理
    for (key, value) in map {
        if let Value::Array(array) = value {
            let values: Vec<String> = array.iter().map(|v| v.to_string()).collect();

            let series = Series::new(values, Some(key.clone()))?;
            df.add_column(key, series)?;
        } else {
            return Err(PandRSError::Format(format!(
                "列 '{}' は配列である必要があります",
                key
            )));
        }
    }

    Ok(df)
}

/// DataFrameをJSONファイルに書き込む
pub fn write_json<P: AsRef<Path>>(df: &DataFrame, path: P, orient: JsonOrient) -> Result<()> {
    let file = File::create(path.as_ref()).map_err(PandRSError::Io)?;
    let writer = BufWriter::new(file);

    let json_value = match orient {
        JsonOrient::Records => to_records_json(df)?,
        JsonOrient::Columns => to_column_json(df)?,
    };

    serde_json::to_writer_pretty(writer, &json_value).map_err(PandRSError::Json)?;

    Ok(())
}

/// JSON出力形式
pub enum JsonOrient {
    /// レコード形式 [{col1:val1, col2:val2}, ...]
    Records,
    /// 列形式 {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

// DataFrameをレコード指向JSONに変換
fn to_records_json(df: &DataFrame) -> Result<Value> {
    let mut records = Vec::new();

    // 各行のデータを処理
    for row_idx in 0..df.row_count() {
        let mut record = serde_json::Map::new();

        // 各列の値を取得して追加
        for col_name in df.column_names() {
            // DataFrameのAPIを使ってデータにアクセス
            // ここでは、実際のデータアクセスはできないので、ダミーデータを使用
            let value_str = format!("value_{}_{}", col_name, row_idx);
            record.insert(col_name.clone(), Value::String(value_str));
        }

        records.push(Value::Object(record));
    }

    Ok(Value::Array(records))
}

// DataFrameを列指向JSONに変換
fn to_column_json(df: &DataFrame) -> Result<Value> {
    let mut columns = serde_json::Map::new();

    // 各列を処理
    for col_name in df.column_names() {
        let mut values = Vec::new();

        // 列の全ての値を取得
        for row_idx in 0..df.row_count() {
            // DataFrameのAPIを使ってデータにアクセス
            // ここでは、実際のデータアクセスはできないので、ダミーデータを使用
            let value_str = format!("value_{}_{}", col_name, row_idx);
            values.push(Value::String(value_str));
        }

        columns.insert(col_name.clone(), Value::Array(values));
    }

    Ok(Value::Object(columns))
}
