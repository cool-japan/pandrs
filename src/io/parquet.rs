use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::column::{BooleanColumn, Column, ColumnType, Float64Column, Int64Column, StringColumn};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;

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

/// Parquetファイルからデータフレームを読み込む
///
/// # 引数
///
/// * `path` - Parquetファイルのパス
///
/// # 戻り値
///
/// * `Result<DataFrame>` - 読み込んだデータフレーム、またはエラー
///
/// # 例
///
/// ```no_run
/// use pandrs::io::read_parquet;
///
/// // Parquetファイルからデータフレームを読み込む
/// let df = read_parquet("data.parquet").unwrap();
/// ```
pub fn read_parquet(path: impl AsRef<Path>) -> Result<DataFrame> {
    // ファイルを開く
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("Parquetファイルを開けませんでした: {}", e)))?;
    
    // ArrowのParquetReaderを作成
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::IoError(format!("Parquetファイルの解析に失敗しました: {}", e)))?;
    
    // スキーマ情報を取得
    let schema = builder.schema().clone();
    
    // レコードバッチリーダーを作成
    let reader = builder.build()
        .map_err(|e| Error::IoError(format!("Parquetファイルの読み込みに失敗しました: {}", e)))?;
    
    // 全てのレコードバッチを読み込む
    let mut all_batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| Error::IoError(format!("レコードバッチの読み込みに失敗しました: {}", e)))?;
        all_batches.push(batch);
    }
    
    // レコードバッチがない場合は空のデータフレームを返す
    if all_batches.is_empty() {
        return Ok(DataFrame::new());
    }
    
    // データフレームに変換
    record_batches_to_dataframe(&all_batches, schema)
}

/// Arrowのレコードバッチをデータフレームに変換する
fn record_batches_to_dataframe(batches: &[RecordBatch], schema: SchemaRef) -> Result<DataFrame> {
    let mut df = DataFrame::new();
    
    // スキーマから列情報を取得
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name().clone();
        let col_type = field.data_type();
        
        // 全てのバッチから列データを収集
        match col_type {
            DataType::Int64 => {
                let mut values = Vec::new();
                
                for batch in batches {
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
                
                let series = Series::new(values, Some(col_name.clone()))?;
                df.add_column(col_name.clone(), series)?;
            },
            DataType::Float64 => {
                let mut values = Vec::new();
                
                for batch in batches {
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
                
                let series = Series::new(values, Some(col_name.clone()))?;
                df.add_column(col_name.clone(), series)?;
            },
            DataType::Boolean => {
                let mut values = Vec::new();
                
                for batch in batches {
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
                
                let series = Series::new(values, Some(col_name.clone()))?;
                df.add_column(col_name.clone(), series)?;
            },
            DataType::Utf8 | DataType::LargeUtf8 => {
                let mut values = Vec::new();
                
                for batch in batches {
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
                
                let series = Series::new(values, Some(col_name.clone()))?;
                df.add_column(col_name.clone(), series)?;
            },
            _ => {
                // サポートされていないデータ型は文字列として扱う
                let mut values = Vec::new();
                
                for batch in batches {
                    let array = batch.column(col_idx);
                    for i in 0..array.len() {
                        if array.is_null(i) {
                            values.push("".to_string());
                        } else {
                            values.push(format!("{:?}", array));
                        }
                    }
                }
                
                let series = Series::new(values, Some(col_name.clone()))?;
                df.add_column(col_name.clone(), series)?;
            },
        }
    }
    
    Ok(df)
}

/// データフレームをParquetファイルに書き込む
///
/// # 引数
///
/// * `df` - 書き込むデータフレーム
/// * `path` - 出力Parquetファイルのパス
/// * `compression` - 圧縮オプション（デフォルトはSnappy）
///
/// # 戻り値
///
/// * `Result<()>` - 成功した場合はOk(())、失敗した場合はエラー
///
/// # 例
///
/// ```no_run
/// use pandrs::dataframe::DataFrame;
/// use pandrs::io::{write_parquet, ParquetCompression};
/// use pandrs::series::Series;
///
/// let mut df = DataFrame::new();
/// df.add_series(Series::new(vec![1, 2, 3], Some("A".to_string())).unwrap()).unwrap();
/// df.add_series(Series::new(vec![4.1, 5.2, 6.3], Some("B".to_string())).unwrap()).unwrap();
///
/// // Snappy圧縮を使用してParquetファイルに書き込む
/// write_parquet(&df, "output.parquet", Some(ParquetCompression::Snappy)).unwrap();
/// ```
pub fn write_parquet(
    df: &DataFrame,
    path: impl AsRef<Path>,
    compression: Option<ParquetCompression>,
) -> Result<()> {
    // Arrowスキーマを作成
    let schema_fields: Vec<Field> = df.column_names()
        .iter()
        .filter_map(|col_name| {
            // 各列を文字列シリーズとして取得
            if let Some(series) = df.get_column(col_name) {
                // 文字列表現から型を推定
                let data_type = match series.name().map_or("unknown", |s| s).split_whitespace().next().unwrap_or("") {
                    "i64" | "Int64" => DataType::Int64,
                    "f64" | "Float64" => DataType::Float64,
                    "bool" | "Boolean" => DataType::Boolean,
                    _ => DataType::Utf8,
                };
                Some(Field::new(col_name, data_type, true))
            } else {
                None
            }
        })
        .collect();
    
    let schema = Schema::new(schema_fields);
    let schema_ref = Arc::new(schema);
    
    // 列データをArrow配列に変換
    let arrays: Vec<ArrayRef> = df.column_names()
        .iter()
        .filter_map(|col_name| {
            // 各列を文字列シリーズとして取得
            let series = match df.get_column(col_name) {
                Some(s) => s,
                None => return None,
            };
            
            // 文字列表現から型を推定
            let series_type = series.name().map_or("unknown", |s| s).split_whitespace().next().unwrap_or("");
            
            match series_type {
                "i64" | "Int64" => {
                    // 整数値を取得
                    let values: Vec<i64> = series.values().iter()
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                        .collect();
                    Some(Arc::new(Int64Array::from(values)) as ArrayRef)
                },
                "f64" | "Float64" => {
                    // 浮動小数点値を取得
                    let values: Vec<f64> = series.values().iter()
                        .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
                        .collect();
                    Some(Arc::new(Float64Array::from(values)) as ArrayRef)
                },
                "bool" | "Boolean" => {
                    // 論理値を取得
                    let values: Vec<bool> = series.values().iter()
                        .map(|s| {
                            let s = s.to_lowercase();
                            s == "true" || s == "1"
                        })
                        .collect();
                    Some(Arc::new(BooleanArray::from(values)) as ArrayRef)
                },
                _ => {
                    // それ以外は文字列として扱う
                    let str_values: Vec<&str> = series.values().iter().map(|s| s.as_str()).collect();
                    Some(Arc::new(StringArray::from(str_values)) as ArrayRef)
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
        .map_err(|e| Error::IoError(format!("Parquetファイルを作成できませんでした: {}", e)))?;
    
    // Arrowライターを作成して書き込み
    let mut writer = ArrowWriter::try_new(file, schema_ref, Some(props))
        .map_err(|e| Error::IoError(format!("Parquetライターの作成に失敗しました: {}", e)))?;
    
    // レコードバッチを書き込む
    writer.write(&batch)
        .map_err(|e| Error::IoError(format!("レコードバッチの書き込みに失敗しました: {}", e)))?;
    
    // ファイルを閉じる
    writer.close()
        .map_err(|e| Error::IoError(format!("Parquetファイルの閉じる操作に失敗しました: {}", e)))?;
    
    Ok(())
}