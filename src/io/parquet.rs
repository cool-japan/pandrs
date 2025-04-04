use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
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
pub fn read_parquet(path: impl AsRef<Path>) -> Result<DataFrame> {
    // 現在は実装されていないので、未実装エラーを返す
    // ただし、依存関係は導入されたので、今後の更新で実装予定
    Err(Error::NotImplemented(
        "Parquet読み込み機能は現在実装中です。将来のバージョンで利用可能になります。".to_string()
    ))
}

/// データフレームをParquetファイルに書き込む
pub fn write_parquet(
    df: &DataFrame,
    path: impl AsRef<Path>,
    compression: Option<ParquetCompression>,
) -> Result<()> {
    // 現在は実装されていないので、未実装エラーを返す
    // ただし、依存関係は導入されたので、今後の更新で実装予定
    Err(Error::NotImplemented(
        "Parquet書き込み機能は現在実装中です。将来のバージョンで利用可能になります。".to_string()
    ))
}