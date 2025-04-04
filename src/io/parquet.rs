use std::fs::File;
use std::path::Path;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};

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

/// Parquetファイルからデータフレームを読み込む
pub fn read_parquet(path: impl AsRef<Path>) -> Result<DataFrame> {
    // 現在は実装されていないので、未実装エラーを返す
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
    Err(Error::NotImplemented(
        "Parquet書き込み機能は現在実装中です。将来のバージョンで利用可能になります。".to_string()
    ))
}