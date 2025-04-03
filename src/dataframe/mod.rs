mod join;

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

use crate::error::{PandRSError, Result};
use crate::index::RangeIndex;
use crate::io;
use crate::series::Series;

/// データの値を表すトレイト
/// 型消去のための基本トレイト
pub trait DataValue: Debug + Any + Send + Sync {
    /// 値のクローンを作成
    fn box_clone(&self) -> Box<dyn DataValue>;

    /// Anyトレイトへのダウンキャスト
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static + Debug + Clone + Send + Sync> DataValue for T {
    fn box_clone(&self) -> Box<dyn DataValue> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// DataValueを格納するラッパー型
#[derive(Debug)]
pub struct DataBox(Box<dyn DataValue>);

impl Clone for DataBox {
    fn clone(&self) -> Self {
        DataBox(self.0.box_clone())
    }
}

/// DataFrame構造体: 列指向の2次元データ構造
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// 列名から列データへのマッピング
    columns: HashMap<String, Series<DataBox>>,

    /// 列の順序を保持
    column_names: Vec<String>,

    /// 行インデックス
    index: RangeIndex,
}

impl DataFrame {
    /// 新しい空のDataFrameを作成
    pub fn new() -> Self {
        // 空のインデックスを作成
        let index = RangeIndex::from_range(0..0).unwrap();

        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index,
        }
    }

    /// 列を追加
    pub fn add_column<T>(&mut self, name: String, series: Series<T>) -> Result<()>
    where
        T: Debug + Clone + 'static + Send + Sync,
    {
        // 既にその名前の列が存在するかチェック
        if self.columns.contains_key(&name) {
            return Err(PandRSError::Column(format!(
                "列名 '{}' は既に存在します",
                name
            )));
        }

        // 空のDataFrameでなければ行数をチェック
        if !self.column_names.is_empty() && series.len() != self.index.len() {
            return Err(PandRSError::Consistency(format!(
                "列の長さ ({}) がDataFrameの行数 ({}) と一致しません",
                series.len(),
                self.index.len()
            )));
        }

        // 型消去してBoxに変換
        let boxed_series = self.box_series(series);

        // 最初の列の場合、インデックスを設定
        if self.column_names.is_empty() {
            self.index = RangeIndex::from_range(0..boxed_series.len())?;
        }

        // 列を追加
        self.columns.insert(name.clone(), boxed_series);
        self.column_names.push(name);

        Ok(())
    }

    // Seriesをボックス化する内部ヘルパー
    fn box_series<T>(&self, series: Series<T>) -> Series<DataBox>
    where
        T: Debug + Clone + 'static + Send + Sync,
    {
        // 値をボックス化
        let boxed_values: Vec<DataBox> = series
            .values()
            .iter()
            .map(|v| DataBox(Box::new(v.clone())))
            .collect();

        // 新しいSeriesを作成
        Series::new(boxed_values, series.name().cloned()).unwrap()
    }

    /// 列数を取得
    pub fn column_count(&self) -> usize {
        self.column_names.len()
    }

    /// 行数を取得
    pub fn row_count(&self) -> usize {
        self.index.len()
    }

    /// 列名リストを取得
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// 列が存在するか確認
    pub fn contains_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// DataFrameがCSVファイルに保存
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        io::csv::write_csv(self, path)
    }

    /// CSVファイルからDataFrameを作成
    pub fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        io::csv::read_csv(path, has_header)
    }

    /// 列の文字列値を取得する（ピボット用）
    pub fn get_column_string_values(&self, column: &str) -> Result<Vec<String>> {
        if let Some(series) = self.columns.get(column) {
            let values = series
                .values()
                .iter()
                .map(|data_box| {
                    // DataBoxからの文字列化
                    format!("{:?}", data_box)
                })
                .collect();
            Ok(values)
        } else {
            Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                column
            )))
        }
    }

    /// 列の数値データを取得する（ピボット用）
    pub fn get_column_numeric_values(&self, column: &str) -> Result<Vec<f64>> {
        if let Some(series) = self.columns.get(column) {
            let values = series
                .values()
                .iter()
                .map(|data_box| {
                    // DataBoxから数値化（文字列から数値への変換）
                    // 実際のアプリケーションではより精緻な型変換が必要
                    let str_val = format!("{:?}", data_box);
                    str_val.parse::<f64>().unwrap_or(0.0)
                })
                .collect();
            Ok(values)
        } else {
            Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                column
            )))
        }
    }
}

// Defaultトレイト実装
impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}
