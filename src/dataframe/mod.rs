mod join;
pub mod apply;
mod categorical;
mod transform;

pub use apply::Axis;
pub use transform::{MeltOptions, StackOptions, UnstackOptions};

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

use crate::error::{PandRSError, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait, RangeIndex};
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
    index: DataFrameIndex<String>,
}

impl DataFrame {
    /// 新しい空のDataFrameを作成
    pub fn new() -> Self {
        // 空のインデックスを作成
        let range_idx = Index::<usize>::from_range(0..0).unwrap();
        // 数値インデックスを文字列インデックスに変換
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values).unwrap();
        let index = DataFrameIndex::<String>::from_simple(string_idx);

        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index,
        }
    }
    
    /// 行データをDataFrameに追加する
    /// 
    /// この関数は文字列表現のデータをDataFrameに1行追加します。
    /// 指定したマップには、列名とその値の文字列表現が含まれます。
    /// マップに含まれていない列には空文字列が入ります。
    pub fn add_row_data(&mut self, row_data: HashMap<String, String>) -> Result<()> {
        if self.columns.is_empty() {
            return Err(PandRSError::InvalidOperation("列がないDataFrameに行を追加できません".to_string()));
        }
        
        // 既存のすべての列と値を取得
        let mut all_column_data = HashMap::new();
        let current_row_count = self.row_count();
        
        // 現在のデータを収集
        for col_name in self.column_names.to_vec() {
            if let Some(series) = self.columns.get(&col_name) {
                let mut values = Vec::with_capacity(current_row_count + 1);
                
                // 既存のデータを収集
                for i in 0..current_row_count {
                    values.push(series.get(i).cloned().unwrap_or_else(|| DataBox(Box::new("".to_string()))));
                }
                
                // 新しい行のデータを追加
                if let Some(value) = row_data.get(&col_name) {
                    values.push(DataBox(Box::new(value.clone())));
                } else {
                    // デフォルト値を追加（空文字列など）
                    values.push(DataBox(Box::new("".to_string())));
                }
                
                all_column_data.insert(col_name, values);
            }
        }
        
        // 新しいDataFrameの構築
        let mut new_df = DataFrame::new();
        
        // 元のカラム順序を維持して、新しいDataFrameに追加
        for col_name in self.column_names.to_vec() {
            if let Some(values) = all_column_data.get(&col_name) {
                let new_series = Series::<DataBox>::new(values.clone(), Some(col_name.clone()))?;
                new_df.add_column(col_name.clone(), new_series)?;
            }
        }
        
        // 結果を更新
        *self = new_df;
        
        Ok(())
    }

    /// 列名を変更する
    pub fn rename_columns(&mut self, column_map: &HashMap<String, String>) -> Result<()> {
        // 古い列名から新しい列名への変換マップを作成
        let mut new_columns = HashMap::new();
        let mut new_column_names = Vec::with_capacity(self.column_names.len());
        
        // 元の列名のリストを取得
        for old_name in &self.column_names {
            // 変換マップに存在するかチェック
            let new_name = if let Some(new_col_name) = column_map.get(old_name) {
                new_col_name.clone()
            } else {
                // マップに存在しない場合は元の名前を使用
                old_name.clone()
            };
            
            // 同名の列が既に存在するかチェック（変換後の列名が重複する場合）
            if new_column_names.contains(&new_name) && old_name != &new_name {
                return Err(PandRSError::Column(format!("列名 '{}' は既に存在します", new_name)));
            }
            
            // 列データをコピー
            if let Some(col_data) = self.columns.get(old_name) {
                new_columns.insert(new_name.clone(), col_data.clone());
                new_column_names.push(new_name);
            }
        }
        
        // 新しい列データで更新
        self.columns = new_columns;
        self.column_names = new_column_names;
        
        Ok(())
    }
    
    /// 文字列インデックスでDataFrameを作成
    pub fn with_index(index: Index<String>) -> Self  
    {
        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index: DataFrameIndex::<String>::from_simple(index),
        }
    }
    
    /// マルチインデックスでDataFrameを作成
    pub fn with_multi_index(index: crate::index::MultiIndex<String>) -> Self  
    {
        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index: DataFrameIndex::<String>::from_multi(index),
        }
    }
    
    /// 整数インデックスでDataFrameを作成
    pub fn with_range_index(range: std::ops::Range<usize>) -> Result<Self> {
        let range_idx = Index::<usize>::from_range(range)?;
        // 数値インデックスを文字列インデックスに変換
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values)?;

        Ok(DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index: DataFrameIndex::<String>::from_simple(string_idx),
        })
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
            let range_idx = Index::<usize>::from_range(0..boxed_series.len())?;
            let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
            let string_idx = Index::<String>::new(string_values)?;
            self.index = DataFrameIndex::<String>::from_simple(string_idx);
        }

        // 列を追加
        self.columns.insert(name.clone(), boxed_series);
        self.column_names.push(name);

        Ok(())
    }
    
    /// 文字列インデックスを設定
    pub fn set_index(&mut self, index: Index<String>) -> Result<()>
    {
        // インデックス長チェック
        if !self.column_names.is_empty() && index.len() != self.index.len() {
            return Err(PandRSError::Consistency(format!(
                "新しいインデックスの長さ ({}) がDataFrameの行数 ({}) と一致しません",
                index.len(),
                self.index.len()
            )));
        }
        
        // インデックスを設定
        self.index = DataFrameIndex::<String>::from_simple(index);
        
        Ok(())
    }
    
    /// マルチインデックスを設定
    pub fn set_multi_index(&mut self, index: crate::index::MultiIndex<String>) -> Result<()>
    {
        // インデックス長チェック
        if !self.column_names.is_empty() && index.len() != self.index.len() {
            return Err(PandRSError::Consistency(format!(
                "新しいマルチインデックスの長さ ({}) がDataFrameの行数 ({}) と一致しません",
                index.len(),
                self.index.len()
            )));
        }
        
        // インデックスを設定
        self.index = DataFrameIndex::<String>::from_multi(index);
        
        Ok(())
    }
    
    /// インデックスを取得
    pub fn get_index(&self) -> &DataFrameIndex<String> {
        &self.index
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
    
    /// 列をインデックスとして設定
    pub fn set_column_as_index(&mut self, column_name: &str) -> Result<()> {
        // 列が存在するかチェック
        if !self.contains_column(column_name) {
            return Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                column_name
            )));
        }
        
        // 列の値を取得
        let values = self.get_column_string_values(column_name)?;
        
        // 文字列インデックスを作成
        let string_index = Index::new(values)?;
        
        // インデックスを設定
        self.index = DataFrameIndex::<String>::from_simple(string_index);
        
        Ok(())
    }
    
    /// 列から取得する関数を追加
    pub fn get_column(&self, name: &str) -> Option<Series<String>> {
        if let Some(series) = self.columns.get(name) {
            // DataBox型のSeriesから文字列型のSeriesに変換
            let string_values: Vec<String> = series
                .values()
                .iter()
                .map(|v| format!("{:?}", v))
                .collect();
                
            Series::new(string_values, Some(name.to_string())).ok()
        } else {
            None
        }
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
    
    /// DataFrameをJSON文字列に変換
    pub fn to_json(&self) -> Result<String> {
        let mut obj = serde_json::Map::new();
        
        // 各列をJSONに変換
        for col in self.column_names() {
            let values = self.get_column(col).unwrap_or_default();
            obj.insert(col.clone(), serde_json::Value::Array(
                values.values().iter().map(|v| serde_json::Value::String(v.clone())).collect()
            ));
        }
        
        // インデックスがあれば追加
        if let Some(index_values) = self.index.string_values() {
            obj.insert("index".to_string(), serde_json::Value::Array(
                index_values.iter().map(|v| serde_json::Value::String(v.clone())).collect()
            ));
        }
        
        Ok(serde_json::to_string(&obj)?)
    }
    
    /// JSON文字列からDataFrameを生成
    pub fn from_json(json: &str) -> Result<Self> {
        let parsed: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json)?;
        let mut data = HashMap::new();
        let mut index_values = None;
        
        // 各フィールドを処理
        for (key, value) in parsed.iter() {
            if key == "index" {
                if let serde_json::Value::Array(arr) = value {
                    index_values = Some(
                        arr.iter()
                           .map(|v| match v {
                               serde_json::Value::String(s) => s.clone(),
                               _ => v.to_string(),
                           })
                           .collect()
                    );
                }
                continue;
            }
            
            if let serde_json::Value::Array(arr) = value {
                let values: Vec<String> = arr.iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .collect();
                data.insert(key.clone(), values);
            }
        }
        
        Self::from_map(data, index_values)
    }
    
    /// マップ（HashMap）からDataFrameを作成
    pub fn from_map(data: HashMap<String, Vec<String>>, index: Option<Vec<String>>) -> Result<Self> {
        let mut df = Self::new();
        
        // 行数を確認（すべての列が同じ長さであることを確認）
        let row_count = if !data.is_empty() {
            data.values().next().unwrap().len()
        } else {
            0
        };
        
        for (col_name, values) in data {
            if values.len() != row_count {
                return Err(PandRSError::Consistency(format!(
                    "列 '{}' の長さ ({}) が他の列の長さ ({}) と一致しません",
                    col_name, values.len(), row_count
                )));
            }
            
            df.add_column(col_name, Series::new(values, None)?)?;
        }
        
        // インデックスがあれば設定
        if let Some(idx_values) = index {
            if idx_values.len() == row_count {
                let idx = Index::new(idx_values)?;
                df.set_index(idx)?;
            } else {
                return Err(PandRSError::Consistency(format!(
                    "インデックスの長さ ({}) がデータの行数 ({}) と一致しません",
                    idx_values.len(), row_count
                )));
            }
        }
        
        Ok(df)
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
