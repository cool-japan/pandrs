// optimized/dataframe.rs (修正版)

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::column::{Column, ColumnTrait, ColumnType, Int64Column};
use crate::column::string_column::{StringColumn, StringColumnOptimizationMode};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait, Index};
use crate::optimized::operations::JoinType;
use crate::series::CategoricalOrder;


/// 最適化されたデータフレーム構造体
#[derive(Clone)]
pub struct OptimizedDataFrame {
    /// 列データ
    columns: Vec<Column>,
    /// 列名から列インデックスへのマッピング
    column_indices: HashMap<String, usize>,
    /// 列名のリスト
    column_names: Vec<String>,
    /// 行数
    row_count: usize,
    /// データフレームのインデックス (オプショナル)
    index: Option<DataFrameIndex<String>>,
}

/// 列への読み取り専用ビュー
pub struct ColumnView<'a> {
    /// 参照する列
    column: &'a Column,
}

impl Debug for OptimizedDataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 表示する最大行数
        const MAX_ROWS: usize = 10;

        if self.columns.is_empty() {
            return write!(f, "OptimizedDataFrame (0 rows x 0 columns)");
        }

        writeln!(f, "OptimizedDataFrame ({} rows x {} columns):", self.row_count, self.columns.len())?;

        // ヘッダー行の表示
        write!(f, "{:<5} |", "idx")?; // インデックス列ヘッダー
        for name in &self.column_names {
            write!(f, " {:<15} |", name)?; // 各列ヘッダー
        }
        writeln!(f)?;

        // 区切り線の表示
        write!(f, "{:-<5}-+", "")?;
        for _ in &self.column_names {
            write!(f, "-{:-<15}-+", "")?;
        }
        writeln!(f)?;

        // データ行の表示 (最大MAX_ROWS行)
        let display_rows = std::cmp::min(self.row_count, MAX_ROWS);
        for i in 0..display_rows {
            // インデックス値の表示
            if let Some(ref idx) = self.index {
                let idx_value = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => {
                        // simple_idxの範囲内かチェック
                        if i < simple_idx.len() {
                            simple_idx.get_value(i).map(|s| s.to_string()).unwrap_or_else(|| i.to_string())
                        } else {
                            i.to_string() // 範囲外の場合は行番号を使用
                        }
                    },
                    DataFrameIndex::Multi(_) => i.to_string() // MultiIndexの場合は行番号を使用
                };
                write!(f, "{:<5} |", idx_value)?;
            } else {
                write!(f, "{:<5} |", i)?; // インデックスがない場合は行番号
            }

            // 各列の値の表示
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
                            format!("{:.3}", val) // 小数点以下3桁で表示
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("\"{}\"", val) // 文字列はダブルクォートで囲む
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
                write!(f, " {:<15} |", value)?; // 左寄せ15文字幅で表示
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

// カテゴリカルデータ用のメタデータキー
const CATEGORICAL_META_KEY: &str = "_categorical";
const CATEGORICAL_ORDER_META_KEY: &str = "_categorical_order";

/// 重複行の保持戦略
#[derive(Debug, Clone, Copy, PartialEq)]
enum KeepStrategy {
    /// 最初の行を保持
    First,
    /// 最後の行を保持
    Last,
    /// 全ての重複行を削除 (保持しない)
    All, // Note: 'All' is typically used to mark all duplicates as True, not to keep all. This might need clarification based on pandas behavior.
}

/// GroupBy操作のための構造体
pub struct GroupBy<'a> {
    /// 対象のデータフレーム
    df: &'a OptimizedDataFrame,
    /// グループ化する列名
    by: String,
}

impl<'a> GroupBy<'a> {
    /// 新しいGroupByオブジェクトを作成
    fn new(df: &'a OptimizedDataFrame, by: &str) -> Self {
        Self {
            df,
            by: by.to_string(),
        }
    }

    /// 指定された列の合計値を計算
    pub fn sum(&self, columns: &[&str]) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();

        // グループごとの合計値を格納するHashMap
        // Key: グループ化キーの値 (String), Value: (列名 -> 合計値) のHashMap
        let mut groups: HashMap<String, HashMap<String, f64>> = HashMap::new();

        // グループ化列のインデックスと列データを取得
        let by_idx = self.df.column_indices[&self.by];
        let by_col = &self.df.columns[by_idx];

        // 集計対象列のインデックスを取得
        let mut column_indices = Vec::new();
        for &col_name in columns {
            if let Some(&idx) = self.df.column_indices.get(col_name) {
                column_indices.push((col_name, idx));
            } else {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }

        // 各行を処理してグループ化と集計
        for row_idx in 0..self.df.row_count {
            // グループ化キーの値を取得
            let group_key = match by_col {
                Column::Int64(col) => {
                    if let Ok(Some(v)) = col.get(row_idx) { v.to_string() } else { "NULL".to_string() }
                },
                Column::Float64(col) => {
                    if let Ok(Some(v)) = col.get(row_idx) { v.to_string() } else { "NULL".to_string() }
                },
                Column::String(col) => {
                    if let Ok(Some(v)) = col.get(row_idx) { v.to_string() } else { "NULL".to_string() }
                },
                Column::Boolean(col) => {
                    if let Ok(Some(v)) = col.get(row_idx) { v.to_string() } else { "NULL".to_string() }
                },
            };

            // グループの合計値マップを取得または新規作成
            let group_sums = groups.entry(group_key).or_insert_with(|| {
                let mut sums = HashMap::new();
                for &(col_name, _) in &column_indices {
                    sums.insert(col_name.to_string(), 0f64);
                }
                sums
            });

            // 集計対象列の値を合計に加算
            for &(col_name, col_idx) in &column_indices {
                let col = &self.df.columns[col_idx];
                let value = match col {
                    Column::Int64(c) => if let Ok(Some(v)) = c.get(row_idx) { v as f64 } else { 0.0 },
                    Column::Float64(c) => if let Ok(Some(v)) = c.get(row_idx) { v } else { 0.0 },
                    Column::Boolean(c) => if let Ok(Some(v)) = c.get(row_idx) { if v { 1.0 } else { 0.0 } } else { 0.0 },
                    _ => 0.0, // 文字列などは合計対象外
                };
                *group_sums.get_mut(col_name).unwrap() += value;
            }
        }

        // 結果のDataFrameを構築
        let mut group_keys = Vec::new();
        let mut group_values_list = Vec::new(); // 各グループの集計結果を保持

        for (key, sums) in groups {
            group_keys.push(key);
            group_values_list.push(sums);
        }

        // グループ化キー列を追加
        let group_key_col = crate::column::StringColumn::new(group_keys);
        result.add_column(self.by.clone(), Column::String(group_key_col))?; // Use Column::String

        // 集計結果列を追加
        for &(col_name, _) in &column_indices {
            let mut values = Vec::new();
            for sums in &group_values_list {
                values.push(*sums.get(col_name).unwrap_or(&0.0));
            }
            let sum_col = crate::column::Float64Column::new(values);
            let sum_col_name = format!("sum_{}", col_name);
            result.add_column(sum_col_name, Column::Float64(sum_col))?; // Use Column::Float64
        }

        Ok(result)
    }
}


impl OptimizedDataFrame {
    /// 新しい空のOptimizedDataFrameを作成
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: 0,
            index: None,
        }
    }

    /// 列を追加
    pub fn add_column<C: Into<Column>>(&mut self, name: impl Into<String>, column: C) -> Result<()> {
        let name = name.into();
        let column = column.into();

        // 列名の重複チェック
        if self.column_indices.contains_key(&name) {
            return Err(Error::DuplicateColumnName(name));
        }

        // 行数の一貫性チェック
        let column_len = column.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }

        // 列データを追加
        let column_idx = self.columns.len();
        self.columns.push(column);
        self.column_indices.insert(name.clone(), column_idx);
        self.column_names.push(name);

        // DataFrameの行数を更新 (初回追加時)
        if self.row_count == 0 {
            self.row_count = column_len;
        }

        Ok(())
    }

    /// 指定された名前の列へのビューを取得
    pub fn column(&self, name: &str) -> Result<ColumnView<'_>> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        let column = &self.columns[*column_idx];
        Ok(ColumnView { column })
    }

    /// `column` のエイリアス
    pub fn get_column(&self, name: &str) -> Result<ColumnView<'_>> {
        self.column(name)
    }

    /// 各列に関数を適用 (逐次処理)
    pub fn apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView<'_>) -> Result<Column>, // クロージャは列ビューを受け取り、新しい列を返す
    {
        let mut result = Self::new();
        for (idx, col_name) in self.column_names.iter().enumerate() {
            let col = &self.columns[idx];
            let view = ColumnView { column: col };
            let new_col = func(&view)?; // 関数を適用
            result.add_column(col_name.clone(), new_col)?; // 結果の列を追加
        }
         // インデックスをコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        Ok(result)
    }

    /// DataFrameをCSVファイルに書き出す
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        let file = std::fs::File::create(path_ref)
            .map_err(|e| Error::IoError(format!("CSVファイルの作成に失敗: {}", e)))?;
        let mut writer = csv::Writer::from_writer(file);

        // ヘッダーを書き込み
        writer.write_record(&self.column_names)
            .map_err(|e| Error::CsvError(format!("CSVヘッダーの書き込みに失敗: {}", e)))?;

        // 各行のデータを書き込み
        for row_idx in 0..self.row_count {
            let mut row = Vec::with_capacity(self.column_names.len());
            for col_name in &self.column_names {
                let col_idx = self.column_indices[col_name];
                let col = &self.columns[col_idx];
                // 値を文字列に変換して追加
                let value = match col {
                    Column::Int64(int_col) => int_col.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                    Column::Float64(float_col) => float_col.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                    Column::String(string_col) => string_col.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                    Column::Boolean(bool_col) => bool_col.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                };
                row.push(value);
            }
            writer.write_record(&row)
                .map_err(|e| Error::CsvError(format!("CSVデータの書き込みに失敗: {}", e)))?;
        }

        writer.flush()
            .map_err(|e| Error::IoError(format!("CSVファイルのフラッシュに失敗: {}", e)))?;
        Ok(())
    }

    /// 指定された列の型を取得
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        Ok(self.columns[*column_idx].column_type())
    }

    /// 列名のスライスを取得
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// 行数を取得
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// 列数を取得
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// 指定された名前の列が存在するかどうかを確認
    pub fn contains_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }

    /// 指定された列をインデックスとして設定
    /// 現在は String または Int64 列のみサポート
    pub fn set_index(&mut self, name: &str) -> Result<()> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        let column = &self.columns[*column_idx];

        // 列の値をインデックス用の文字列ベクターに変換
        let index_values: Result<Vec<String>> = match column {
            Column::String(string_col) => {
                (0..string_col.len())
                    .map(|i| string_col.get(i).map(|opt_s| opt_s.map(|s| s.to_string()).unwrap_or_else(|| i.to_string()))) // Noneは行番号で代替
                    .collect()
            },
            Column::Int64(int_col) => {
                 (0..int_col.len())
                    .map(|i| int_col.get(i).map(|opt_i| opt_i.map(|i_val| i_val.to_string()).unwrap_or_else(|| i.to_string())))
                    .collect()
            },
            // 他の型は現在サポート外
            _ => return Err(Error::OperationFailed(format!(
                "列 '{}' (型: {:?}) はインデックスとして使用できません", name, column.column_type()
            ))),
        };

        let index = Index::with_name(index_values?, Some(name.to_string()))?;
        self.index = Some(DataFrameIndex::from_simple(index));
        Ok(())
    }

    /// 指定されたインデックスの行を取得 (新しいDataFrameとして返す)
    pub fn get_row(&self, row_idx: usize) -> Result<Self> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds { index: row_idx, size: self.row_count });
        }

        let mut result = Self::new();
        for (i, name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];
            // 各列から指定された行の値を取得し、長さ1の新しい列を作成
            let new_column = match column {
                 Column::Int64(col) => Column::Int64(Int64Column::new(vec![col.get(row_idx)?.unwrap_or_default()])),
                 Column::Float64(col) => Column::Float64(crate::column::Float64Column::new(vec![col.get(row_idx)?.unwrap_or_default()])),
                 Column::String(col) => Column::String(crate::column::StringColumn::new(vec![col.get(row_idx)?.unwrap_or_default().to_string()])),
                 Column::Boolean(col) => Column::Boolean(crate::column::BooleanColumn::new(vec![col.get(row_idx)?.unwrap_or_default()])),
            };
            result.add_column(name.clone(), new_column)?;
        }
         // 元のインデックスがあれば、対応するインデックス値を設定
        if let Some(ref original_index) = self.index {
             match original_index {
                DataFrameIndex::Simple(simple_idx) => {
                    if row_idx < simple_idx.len() {
                         if let Some(idx_val) = simple_idx.get_value(row_idx) {
                             let new_idx = Index::with_name(vec![idx_val.to_string()], simple_idx.name().cloned())?;
                             result.index = Some(DataFrameIndex::Simple(new_idx));
                         }
                    }
                },
                DataFrameIndex::Multi(_) => { /* MultiIndexの行取得は複雑 */ }
            }
        }
        Ok(result)
    }

    /// 指定された列のみを選択して新しいDataFrameを作成
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();
        for &name in columns {
            let column_idx = self.column_indices.get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
            // 列データをクローンして追加
            result.add_column(name.to_string(), self.columns[*column_idx].clone())?;
        }
        // インデックスもクローン
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        Ok(result)
    }

    /// ブール列に基づいて行をフィルタリング (逐次処理)
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        let column_idx = self.column_indices.get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;
        let condition = &self.columns[*column_idx];

        // 条件列がブール型であることを確認
        if let Column::Boolean(bool_col) = condition {
            // 条件がtrueの行インデックスを収集
            let indices: Vec<usize> = (0..bool_col.len())
                .filter_map(|i| bool_col.get(i).ok().flatten().filter(|&b| b).map(|_| i))
                .collect();

            // フィルタリングされたDataFrameを作成
            self.filter_by_indices(&indices)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }

    /// 各列に関数を並列適用 (Rayonを使用)
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send, // クロージャはSync + Sendである必要あり
    {
        use rayon::prelude::*;

        // 各列に対するビューを作成
        let column_views: Vec<_> = self.column_names.iter()
            .map(|name| self.column(name))
            .collect::<Result<_>>()?;

        // Rayonを使って並列に関数を適用し、新しい列を収集
        let new_columns: Result<Vec<_>> = column_views.par_iter()
            .map(|view| func(view))
            .collect();

        // 結果のDataFrameを構築
        let new_columns = new_columns?;
        let mut result = Self::new();
        for (name, column) in self.column_names.iter().zip(new_columns) {
            result.add_column(name.clone(), column)?;
        }
        // インデックスをコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        Ok(result)
    }

    /// ブール列に基づいて行を並列フィルタリング (Rayonを使用)
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        use rayon::prelude::*;

        // 並列処理の閾値 (この行数以下なら逐次処理)
        const PARALLEL_THRESHOLD: usize = 100_000;

        let column_idx = self.column_indices.get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;
        let condition = &self.columns[*column_idx];

        if let Column::Boolean(bool_col) = condition {
            let row_count = bool_col.len();

            // 条件がtrueの行インデックスを収集 (並列または逐次)
            let indices: Vec<usize> = if row_count < PARALLEL_THRESHOLD {
                (0..row_count)
                    .filter_map(|i| bool_col.get(i).ok().flatten().filter(|&b| b).map(|_| i))
                    .collect()
            } else {
                // Rayonを使って並列にフィルタリング
                 (0..row_count).into_par_iter()
                    .filter_map(|i| bool_col.get(i).ok().flatten().filter(|&b| b).map(|_| i))
                    .collect()
            };

            // フィルタリングされたDataFrameを作成 (並列)
            self.filter_by_indices_parallel(&indices) // 並列版のフィルタリング関数を呼ぶ
        } else {
            Err(Error::OperationFailed(format!(
                "列 '{}' はブール型ではありません", condition_column
            )))
        }
    }

    /// グループ化キーに基づいてDataFrameを複数のグループに分割 (並列処理)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        use rayon::prelude::*;
        use std::collections::hash_map::Entry;
        use std::sync::{Arc, Mutex};

        // 並列処理の閾値
        const PARALLEL_THRESHOLD: usize = 50_000;

        // グループ化列のインデックスを取得
        let mut group_col_indices = Vec::with_capacity(group_by_columns.len());
        for &col_name in group_by_columns {
            let col_idx = self.column_indices.get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            group_col_indices.push(*col_idx);
        }

        // --- 1. グループキーと対応する行インデックスのマップを作成 (並列または逐次) ---
        let groups: HashMap<String, Vec<usize>> = if self.row_count < PARALLEL_THRESHOLD {
            // 逐次処理
            let mut groups = HashMap::new();
            for row_idx in 0..self.row_count {
                let group_key = self.get_group_key(row_idx, &group_col_indices)?;
                match groups.entry(group_key) {
                    Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                    Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                }
            }
            groups
        } else {
            // 並列処理
            // 各スレッドでローカルなHashMapを作成し、最後にマージする
            (0..self.row_count).into_par_iter()
                .fold(
                    || HashMap::<String, Vec<usize>>::new(), 
                    |mut local_map, row_idx| {
                        if let Ok(group_key) = self.get_group_key(row_idx, &group_col_indices) {
                            match local_map.entry(group_key) {
                                Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                                Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                            }
                        }
                        local_map
                    }
                )
                .reduce(
                    || HashMap::<String, Vec<usize>>::new(),
                    |mut map1, map2| {
                        for (key, indices) in map2 {
                            match map1.entry(key) {
                                Entry::Vacant(e) => { e.insert(indices); },
                                Entry::Occupied(mut e) => { e.get_mut().extend(indices); }
                            }
                        }
                        map1
                    }
                )
        };

        // --- 2. 各グループのDataFrameを作成 (並列または逐次) ---
        let result = if groups.len() < 100 || self.row_count < PARALLEL_THRESHOLD {
            // 逐次処理
            let mut result_map = HashMap::with_capacity(groups.len());
            for (key, indices) in groups {
                // filter_by_indices を使う (逐次版)
                let group_df = self.filter_by_indices(&indices)?;
                result_map.insert(key, group_df);
            }
            result_map
        } else {
            // 並列処理
            let result_mutex = Arc::new(Mutex::new(HashMap::with_capacity(groups.len())));
            let group_items: Vec<(String, Vec<usize>)> = groups.into_iter().collect();

            group_items.par_iter()
                .for_each(|(key, indices)| {
                    // filter_by_indices_parallel を使う (並列版)
                    if let Ok(group_df) = self.filter_by_indices_parallel(indices) {
                         if let Ok(mut result_guard) = result_mutex.lock() {
                             result_guard.insert(key.clone(), group_df);
                         }
                    }
                });

            // Mutexから結果を取り出す
            match Arc::try_unwrap(result_mutex) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(_) => HashMap::new(), // エラーの場合は空
            }
        };

        Ok(result)
    }

    /// 指定された行インデックスに基づいて新しいDataFrameを作成 (逐次処理)
    fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let mut result = Self::new();
        if indices.is_empty() {
            // 空のDataFrameを返す
            for name in &self.column_names {
                let col_idx = self.column_indices[name];
                let empty_col = self.columns[col_idx].empty_clone();
                result.add_column(name.clone(), empty_col)?;
            }
            return Ok(result);
        }

        // 各列をフィルタリング
        for name in &self.column_names {
            let i = self.column_indices[name];
            let column = &self.columns[i];
            let filtered_column = column.filter_by_indices(indices)?;
            result.add_column(name.clone(), filtered_column)?;
        }

        // インデックスもフィルタリング
        if let Some(ref idx) = self.index {
             match idx {
                DataFrameIndex::Simple(simple_idx) => {
                    let new_index_values: Vec<String> = indices.iter()
                        .filter_map(|&i| simple_idx.get_value(i).map(|s| s.to_string()))
                        .collect();
                    if new_index_values.len() == result.row_count {
                        let new_idx = Index::with_name(new_index_values, simple_idx.name().cloned())?;
                        result.index = Some(DataFrameIndex::Simple(new_idx));
                    } else {
                         result.index = None; // 長さが合わない場合はリセット
                    }
                },
                DataFrameIndex::Multi(_) => { result.index = None; } // MultiIndexはリセット
            }
        }

        Ok(result)
    }

     /// 指定された行インデックスに基づいて新しいDataFrameを作成 (並列処理)
    fn filter_by_indices_parallel(&self, indices: &[usize]) -> Result<Self> {
        use rayon::prelude::*;
        let mut result = Self::new();
         if indices.is_empty() {
            // 空のDataFrameを返す
            for name in &self.column_names {
                let col_idx = self.column_indices[name];
                let empty_col = self.columns[col_idx].empty_clone();
                result.add_column(name.clone(), empty_col)?;
            }
            return Ok(result);
        }

        // 各列を並列にフィルタリング
        let column_results: Result<Vec<(String, Column)>> = self.column_names.par_iter()
            .map(|name| {
                let i = self.column_indices[name];
                let column = &self.columns[i];
                let filtered_column = column.filter_by_indices(indices)?; // ColumnTraitのfilter_by_indicesを呼ぶ
                Ok((name.clone(), filtered_column))
            })
            .collect();

        // 結果をDataFrameに追加
        for (name, column) in column_results? {
            result.add_column(name, column)?;
        }

        // インデックスもフィルタリング (逐次処理でOK)
        if let Some(ref idx) = self.index {
            match idx {
                DataFrameIndex::Simple(simple_idx) => {
                    let new_index_values: Vec<String> = indices.iter()
                        .filter_map(|&i| simple_idx.get_value(i).map(|s| s.to_string()))
                        .collect();
                     if new_index_values.len() == result.row_count {
                        let new_idx = Index::with_name(new_index_values, simple_idx.name().cloned())?;
                        result.index = Some(DataFrameIndex::Simple(new_idx));
                    } else {
                         result.index = None;
                    }
                },
                DataFrameIndex::Multi(_) => { result.index = None; }
            }
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

    /// 結合の内部実装
    fn join_impl(&self, other: &Self, left_on: &str, right_on: &str, join_type: JoinType) -> Result<Self> {
        // 結合キー列を取得
        let left_col_idx = self.column_indices.get(left_on).ok_or_else(|| Error::ColumnNotFound(left_on.to_string()))?;
        let right_col_idx = other.column_indices.get(right_on).ok_or_else(|| Error::ColumnNotFound(right_on.to_string()))?;
        let left_col = &self.columns[*left_col_idx];
        let right_col = &other.columns[*right_col_idx];

        // 結合キーの型チェック (現状は文字列化して比較するため、厳密な型チェックは省略可能だが、パフォーマンス考慮なら必要)
        // if left_col.column_type() != right_col.column_type() { ... }

        // --- 1. 右側のDataFrameのキーと行インデックスのマップを作成 ---
        let mut right_key_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..other.row_count() {
            let key = other.get_value_as_string(*right_col_idx, i)?.unwrap_or_else(|| "NULL".to_string());
            right_key_to_indices.entry(key).or_default().push(i);
        }

        // --- 2. 結合結果の行インデックスペア (Option<usize>, Option<usize>) を作成 ---
        let mut join_indices: Vec<(Option<usize>, Option<usize>)> = Vec::new();
        let mut left_matched = vec![false; self.row_count()]; // 左側のマッチ状況 (Right/Outer用)

        for i in 0..self.row_count() {
            let key = self.get_value_as_string(*left_col_idx, i)?.unwrap_or_else(|| "NULL".to_string());
            if let Some(right_indices) = right_key_to_indices.get(&key) {
                // マッチした場合
                left_matched[i] = true;
                for &right_idx in right_indices {
                    join_indices.push((Some(i), Some(right_idx)));
                }
            } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                // 左結合または外部結合でマッチしなかった場合
                join_indices.push((Some(i), None));
                 left_matched[i] = true; // 左側は含まれる
            }
        }

        // --- 3. 右結合または外部結合のために、右側でマッチしなかった行を追加 ---
        if join_type == JoinType::Right || join_type == JoinType::Outer {
            let mut right_matched = vec![false; other.row_count()];
            for (_, right_idx_opt) in &join_indices {
                if let Some(right_idx) = right_idx_opt {
                    right_matched[*right_idx] = true;
                }
            }
            for i in 0..other.row_count() {
                if !right_matched[i] {
                    join_indices.push((None, Some(i)));
                }
            }
        }

        // --- 4. 結果のDataFrameを構築 ---
        let mut result = Self::new();
        if join_indices.is_empty() {
            // 結合結果が空の場合、空の列を持つDataFrameを返す
             for name in &self.column_names {
                 let col_idx = self.column_indices[name];
                 result.add_column(name.clone(), self.columns[col_idx].empty_clone())?;
             }
             for name in &other.column_names {
                 if name != right_on { // 結合キー列は含めない (左側で追加されるため)
                     let new_name = if result.contains_column(name) { format!("{}_right", name) } else { name.clone() };
                     let col_idx = other.column_indices[name];
                     result.add_column(new_name, other.columns[col_idx].empty_clone())?;
                 }
             }
            return Ok(result);
        }

        let result_row_count = join_indices.len();

        // 左側の列を追加
        for name in &self.column_names {
            let col_idx = self.column_indices[name];
            let col = &self.columns[col_idx];
            let mut builder = ColumnBuilder::new();
            for (left_idx_opt, _) in &join_indices {
                if let Some(left_idx) = left_idx_opt {
                     builder.append_value(col, *left_idx)?;
                } else {
                     builder.append_na(1)?; // 左側がない場合はNAを追加
                }
            }
            result.add_column(name.clone(), builder.build()?)?;
        }

        // 右側の列を追加 (重複名はリネーム)
        for name in &other.column_names {
            if name != right_on { // 結合キー列はスキップ
                let new_name = if result.contains_column(name) { format!("{}_right", name) } else { name.clone() };
                let col_idx = other.column_indices[name];
                let col = &other.columns[col_idx];
                let mut builder = ColumnBuilder::new();
                 for (_, right_idx_opt) in &join_indices {
                    if let Some(right_idx) = right_idx_opt {
                         builder.append_value(col, *right_idx)?;
                    } else {
                         builder.append_na(1)?; // 右側がない場合はNAを追加
                    }
                }
                result.add_column(new_name, builder.build()?)?;
            }
        }

        // インデックスはリセット
        result.index = None;

        Ok(result)
    }

    /// 指定した行インデックスと列インデックスの値を取得 (文字列として)
    fn get_value_as_string(&self, col_idx: usize, row_idx: usize) -> Result<Option<String>> {
        if col_idx >= self.columns.len() || row_idx >= self.row_count {
            return Ok(None); // 範囲外アクセス防止
        }
        let col = &self.columns[col_idx];
        let value_str = match col {
            Column::Int64(c) => c.get(row_idx)?.map(|v| v.to_string()),
            Column::Float64(c) => c.get(row_idx)?.map(|v| v.to_string()),
            Column::String(c) => c.get(row_idx)?.map(|s| s.to_string()),
            Column::Boolean(c) => c.get(row_idx)?.map(|v| v.to_string()),
        };
        Ok(value_str)
    }

    /// 指定した行のグループ化キーを生成
    fn get_group_key(&self, row_idx: usize, group_col_indices: &[usize]) -> Result<String> {
        let mut key_parts = Vec::with_capacity(group_col_indices.len());
        for &col_idx in group_col_indices {
             let part = self.get_value_as_string(col_idx, row_idx)?.unwrap_or_else(|| "NA".to_string());
             key_parts.push(part);
        }
        Ok(key_parts.join("_")) // キーを結合して返す
    }

    // --- 以下は DataFrameCompat トレイトから移動した、または関連するメソッド ---

    /// 指定された列の値の出現回数をカウントする内部実装
    fn value_counts_impl(&self, column: &str) -> Result<crate::series::Series<usize>> {
        if !self.column_indices.contains_key(column) {
            return Err(Error::ColumnNotFound(column.to_string()));
        }

        let col_idx = self.column_indices[column];
        let col = &self.columns[col_idx];

        // 値ごとのカウントを格納するHashMap
        let mut counts = HashMap::new();

        // 各行の値をカウント
        for i in 0..self.row_count {
            let value_str = match col {
                Column::Int64(int_col) => if let Ok(Some(v)) = int_col.get(i) { v.to_string() } else { "NULL".to_string() },
                Column::Float64(float_col) => if let Ok(Some(v)) = float_col.get(i) { v.to_string() } else { "NULL".to_string() },
                Column::String(string_col) => if let Ok(Some(v)) = string_col.get(i) { v.to_string() } else { "NULL".to_string() },
                Column::Boolean(bool_col) => if let Ok(Some(v)) = bool_col.get(i) { v.to_string() } else { "NULL".to_string() },
            };
            *counts.entry(value_str).or_insert(0) += 1;
        }

        // カウント結果を降順にソート
        let mut items: Vec<(String, usize)> = counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1)); // カウント数で降順ソート

        // Seriesのインデックスと値を作成
        let unique_values: Vec<String> = items.iter().map(|(k, _)| k.clone()).collect();
        let count_values: Vec<usize> = items.iter().map(|(_, v)| *v).collect();

        // Seriesのインデックスを作成
        let index = crate::index::Index::new(unique_values)?;

        // Seriesを作成して返す
        crate::series::Series::with_index(count_values, index, Some("count".to_string()))
    }

    /// 重複行を削除
    ///
    /// # Arguments
    /// * `subset` - 重複判定に使用する列名のスライス (Noneの場合は全列)
    /// * `keep` - 保持する行 ('first', 'last', 'all' のいずれか、デフォルトは 'first')
    pub fn drop_duplicates(&self, subset: Option<&[&str]>, keep: Option<&str>) -> Result<Self> {
        // 保持戦略を決定
        let keep_strategy = match keep.unwrap_or("first") {
            "first" => KeepStrategy::First,
            "last" => KeepStrategy::Last,
            // "all" は通常、重複するものをすべて削除する意味ではないため注意
            "all" => return Err(Error::invalid_argument("'all' keep strategy is not directly supported for dropping, did you mean marking duplicates?".to_string())),
            _ => return Err(Error::invalid_argument("keep引数は'first'または'last'のいずれかでなければなりません".to_string())),
        };

        // 重複判定に使用する列
        let columns_to_check = match subset {
            Some(cols) => cols.to_vec(),
            None => self.column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        };

        let mut result = Self::new();

        // 見たキーとその行インデックスを格納 (keep='last' のためにインデックスを保持)
        let mut seen: HashMap<String, usize> = HashMap::new();
        // 保持する行のインデックス
        let mut keep_indices = Vec::new();

        // 各行を処理
        for row_idx in 0..self.row_count {
            // 重複判定用のキーを作成
            let mut key = String::new();
            for &col_name in &columns_to_check {
                if let Some(&col_idx) = self.column_indices.get(col_name) {
                    let col = &self.columns[col_idx];
                    let val = match col {
                        Column::Int64(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                        Column::Float64(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                        Column::String(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                        Column::Boolean(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                    };
                    key.push_str(&val);
                } else {
                     return Err(Error::ColumnNotFound(col_name.to_string())); // subsetに指定された列が存在しない場合
                }
            }

            // 保持するかどうかを判定
            let is_duplicate = seen.contains_key(&key);

            match keep_strategy {
                KeepStrategy::First => {
                    if !is_duplicate {
                        seen.insert(key, row_idx);
                        keep_indices.push(row_idx);
                    }
                }
                KeepStrategy::Last => {
                    // 以前に同じキーがあれば、keep_indicesから古いインデックスを削除
                    if let Some(old_idx) = seen.insert(key, row_idx) {
                        if let Some(pos) = keep_indices.iter().position(|&x| x == old_idx) {
                            keep_indices.remove(pos);
                        }
                    }
                    // 新しいインデックスを追加
                    keep_indices.push(row_idx);
                }
                 KeepStrategy::All => { /* Not applicable for drop_duplicates */ }
            }
        }

        // 結果のDataFrameを構築
        for col_name in &self.column_names {
            if let Some(&col_idx) = self.column_indices.get(col_name) {
                let col = &self.columns[col_idx];

                // 保持するインデックスに基づいて新しい列データを作成
                match col {
                    Column::Int64(int_col) => {
                        let new_data: Vec<i64> = keep_indices.iter().map(|&idx| int_col.get(idx).ok().flatten().unwrap_or(0)).collect();
                        let mut new_col = crate::column::Int64Column::new(new_data);
                        new_col.name = Some(col_name.clone());
                        result.add_column(col_name.clone(), Column::Int64(new_col))?;
                    },
                    Column::Float64(float_col) => {
                        let new_data: Vec<f64> = keep_indices.iter().map(|&idx| float_col.get(idx).ok().flatten().unwrap_or(0.0)).collect();
                        let mut new_col = crate::column::Float64Column::new(new_data);
                        new_col.name = Some(col_name.clone());
                        result.add_column(col_name.clone(), Column::Float64(new_col))?;
                    },
                    Column::String(string_col) => {
                        let new_data: Vec<String> = keep_indices.iter().map(|&idx| string_col.get(idx).ok().flatten().map(|s| s.to_string()).unwrap_or_default()).collect();
                        let mut new_col = crate::column::StringColumn::new(new_data);
                        new_col.name = Some(col_name.clone());
                        result.add_column(col_name.clone(), Column::String(new_col))?;
                    },
                    Column::Boolean(bool_col) => {
                        let new_data: Vec<bool> = keep_indices.iter().map(|&idx| bool_col.get(idx).ok().flatten().unwrap_or(false)).collect();
                        let mut new_col = crate::column::BooleanColumn::new(new_data);
                        new_col.name = Some(col_name.clone());
                        result.add_column(col_name.clone(), Column::Boolean(new_col))?;
                    },
                }
            }
        }

        // インデックスもフィルタリング (もしあれば)
        if let Some(ref original_index) = self.index {
             match original_index {
                DataFrameIndex::Simple(simple_idx) => {
                    let new_index_values: Vec<String> = keep_indices.iter()
                        .filter_map(|&idx| simple_idx.get_value(idx).map(|s| s.to_string()))
                        .collect();
                    if new_index_values.len() == result.row_count { // Ensure index length matches result rows
                        let new_idx = Index::with_name(new_index_values, simple_idx.name().cloned())?;
                        result.index = Some(DataFrameIndex::Simple(new_idx));
                    } else {
                        // If index filtering fails or length mismatch, reset index
                        result.index = None;
                    }
                },
                DataFrameIndex::Multi(_) => {
                    // MultiIndex filtering is more complex, reset for now
                    result.index = None;
                }
            }
        }


        Ok(result)
    }

    /// 各行が重複しているかどうかを示すブール列を返す
    ///
    /// # Arguments
    /// * `subset` - 重複判定に使用する列名のスライス (Noneの場合は全列)
    /// * `keep` - どの重複行を `false` (重複でない) とマークするか ('first', 'last', 'all' のいずれか、デフォルトは 'first')
    ///            'all' の場合はすべての重複行を `true` とマークする
    pub fn duplicated(&self, subset: Option<&[&str]>, keep: Option<&str>) -> Result<Column> {
        // 保持戦略を決定
        let keep_strategy = match keep.unwrap_or("first") {
            "first" => KeepStrategy::First,
            "last" => KeepStrategy::Last,
            "all" => KeepStrategy::All,
            _ => return Err(Error::invalid_argument("keep引数は'first'、'last'、または'all'のいずれかでなければなりません".to_string())),
        };

        self.duplicated_impl(subset, keep_strategy)
    }

    /// `duplicated` の `Vec<String>` 版
    pub fn duplicated_vec(&self, subset: Option<&Vec<String>>, keep: Option<&str>) -> Result<Column> {
        // &Vec<String> を Option<&[&str]> に変換
        let subset_strs = subset.map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<&str>>());
        let subset_slice = subset_strs.as_ref().map(|v| v.as_slice());

        self.duplicated(subset_slice, keep)
    }

    /// `duplicated` の内部実装
    fn duplicated_impl(&self, subset: Option<&[&str]>, keep_strategy: KeepStrategy) -> Result<Column> {
        // 重複判定に使用する列
        let columns_to_check = match subset {
            Some(cols) => cols.to_vec(),
            None => self.column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        };

        // 結果を格納するブール値のベクター
        let mut is_duplicate = vec![false; self.row_count];
        // 見たキーとその最初の/最後の出現インデックスを格納
        let mut seen: HashMap<String, usize> = HashMap::new();

        // 各行を処理
        for row_idx in 0..self.row_count {
            // 重複判定用のキーを作成
            let mut key = String::new();
            for &col_name in &columns_to_check {
                 if let Some(&col_idx) = self.column_indices.get(col_name) {
                    let col = &self.columns[col_idx];
                    let val = match col {
                        Column::Int64(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                        Column::Float64(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                        Column::String(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                        Column::Boolean(c) => if let Ok(Some(v)) = c.get(row_idx) { format!("{}:", v) } else { "NULL:".to_string() },
                    };
                    key.push_str(&val);
                } else {
                     return Err(Error::ColumnNotFound(col_name.to_string()));
                 }
            }

            // 重複判定とマーク
            match keep_strategy {
                KeepStrategy::First => {
                    if seen.contains_key(&key) {
                        is_duplicate[row_idx] = true; // 2回目以降は重複
                    } else {
                        seen.insert(key, row_idx); // 初めて見たキー
                    }
                },
                KeepStrategy::Last => {
                    // 以前に同じキーがあれば、前のインデックスを重複マーク
                    if let Some(prev_idx) = seen.insert(key, row_idx) {
                        is_duplicate[prev_idx] = true;
                    }
                    // 現在の行はまだ重複マークしない (最後の出現かもしれない)
                },
                KeepStrategy::All => {
                    // 以前に同じキーがあれば、前のインデックスも現在のインデックスも重複マーク
                    if let Some(prev_idx) = seen.get(&key) {
                         is_duplicate[*prev_idx] = true; // 前のやつを True に
                         is_duplicate[row_idx] = true;   // 今回のも True に
                    } else {
                        seen.insert(key, row_idx); // 初めて見た
                    }
                },
            }
        }

        // KeepStrategy::Last の場合、最後に seen に残ったインデックスは重複ではない
        if keep_strategy == KeepStrategy::Last {
            for &last_idx in seen.values() {
                is_duplicate[last_idx] = false;
            }
        }


        // 結果をBooleanColumnとして返す
        let mut result = crate::column::BooleanColumn::new(is_duplicate);
        result.name = Some("duplicated".to_string());
        Ok(Column::Boolean(result))
    }

    /// 条件付き集計 (未実装)
    pub fn conditional_aggregate<F>(&self, _group_by: &str, _agg_column: &str, _condition: F) -> Result<Self>
    where
        F: Fn(&ColumnView<'_>) -> Result<Column> + Sync + Send,
    {
        // TODO: 実装する
        Err(Error::NotImplemented("conditional_aggregate".to_string()))
    }

    /// GroupByオブジェクトを作成
    pub fn groupby(&self, by: &str) -> Result<GroupBy> {
        // グループ化列が存在するか確認
        if !self.column_indices.contains_key(by) {
            return Err(Error::ColumnNotFound(by.to_string()));
        }
        Ok(GroupBy::new(self, by))
    }

    /// DataFrameを縦長形式 (melt) に変換
    pub fn melt(&self, options: &crate::dataframe::MeltOptions) -> Result<Self> {
        // 識別子変数 (ID variables)
        let id_vars = match &options.id_vars {
            Some(vars) => vars.clone(),
            None => Vec::new(), // 指定がない場合は空
        };

        // 値変数 (value variables)
        let value_vars = match &options.value_vars {
            Some(vars) => {
                // 指定された列が存在するか確認
                for var in vars {
                    if !self.column_indices.contains_key(var) {
                        return Err(Error::ColumnNotFound(var.clone()));
                    }
                }
                vars.clone()
            },
            None => {
                // 指定がない場合は、id_vars以外のすべての列
                self.column_names.iter()
                    .filter(|col| !id_vars.contains(col))
                    .cloned()
                    .collect()
            },
        };

        // 値変数が空の場合はエラー
        if value_vars.is_empty() {
            return Err(Error::Empty("value_varsが空です。meltする列がありません。".to_string()));
        }

        // 新しい列名
        let var_name = options.var_name.clone().unwrap_or_else(|| "variable".to_string());
        let value_name = options.value_name.clone().unwrap_or_else(|| "value".to_string());

        // 結果のDataFrame
        let mut result = Self::new();
        let result_rows = self.row_count * value_vars.len();

        // id_vars 列を結果に追加 (値を繰り返す)
        for id_col_name in &id_vars {
            let col_idx = self.column_indices[id_col_name];
            let col = &self.columns[col_idx];

            match col {
                Column::Int64(int_col) => {
                    let mut values = Vec::with_capacity(result_rows);
                    for row_idx in 0..self.row_count {
                        let value = int_col.get(row_idx).ok().flatten().unwrap_or(0);
                        for _ in 0..value_vars.len() { values.push(value); }
                    }
                    result.add_column(id_col_name.clone(), Column::Int64(crate::column::Int64Column::new(values)))?;
                },
                Column::Float64(float_col) => {
                    let mut values = Vec::with_capacity(result_rows);
                    for row_idx in 0..self.row_count {
                        let value = float_col.get(row_idx).ok().flatten().unwrap_or(0.0);
                        for _ in 0..value_vars.len() { values.push(value); }
                    }
                     result.add_column(id_col_name.clone(), Column::Float64(crate::column::Float64Column::new(values)))?;
                },
                Column::String(string_col) => {
                    let mut values = Vec::with_capacity(result_rows);
                    for row_idx in 0..self.row_count {
                        let value = string_col.get(row_idx).ok().flatten().map(|s| s.to_string()).unwrap_or_default();
                        for _ in 0..value_vars.len() { values.push(value.clone()); }
                    }
                     result.add_column(id_col_name.clone(), Column::String(crate::column::StringColumn::new(values)))?;
                },
                Column::Boolean(bool_col) => {
                    let mut values = Vec::with_capacity(result_rows);
                    for row_idx in 0..self.row_count {
                        let value = bool_col.get(row_idx).ok().flatten().unwrap_or(false);
                        for _ in 0..value_vars.len() { values.push(value); }
                    }
                    result.add_column(id_col_name.clone(), Column::Boolean(crate::column::BooleanColumn::new(values)))?;
                },
            }
        }

        // variable 列を追加
        let mut var_values = Vec::with_capacity(result_rows);
        for _ in 0..self.row_count {
            for value_var in &value_vars {
                var_values.push(value_var.clone());
            }
        }
        result.add_column(var_name, Column::String(crate::column::StringColumn::new(var_values)))?;

        // value 列を追加 (元の値を文字列として格納)
        let mut all_values_string = Vec::with_capacity(result_rows);
        for row_idx in 0..self.row_count {
            for value_var in &value_vars {
                let col_idx = self.column_indices[value_var];
                let col = &self.columns[col_idx];
                let value_str = match col {
                    Column::Int64(c) => c.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                    Column::Float64(c) => c.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                    Column::String(c) => c.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                    Column::Boolean(c) => c.get(row_idx).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                };
                all_values_string.push(value_str);
            }
        }
        result.add_column(value_name, Column::String(crate::column::StringColumn::new(all_values_string)))?;

        Ok(result)
    }

    /// DataFrameを指定された列でスタック (縦持ちに変換)
    pub fn stack(&self, options: &crate::dataframe::StackOptions) -> Result<Self> {
        // スタックする列
        let columns_to_stack = match &options.columns {
            Some(cols) => {
                // 指定された列が存在するか確認
                for col in cols {
                    if !self.column_indices.contains_key(col) {
                        return Err(Error::ColumnNotFound(col.clone()));
                    }
                }
                cols.clone()
            },
            None => self.column_names.clone(), // 指定がない場合は全列
        };

        if columns_to_stack.is_empty() {
            return Err(Error::Empty("スタックする列が指定されていません".to_string()));
        }

        // 新しい列名
        let var_name = options.var_name.clone().unwrap_or_else(|| "level_1".to_string()); // pandasに合わせる
        let value_name = options.value_name.clone().unwrap_or_else(|| "0".to_string()); // pandasに合わせる

        // スタックしない列 (id_vars になる)
        let non_stack_columns: Vec<String> = self.column_names.iter()
            .filter(|col| !columns_to_stack.contains(col))
            .cloned()
            .collect();

        // melt を使ってスタックを実現
        let melt_options = crate::dataframe::MeltOptions {
            id_vars: Some(non_stack_columns),
            value_vars: Some(columns_to_stack),
            var_name: Some(var_name),
            value_name: Some(value_name),
            ..Default::default()
        };

        let mut result = self.melt(&melt_options)?;

        // dropna オプション (現状では value 列が NULL 文字列になる可能性があるため、それを取り除く)
        if options.dropna {
            let value_col_name = melt_options.value_name.unwrap(); // meltで設定した名前
            let filter_col_name = "_filter_col";

            // value列が "NULL" でない行をフィルタリングするためのブール列を作成
            let value_view = result.column(&value_col_name)?;
            let bool_values: Vec<bool> = (0..result.row_count())
                .map(|i| value_view.get(i).map_or(false, |s| s != "NULL"))
                .collect();
            let filter_column = crate::column::BooleanColumn::new(bool_values);
            result.add_column(filter_col_name.to_string(), Column::Boolean(filter_column))?;

            // フィルタリング実行
            result = result.filter(filter_col_name)?;
            // 一時的なフィルタ列を削除 (削除処理が必要)
            // result.drop_column(filter_col_name)?; // drop_column が必要
             if let Some(idx) = result.column_indices.remove(filter_col_name) {
                result.columns.remove(idx);
                result.column_names.retain(|name| name != filter_col_name);
                // column_indices の再構築が必要になる場合がある
                 result.column_indices = result.column_names.iter().enumerate().map(|(i, name)| (name.clone(), i)).collect();
            }

        }

        // TODO: stack は通常 MultiIndex を生成する。現状は単純な melt のラッパー。

        Ok(result)
    }

    /// ピボットテーブルを作成
    pub fn pivot_table(&self, index: &str, columns: &str, values: &str, agg_func: crate::pivot::AggFunction) -> Result<Self> {
        // 必要な列が存在するか確認
        if !self.column_indices.contains_key(index) { return Err(Error::ColumnNotFound(index.to_string())); }
        if !self.column_indices.contains_key(columns) { return Err(Error::ColumnNotFound(columns.to_string())); }
        if !self.column_indices.contains_key(values) { return Err(Error::ColumnNotFound(values.to_string())); }

        // --- 1. ユニークなインデックスと列の値を取得 ---
        let index_col_idx = self.column_indices[index];
        let columns_col_idx = self.column_indices[columns];
        let values_col_idx = self.column_indices[values];

        let index_col = &self.columns[index_col_idx];
        let columns_col = &self.columns[columns_col_idx];
        let values_col = &self.columns[values_col_idx];

        let mut unique_indices_map: HashMap<String, usize> = HashMap::new();
        let mut unique_indices_vec: Vec<String> = Vec::new();
        let mut unique_columns_map: HashMap<String, usize> = HashMap::new();
        let mut unique_columns_vec: Vec<String> = Vec::new();

        for i in 0..self.row_count {
            // インデックスの値を取得 (文字列化)
            let index_val_str = match index_col {
                 Column::Int64(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                 Column::Float64(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                 Column::String(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                 Column::Boolean(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
            };
            if !unique_indices_map.contains_key(&index_val_str) {
                let new_idx = unique_indices_vec.len();
                unique_indices_map.insert(index_val_str.clone(), new_idx);
                unique_indices_vec.push(index_val_str);
            }

            // 列の値を取得 (文字列化)
            let column_val_str = match columns_col {
                 Column::Int64(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                 Column::Float64(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                 Column::String(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                 Column::Boolean(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
            };
             if !unique_columns_map.contains_key(&column_val_str) {
                let new_idx = unique_columns_vec.len();
                unique_columns_map.insert(column_val_str.clone(), new_idx);
                unique_columns_vec.push(column_val_str);
            }
        }

        // --- 2. ピボットテーブルのデータ構造を初期化 ---
        let num_rows = unique_indices_vec.len();
        let num_cols = unique_columns_vec.len();
        // 値を格納 (f64として集計するため)
        let mut pivot_data = vec![vec![0.0; num_cols]; num_rows];
        // カウント/Min/Max/Meanのために使用
        let mut count_data = vec![vec![0u64; num_cols]; num_rows];
        // Min/Max の初期化用フラグ (一度も値が入っていないセルを区別)
        let mut initialized = vec![vec![false; num_cols]; num_rows];


        // --- 3. 元のDataFrameを走査して値を集計 ---
        for i in 0..self.row_count {
             // インデックスと列の値を取得
            let index_val_str = match index_col {
                 Column::Int64(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
                 Column::Float64(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
                 Column::String(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
                 Column::Boolean(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
            }.unwrap_or_else(|| "NULL".to_string()); // Noneは "NULL" として扱う

            let column_val_str = match columns_col {
                 Column::Int64(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
                 Column::Float64(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
                 Column::String(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
                 Column::Boolean(c) => c.get(i).ok().flatten().map(|v| v.to_string()),
            }.unwrap_or_else(|| "NULL".to_string());

             // 集計する値を取得 (f64に変換)
            let value_f64 = match values_col {
                Column::Int64(c) => c.get(i).ok().flatten().map(|v| v as f64),
                Column::Float64(c) => c.get(i).ok().flatten(),
                Column::Boolean(c) => c.get(i).ok().flatten().map(|v| if v { 1.0 } else { 0.0 }),
                Column::String(_) => None, // 文字列は集計不可 (エラーにするか、Countのみ許容するか)
            };

            // 値が存在しない場合はスキップ (Count以外)
            if value_f64.is_none() && agg_func != crate::pivot::AggFunction::Count {
                continue;
            }
             let current_value = value_f64.unwrap_or(0.0); // Countの場合は0.0でOK

            // 対応するピボットテーブルのセルインデックスを取得
            if let (Some(&row_idx), Some(&col_idx)) = (unique_indices_map.get(&index_val_str), unique_columns_map.get(&column_val_str)) {
                 // 集計関数に基づいて値を更新
                match agg_func {
                    crate::pivot::AggFunction::Sum => {
                        pivot_data[row_idx][col_idx] += current_value;
                    },
                    crate::pivot::AggFunction::Mean => {
                        pivot_data[row_idx][col_idx] += current_value;
                        count_data[row_idx][col_idx] += 1;
                    },
                     crate::pivot::AggFunction::Min => {
                        if !initialized[row_idx][col_idx] || current_value < pivot_data[row_idx][col_idx] {
                            pivot_data[row_idx][col_idx] = current_value;
                            initialized[row_idx][col_idx] = true;
                        }
                    },
                    crate::pivot::AggFunction::Max => {
                         if !initialized[row_idx][col_idx] || current_value > pivot_data[row_idx][col_idx] {
                            pivot_data[row_idx][col_idx] = current_value;
                            initialized[row_idx][col_idx] = true;
                        }
                    },
                    crate::pivot::AggFunction::Count => {
                        // Countの場合は値の存在有無に関わらずカウント
                         count_data[row_idx][col_idx] += 1;
                    },
                }
                 // Count以外で値が存在した場合、カウントも増やす (Min/Max/Mean用)
                if value_f64.is_some() && agg_func != crate::pivot::AggFunction::Count {
                     if agg_func != crate::pivot::AggFunction::Sum { // Sum以外はCountも必要
                         count_data[row_idx][col_idx] += 1;
                     }
                     if !initialized[row_idx][col_idx] && (agg_func == crate::pivot::AggFunction::Min || agg_func == crate::pivot::AggFunction::Max) {
                         initialized[row_idx][col_idx] = true; // Min/Maxのために初期化フラグを立てる
                         pivot_data[row_idx][col_idx] = current_value; // 最初の値を入れる
                     }
                }
            }
        }

        // --- 4. Mean の計算と Count のデータ準備 ---
         let final_data = if agg_func == crate::pivot::AggFunction::Mean {
            for r in 0..num_rows {
                for c in 0..num_cols {
                    if count_data[r][c] > 0 {
                        pivot_data[r][c] /= count_data[r][c] as f64;
                    } else {
                         pivot_data[r][c] = f64::NAN; // データがない場合はNaN
                    }
                }
            }
            pivot_data // Mean計算後のpivot_dataを使用
        } else if agg_func == crate::pivot::AggFunction::Count {
            // Countの場合はcount_dataをf64に変換して使用
             count_data.into_iter()
                .map(|row| row.into_iter().map(|count| count as f64).collect())
                .collect()
        } else {
            // Sum, Min, Max の場合はそのまま pivot_data を使用
            // ただし、Min/Maxで一度も値が入らなかったセルはNaNにする
            if agg_func == crate::pivot::AggFunction::Min || agg_func == crate::pivot::AggFunction::Max {
                 for r in 0..num_rows {
                    for c in 0..num_cols {
                        if !initialized[r][c] {
                             pivot_data[r][c] = f64::NAN;
                        }
                    }
                 }
            }
            pivot_data
        };


        // --- 5. 結果のDataFrameを構築 ---
        let mut result = Self::new();

        // インデックス列を追加
        let index_result_col = crate::column::StringColumn::new(unique_indices_vec);
        result.add_column(index.to_string(), Column::String(index_result_col))?;

        // 各列を追加
        for (j, col_name) in unique_columns_vec.iter().enumerate() {
            let mut col_data = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                col_data.push(final_data[i][j]);
            }
            let col = crate::column::Float64Column::new(col_data);
            result.add_column(col_name.clone(), Column::Float64(col))?;
        }

        // インデックスを設定 (元のインデックス列を結果のインデックスにする)
        result.set_index(index)?;

        Ok(result)
    }

    /// 複数のDataFrameを縦方向に連結
    pub fn concat(dfs: &[&Self], _ignore_index: bool) -> Result<Self> {
        if dfs.is_empty() {
            return Ok(Self::new());
        }

        // --- 1. 全DataFrameの列名を収集し、結果の列順序を決定 ---
        let mut all_column_names_ordered: Vec<String> = Vec::new();
        let mut all_column_names_set: HashSet<String> = HashSet::new();

        for df in dfs.iter() {
            for name in &df.column_names {
                 // カテゴリカルメタデータ列は除外
                if !name.ends_with(CATEGORICAL_META_KEY) && !name.ends_with(CATEGORICAL_ORDER_META_KEY) {
                    if all_column_names_set.insert(name.clone()) {
                        all_column_names_ordered.push(name.clone());
                    }
                }
            }
        }

        // --- 2. 結果DataFrameを初期化 ---
        let mut result = Self::new();
        let mut result_columns_data: HashMap<String, ColumnBuilder> = HashMap::new();

        // --- 3. 各DataFrameを処理してデータを結合 ---
        for df in dfs.iter() {
            let current_row_count = df.row_count();

            for res_col_name in &all_column_names_ordered {
                let builder = result_columns_data.entry(res_col_name.clone()).or_insert_with(|| ColumnBuilder::new());

                 if let Some(col_idx) = df.column_indices.get(res_col_name) {
                    let col = &df.columns[*col_idx];
                    // 既存の列データをビルダーに追加
                    builder.append_column(col, current_row_count)?;
                } else {
                    // 存在しない列は適切な型のNA値で埋める
                    builder.append_na(current_row_count)?;
                }
            }
        }

        // --- 4. ビルダーから最終的な列を構築してDataFrameに追加 ---
        for col_name in all_column_names_ordered {
            if let Some(builder) = result_columns_data.remove(&col_name) {
                let final_column = builder.build()?;
                result.add_column(col_name, final_column)?;
            }
        }

        // --- 5. カテゴリカル情報の結合 (やや複雑) ---
        // 各列について、連結元のいずれかがカテゴリカルなら結果もカテゴリカルにする
        // カテゴリと順序情報をマージする必要がある
        let mut categorical_columns = Vec::new();
        
        for col_name in &result.column_names {
             let mut is_categorical = false;
             let mut merged_categories: HashSet<String> = HashSet::new();
             let final_order = CategoricalOrder::Unordered; // デフォルトはUnordered

             for df in dfs.iter() {
                 if df.is_categorical(col_name) {
                     is_categorical = true;
                     if let Ok(cats) = df.get_categories(col_name) {
                         merged_categories.extend(cats);
                     }
                     // 順序情報を確認 (一つでもOrderedがあればOrderedにするか、ルールを決める必要あり)
                     if let Some(order_key) = df.column_indices.get(&format!("{}{}", col_name, CATEGORICAL_ORDER_META_KEY)) {
                         if let Column::Boolean(order_col) = &df.columns[*order_key] {
                             if order_col.len() > 0 && order_col.get(0).ok().flatten() == Some(true) {
                                 // final_order = CategoricalOrder::Ordered;
                             }
                         }
                     }
                 }
             }

             // カテゴリカル変換フラグをとっておき、後で処理する
             if is_categorical {
                 categorical_columns.push(col_name.to_string());
             }
             
             // TODO: カテゴリのマージと順序設定のロジックを追加
             // result.add_categories_slice(col_name, &merged_categories.into_iter().collect::<Vec<_>>())?;
             // result.set_categorical_ordered(col_name, final_order)?;
        }

        // 保存しておいたカテゴリカル変換を実行
        // これによりfor文の可変借用が終了した後で処理
        for col_name in categorical_columns {
            result.astype_categorical_simple(&col_name)?;
        }

        // --- 6. インデックスの処理 ---
        // ignore_index = true なら新しいデフォルトインデックスを生成
        // ignore_index = false なら元のインデックスを連結 (重複可能性あり)
        // 現状はインデックスをリセット (None)
        result.index = None;


        Ok(result)
    }

    // --- カテゴリカル関連メソッド (DataFrameCompat から移動) ---

    /// 列がカテゴリカルデータかどうかを判定
    fn is_categorical(&self, column: &str) -> bool {
        let meta_key = format!("{}{}", column, CATEGORICAL_META_KEY);
        self.column_indices.contains_key(&meta_key)
    }

    /// 指定された列をカテゴリカル型に変換 (カテゴリや順序は無視する単純版)
    fn astype_categorical_simple(&mut self, column: &str) -> Result<()> {
        if !self.column_indices.contains_key(column) {
            return Err(Error::ColumnNotFound(column.to_string()));
        }
        if self.is_categorical(column) {
            return Ok(()); // 既にカテゴリカルなら何もしない
        }

        let col_idx = self.column_indices[column];
        if let Column::String(ref string_col) = self.columns[col_idx] {
            // 元の列から値を取得
            let string_values: Vec<String> = (0..string_col.len())
                .map(|i| string_col.get(i).ok().flatten().map(|s| s.to_string()).unwrap_or_default())
                .collect();

            // カテゴリカルモードで新しいStringColumnを作成
            let new_col = StringColumn::new_with_mode(string_values, StringColumnOptimizationMode::Categorical);

            // 元の列を置き換え
            if let Some(name) = string_col.get_name() {
                let mut new_col_with_name = new_col;
                new_col_with_name.set_name(name);
                self.columns[col_idx] = Column::String(new_col_with_name);
            } else {
                self.columns[col_idx] = Column::String(new_col);
            }

            // メタデータ列を追加
            let meta_key = format!("{}{}", column, CATEGORICAL_META_KEY);
            if !self.contains_column(&meta_key) { // メタデータがなければ追加
                let meta_values = vec![true; self.row_count];
                let meta_col = Column::Boolean(crate::column::BooleanColumn::new(meta_values));
                self.add_column(meta_key, meta_col)?;
            }

            let order_key = format!("{}{}", column, CATEGORICAL_ORDER_META_KEY);
             if !self.contains_column(&order_key) { // メタデータがなければ追加
                let order_values = vec![false; self.row_count]; // デフォルトはUnordered
                let order_col = Column::Boolean(crate::column::BooleanColumn::new(order_values));
                self.add_column(order_key, order_col)?;
            }

            Ok(())
        } else {
            Err(Error::ColumnTypeMismatch {
                name: column.to_string(),
                expected: ColumnType::String,
                found: self.columns[col_idx].column_type(),
            })
        }
    }

    /// 指定された列のカテゴリを取得
    fn get_categories(&self, column: &str) -> Result<Vec<String>> {
        if !self.is_categorical(column) {
            return Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルデータではありません", column)));
        }
        let col_idx = self.column_indices[column];
        if let Column::String(ref string_col) = self.columns[col_idx] {
            // StringColumnが内部でカテゴリを保持している場合、それを利用する
            // そうでない場合は、ユニークな値を収集する
             match string_col.optimization_mode {
                 StringColumnOptimizationMode::Categorical => {
                     // カテゴリカルモードならプールから取得できるはず (実装依存)
                     // ここでは仮にユニーク値を収集
                     let mut unique_values = HashSet::new();
                     for i in 0..string_col.len() {
                         if let Ok(Some(val)) = string_col.get(i) {
                             unique_values.insert(val.to_string());
                         }
                     }
                     let mut categories: Vec<String> = unique_values.into_iter().collect();
                     categories.sort(); // 一貫性のためソート
                     Ok(categories)
                 },
                 _ => {
                     // カテゴリカルモードでないStringColumnがカテゴリカルとしてマークされているのは矛盾
                     Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルとしてマークされていますが、内部モードが異なります", column)))
                 }
             }
        } else {
            Err(Error::OperationFailed(format!("カテゴリカル列 '{}' が文字列型ではありません", column)))
        }
    }

    /// カテゴリカル列の順序を設定
    fn set_categorical_ordered(&mut self, column: &str, ordered: crate::series::CategoricalOrder) -> Result<()> {
        let is_ordered = match ordered {
            crate::series::CategoricalOrder::Ordered => true,
            crate::series::CategoricalOrder::Unordered => false,
        };
        self.set_categorical_ordered_bool(column, is_ordered)
    }

    /// カテゴリカル列の順序をブール値で設定
    fn set_categorical_ordered_bool(&mut self, column: &str, ordered: bool) -> Result<()> {
        if !self.is_categorical(column) {
            return Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルデータではありません", column)));
        }
        let order_key = format!("{}{}", column, CATEGORICAL_ORDER_META_KEY);
        let order_idx = self.column_indices.get(&order_key)
            .ok_or_else(|| Error::ColumnNotFound(order_key.clone()))?;

        // メタデータ列を更新
        let new_col = Column::Boolean(crate::column::BooleanColumn::new(vec![ordered; self.row_count]));
        self.columns[*order_idx] = new_col;
        Ok(())
    }

    /// カテゴリカル列に新しいカテゴリを追加 (未実装)
    fn add_categories<T: Into<Vec<String>>>(&mut self, column: &str, new_categories: T) -> Result<()> {
        let new_categories_vec = new_categories.into();
        self.add_categories_slice(column, &new_categories_vec)
    }

    /// カテゴリカル列に新しいカテゴリを追加 (スライス版) (未実装)
    fn add_categories_slice(&mut self, column: &str, _new_categories: &[String]) -> Result<()> {
        if !self.is_categorical(column) {
            return Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルデータではありません", column)));
        }
        // TODO: StringColumnの内部実装にカテゴリ追加機能が必要
        Err(Error::NotImplemented("add_categories for OptimizedDataFrame".to_string()))
    }

    /// カテゴリカル列のカテゴリを並び替える (未実装)
    fn reorder_categories(&mut self, column: &str, _new_categories: Vec<String>) -> Result<()> {
         if !self.is_categorical(column) {
            return Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルデータではありません", column)));
        }
        // TODO: StringColumnの内部実装にカテゴリ並び替え機能が必要
        Err(Error::NotImplemented("reorder_categories for OptimizedDataFrame".to_string()))
    }

    /// カテゴリカル列からカテゴリを削除する (未実装)
    fn remove_categories(&mut self, column: &str, _categories_to_remove: &[String]) -> Result<()> {
         if !self.is_categorical(column) {
            return Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルデータではありません", column)));
        }
        // TODO: StringColumnの内部実装にカテゴリ削除機能が必要
        Err(Error::NotImplemented("remove_categories for OptimizedDataFrame".to_string()))
    }

    /// カテゴリカルデータのリストからDataFrameを作成
    fn from_categoricals(categoricals: Vec<(String, crate::series::StringCategorical)>) -> Result<Self> {
        let mut df = Self::new();
        for (name, cat) in categoricals {
            let values = cat.as_values();
            let ordered = cat.ordered().clone();
            let string_values: Vec<String> = values.iter()
                .map(|opt| opt.clone().unwrap_or_default())
                .collect();

            // カテゴリカル列として追加し、順序を設定
            df.add_categorical_column_vec(name.clone(), string_values)?;
            df.set_categorical_ordered(&name, ordered)?;
        }
        Ok(df)
    }

    /// 指定された列を StringCategorical として取得
    fn get_categorical(&self, column: &str) -> Result<crate::series::StringCategorical> {
        if !self.is_categorical(column) {
            return Err(Error::OperationFailed(format!("列 '{}' はカテゴリカルデータではありません", column)));
        }
        let categories = self.get_categories(column)?; // カテゴリを取得
        let col_idx = self.column_indices[column];

        // 値を取得
        let values: Vec<Option<String>> = if let Column::String(ref string_col) = self.columns[col_idx] {
            (0..string_col.len())
                .map(|i| string_col.get(i).ok().flatten().map(|s| s.to_string()))
                .collect()
        } else {
            return Err(Error::OperationFailed(format!("カテゴリカル列 '{}' が文字列型ではありません", column)));
        };

        // 順序情報を取得
        let order_key = format!("{}{}", column, CATEGORICAL_ORDER_META_KEY);
        let ordered = if let Some(&order_idx) = self.column_indices.get(&order_key) {
            if let Column::Boolean(ref bool_col) = self.columns[order_idx] {
                 if bool_col.len() > 0 && bool_col.get(0).ok().flatten() == Some(true) {
                    crate::series::CategoricalOrder::Ordered
                 } else {
                     crate::series::CategoricalOrder::Unordered
                 }
            } else { crate::series::CategoricalOrder::Unordered } // メタデータがBooleanでない場合はUnordered
        } else { crate::series::CategoricalOrder::Unordered }; // メタデータがない場合はUnordered

        // StringCategoricalを作成
        // Option<String>からStringへの変換
        let clean_values: Vec<String> = values.into_iter()
            .filter_map(|v| v)
            .collect();
        
        Ok(crate::series::StringCategorical::new(clean_values, Some(categories), Some(ordered)).unwrap())
    }

    /// 指定された列の値の出現回数をHashMapで取得
    fn value_counts_map(&self, column: &str) -> Result<HashMap<String, usize>> {
         if !self.column_indices.contains_key(column) {
            return Err(Error::ColumnNotFound(column.to_string()));
        }
        let col_idx = self.column_indices[column];
        let col = &self.columns[col_idx];

        let mut counts = HashMap::new();
        for i in 0..self.row_count {
            let value_str = match col {
                Column::Int64(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                Column::Float64(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                Column::String(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
                Column::Boolean(c) => c.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_else(|| "NULL".to_string()),
            };
            *counts.entry(value_str).or_insert(0) += 1;
        }
        Ok(counts)
    }

    /// カテゴリカル列でグループ化し、値列を集計
    fn get_categorical_aggregates<T>(
        &self,
        cat_columns: &[&str],
        value_column: &str,
        aggregator: impl Fn(Vec<String>) -> Result<T>,
    ) -> Result<HashMap<Vec<String>, T>>
    where
        T: Debug + Clone + 'static,
    {
        // 必要な列が存在するか確認
        for &col in cat_columns {
            if !self.contains_column(col) { return Err(Error::ColumnNotFound(col.to_string())); }
        }
        if !self.contains_column(value_column) { return Err(Error::ColumnNotFound(value_column.to_string())); }

        // --- 1. グループキーごとに値のリストを作成 ---
        let mut groups: HashMap<Vec<String>, Vec<String>> = HashMap::new();
        for row_idx in 0..self.row_count {
            // カテゴリ列からキーを作成
            let mut key = Vec::with_capacity(cat_columns.len());
            for &cat_col_name in cat_columns {
                let col_idx = self.column_indices[cat_col_name];
                 let value = self.get_value_as_string(col_idx, row_idx)?.unwrap_or_else(|| "NA".to_string());
                 key.push(value);
            }

            // 値列から値を取得
            let val_col_idx = self.column_indices[value_column];
            let value = self.get_value_as_string(val_col_idx, row_idx)?.unwrap_or_else(|| "NA".to_string());

            // グループに値を追加
            groups.entry(key).or_default().push(value);
        }

        // --- 2. 各グループの値リストにアグリゲーターを適用 ---
        let mut aggregated_result = HashMap::new();
        for (key, values) in groups {
            let agg_value = aggregator(values)?;
            aggregated_result.insert(key, agg_value);
        }

        Ok(aggregated_result)
    }

    /// カテゴリカル列を追加 (StringCategoricalから)
    fn add_categorical_column_from_categorical(&mut self, name: impl Into<String>, categorical: crate::series::StringCategorical) -> Result<()> {
        let name: String = name.into();
        let values = categorical.as_values();
        let ordered = categorical.ordered().clone();
        let string_values: Vec<String> = values.iter()
            .map(|opt| opt.clone().unwrap_or_default())
            .collect();

        self.add_categorical_column_vec(name.clone(), string_values)?;
        self.set_categorical_ordered(&name, ordered)?;
        Ok(())
    }

    /// カテゴリカル列を追加 (Vec<String>から)
    fn add_categorical_column_vec(&mut self, name: String, values: Vec<String>) -> Result<()> {
        // 最適化モードを選択 (データサイズに基づくなど、ここではCategorical固定)
        let optimization_mode = StringColumnOptimizationMode::Categorical;

        // StringColumnを作成
        let string_col = StringColumn::with_name_and_mode(values.clone(), name.clone(), optimization_mode);

        // DataFrameに追加
        self.add_column(name.clone(), Column::String(string_col))?;

        // メタデータ列を追加
        let meta_key = format!("{}{}", name, CATEGORICAL_META_KEY);
        if !self.contains_column(&meta_key) {
            let meta_values = vec![true; values.len()];
            let meta_col = Column::Boolean(crate::column::BooleanColumn::new(meta_values));
            self.add_column(meta_key, meta_col)?;
        }

        let order_key = format!("{}{}", name, CATEGORICAL_ORDER_META_KEY);
         if !self.contains_column(&order_key) {
            let order_values = vec![false; values.len()]; // デフォルトはUnordered
            let order_col = Column::Boolean(crate::column::BooleanColumn::new(order_values));
            self.add_column(order_key, order_col)?;
        }

        Ok(())
    }

     /// 指定された列をカテゴリカル型に変換 (カテゴリや順序も考慮)
    /// 注: astype_categorical_simple とは異なり、引数を取るバージョン
    fn astype_categorical(
        &mut self,
        column: &str,
        categories: Option<Vec<String>>,
        ordered: Option<crate::series::CategoricalOrder>
    ) -> Result<()> {
        self.astype_categorical_simple(column)?; // まず基本変換

        // カテゴリ設定 (未実装)
        if let Some(_cats) = categories {
             // TODO: カテゴリを設定するロジック (StringColumn の実装に依存)
             // self.reorder_categories(column, cats)?; // 例えば並び替えを使うなど
            eprintln!("警告: astype_categoricalでのカテゴリ指定は現在無視されます。");
        }

        // 順序設定
        if let Some(order) = ordered {
            self.set_categorical_ordered(column, order)?;
        }
        Ok(())
    }


} // impl OptimizedDataFrame の終わり


// --- ColumnView の実装 ---
impl<'a> ColumnView<'a> {
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

    /// 指定されたインデックスの値を取得 (文字列として)
    pub fn get(&self, idx: usize) -> Option<String> {
        match self.column {
            Column::Int64(col) => col.get(idx).ok().flatten().map(|v| v.to_string()),
            Column::Float64(col) => col.get(idx).ok().flatten().map(|v| v.to_string()),
            Column::String(col) => col.get(idx).ok().flatten().map(|s| s.to_string()),
            Column::Boolean(col) => col.get(idx).ok().flatten().map(|v| v.to_string()),
        }
    }

    /// Int64Column としての参照を取得 (失敗した場合は None)
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        if let Column::Int64(ref col) = self.column { Some(col) } else { None }
    }

    /// Float64Column としての参照を取得 (失敗した場合は None)
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        if let Column::Float64(ref col) = self.column { Some(col) } else { None }
    }

    /// StringColumn としての参照を取得 (失敗した場合は None)
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        if let Column::String(ref col) = self.column { Some(col) } else { None }
    }

    /// BooleanColumn としての参照を取得 (失敗した場合は None)
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        if let Column::Boolean(ref col) = self.column { Some(col) } else { None }
    }

    /// 内部の Column への参照を取得
    pub fn get_column(&self) -> &Column {
        self.column
    }
}


// --- 互換性トレイトの実装 ---

// StringCategorical から Vec<String> への変換 (便宜上)
impl From<crate::series::StringCategorical> for Vec<String> {
    fn from(categorical: crate::series::StringCategorical) -> Self {
        categorical.as_values()
            .into_iter()
            .map(|opt| opt.unwrap_or_default()) // Noneは空文字列に
            .collect()
    }
}

// DataFrameCompat トレイトの実装
// ここにはトレイトで定義されたメソッドのみを実装します
impl crate::compat::DataFrameCompat for OptimizedDataFrame {
    // NASeries をカテゴリカル列として追加
    fn add_na_series_as_categorical(
        &mut self,
        name: String,
        series: crate::series::NASeries<String>,
        _categories: Option<Vec<String>>, // カテゴリは現在無視
        ordered: Option<crate::series::CategoricalOrder>,
    ) -> crate::error::Result<&mut Self> {
        // NASeriesから値を取り出す (NAは空文字列に)
        let values: Vec<String> = series.values().iter().map(|na_val| {
            match na_val {
                crate::na::NA::Value(s) => s.clone(),
                crate::na::NA::NA => String::new(),
            }
        }).collect();

        let name_for_order = name.clone(); // 順序設定用に名前をコピー

        // カテゴリカル列として追加 (内部メソッドを使用)
        self.add_categorical_column_vec(name, values)?;

        // 順序を設定
        if let Some(order) = ordered {
            self.set_categorical_ordered(&name_for_order, order)?;
        }

        Ok(self)
    }

    // StringCategorical をカテゴリカル列として追加
    fn add_categorical_column(&mut self, name: String, categorical: crate::series::StringCategorical) -> crate::error::Result<&mut Self> {
        self.add_categorical_column_from_categorical(name, categorical)?;
        Ok(self)
    }

    // 値の出現回数をカウント (内部実装を呼び出す)
    fn value_counts(&self, column: &str) -> crate::error::Result<crate::series::Series<usize>> {
        self.value_counts_impl(column)
    }

    // CSVに書き出し (カテゴリカル情報は特別な処理なし)
    fn to_csv_with_categorical<P: AsRef<std::path::Path>>(&self, path: P) -> crate::error::Result<()> {
        self.to_csv(path)
    }

    // CSVから読み込み (カテゴリカル情報の復元は未実装)
    fn from_csv_with_categorical<P: AsRef<std::path::Path>>(_path: P, _has_header: bool) -> crate::error::Result<crate::DataFrame> {
        Err(Error::NotImplemented("from_csv_with_categorical for OptimizedDataFrame".to_string()))
    }
}


// --- ヘルパー: ColumnBuilder ---
// concat時に異なる型の列を結合するためのヘルパー

#[derive(Debug)]
enum ColumnBuilderData {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Bool(Vec<Option<bool>>),
    Empty, // 初期状態または不明な型
}

struct ColumnBuilder {
    data: ColumnBuilderData,
    target_type: Option<ColumnType>, // 最初に非NA値が見つかったときの型
}

impl ColumnBuilder {
    fn new() -> Self {
        Self { data: ColumnBuilderData::Empty, target_type: None }
    }

     // 列全体を追加
    fn append_column(&mut self, col: &Column, expected_len: usize) -> Result<()> {
        match col {
            Column::Int64(c) => self.append_typed_column(c, expected_len, ColumnType::Int64, |v| v, ColumnBuilderData::Int),
            Column::Float64(c) => self.append_typed_column(c, expected_len, ColumnType::Float64, |v| v, ColumnBuilderData::Float),
            Column::String(c) => self.append_typed_column(c, expected_len, ColumnType::String, |s| s.to_string(), ColumnBuilderData::String),
            Column::Boolean(c) => self.append_typed_column(c, expected_len, ColumnType::Boolean, |v| v, ColumnBuilderData::Bool),
        }
    }

    // 型付けされた列を追加する内部ヘルパー
    // 注意: unsafe を使用しています。T と ColumnType が一致していることを強く仮定しています。
    fn append_typed_column<T, C, F, BuilderFn>(
        &mut self,
        col: &C,
        expected_len: usize,
        col_type: ColumnType,
        extractor: F,
        builder_constructor: BuilderFn,
    ) -> Result<()>
    where
        T: Clone + Default + 'static, // 値の型
        C: ColumnTrait,            // 列トレイト
        F: Fn(T) -> T,                // 値抽出/変換関数
        BuilderFn: Fn(Vec<Option<T>>) -> ColumnBuilderData, // ビルダーデータ構築関数
    {
        self.ensure_type(col_type)?;
        let mut values = Vec::with_capacity(expected_len);
        for i in 0..expected_len {
             // 列の長さが足りない場合も考慮
             // 注: get() メソッドが一般的なColumnTraitにない問題があるため一時的に無効化
             // FIXME: ColumnTraitにget()を追加するか、別の方法で値を取得する
             let value = None; // 一時的に常にNoneを返す
            values.push(value);
        }

        match &mut self.data {
            ColumnBuilderData::Empty => self.data = builder_constructor(values),
            // unsafe を使って型を強制的に合わせる。これはconcat時の型整合性が前提。
            ColumnBuilderData::Int(ref mut v) if col_type == ColumnType::Int64 => v.extend(values.into_iter().map(|opt| opt.map(|x| unsafe { std::mem::transmute_copy::<T, i64>(&x) }))),
            ColumnBuilderData::Float(ref mut v) if col_type == ColumnType::Float64 => v.extend(values.into_iter().map(|opt| opt.map(|x| unsafe { std::mem::transmute_copy::<T, f64>(&x) }))),
            ColumnBuilderData::String(ref mut v) if col_type == ColumnType::String => v.extend(values.into_iter().map(|opt| opt.map(|x| unsafe { std::mem::transmute_copy::<T, String>(&x) }))),
            ColumnBuilderData::Bool(ref mut v) if col_type == ColumnType::Boolean => v.extend(values.into_iter().map(|opt| opt.map(|x| unsafe { std::mem::transmute_copy::<T, bool>(&x) }))),
            _ => return Err(Error::OperationFailed(format!(
                "Internal error: Type mismatch in ColumnBuilder {:?} vs {:?}",
                self.target_type, col_type
            ))), // 型が不一致
        }
         Ok(())
    }


    // NA値を追加
    fn append_na(&mut self, count: usize) -> Result<()> {
        match self.target_type {
            Some(ColumnType::Int64) => {
                if let ColumnBuilderData::Int(ref mut v) = self.data { v.extend(std::iter::repeat(None).take(count)); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Int(std::iter::repeat(None).take(count).collect()); }
                 else { return Err(Error::OperationFailed("Builder type mismatch when adding NA".into())); }
            },
             Some(ColumnType::Float64) => {
                 if let ColumnBuilderData::Float(ref mut v) = self.data { v.extend(std::iter::repeat(None).take(count)); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Float(std::iter::repeat(None).take(count).collect()); }
                 else { return Err(Error::OperationFailed("Builder type mismatch when adding NA".into())); }
            },
             Some(ColumnType::String) => {
                 if let ColumnBuilderData::String(ref mut v) = self.data { v.extend(std::iter::repeat(None).take(count)); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::String(std::iter::repeat(None).take(count).collect()); }
                 else { return Err(Error::OperationFailed("Builder type mismatch when adding NA".into())); }
            },
             Some(ColumnType::Boolean) => {
                 if let ColumnBuilderData::Bool(ref mut v) = self.data { v.extend(std::iter::repeat(None).take(count)); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Bool(std::iter::repeat(None).take(count).collect()); }
                 else { return Err(Error::OperationFailed("Builder type mismatch when adding NA".into())); }
            },
            None => {
                 // 型が未定の場合、どの型にするか決められない。エラーにするか、デフォルト型（例：String）にする。
                 // ここではStringにする例
                 self.target_type = Some(ColumnType::String);
                 self.data = ColumnBuilderData::String(std::iter::repeat(None).take(count).collect());
            }
        }
         Ok(())
    }

     // 特定のインデックスの値を追加
    fn append_value(&mut self, col: &Column, index: usize) -> Result<()> {
        self.ensure_type(col.column_type())?;
        match col {
            Column::Int64(c) => {
                let value = c.get(index)?;
                if let ColumnBuilderData::Int(ref mut v) = self.data { v.push(value); }
                else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Int(vec![value]); }
                 else { return Err(Error::OperationFailed("Append value type mismatch".into()));}
            },
            Column::Float64(c) => {
                let value = c.get(index)?;
                 if let ColumnBuilderData::Float(ref mut v) = self.data { v.push(value); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Float(vec![value]); }
                 else { return Err(Error::OperationFailed("Append value type mismatch".into()));}
            },
            Column::String(c) => {
                let value = c.get(index)?.map(|s| s.to_string());
                 if let ColumnBuilderData::String(ref mut v) = self.data { v.push(value); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::String(vec![value]); }
                 else { return Err(Error::OperationFailed("Append value type mismatch".into()));}
            },
            Column::Boolean(c) => {
                let value = c.get(index)?;
                 if let ColumnBuilderData::Bool(ref mut v) = self.data { v.push(value); }
                 else if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Bool(vec![value]); }
                 else { return Err(Error::OperationFailed("Append value type mismatch".into()));}
            },
        }
         Ok(())
    }


    // ビルダーの型を確定または検証
    fn ensure_type(&mut self, new_type: ColumnType) -> Result<()> {
        if let Some(existing_type) = self.target_type {
            if existing_type != new_type {
                // 型の不一致。アップキャスト（Int -> Float）を許容するか、エラーにするか。
                // ここではエラーにする
                return Err(Error::OperationFailed(format!(
                    "Cannot concat columns of different types: {:?} and {:?}",
                    existing_type, new_type
                )));
            }
        } else {
            // 最初の非NA列で型を確定
            self.target_type = Some(new_type);
            // 必要なら self.data を初期化
             match new_type {
                 ColumnType::Int64 => if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Int(Vec::new()); },
                 ColumnType::Float64 => if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Float(Vec::new()); },
                 ColumnType::String => if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::String(Vec::new()); },
                 ColumnType::Boolean => if matches!(self.data, ColumnBuilderData::Empty) { self.data = ColumnBuilderData::Bool(Vec::new()); },
             }
        }
        Ok(())
    }

    // 最終的なColumnを構築
    fn build(self) -> Result<Column> {
        match self.data {
            // new_option 呼び出しを維持。これらのメソッドが crate::column::*Column に存在すると仮定。
            // 存在しない場合は、new メソッドなどで Option<T> を扱えるように修正が必要。
            ColumnBuilderData::Int(v) => {
                let data: Vec<i64> = v.into_iter().filter_map(|x| x).collect();
                Ok(Column::Int64(crate::column::Int64Column::new(data)))
            },
            ColumnBuilderData::Float(v) => {
                let data: Vec<f64> = v.into_iter().filter_map(|x| x).collect();
                Ok(Column::Float64(crate::column::Float64Column::new(data)))
            },
            ColumnBuilderData::String(v) => {
                let data: Vec<String> = v.into_iter().filter_map(|x| x).collect();
                Ok(Column::String(crate::column::StringColumn::new(data)))
            },
            ColumnBuilderData::Bool(v) => {
                let data: Vec<bool> = v.into_iter().filter_map(|x| x).collect();
                Ok(Column::Boolean(crate::column::BooleanColumn::new(data)))
            },
            ColumnBuilderData::Empty => {
                 // 空のビルダーの場合、デフォルト型（例：String）の空列を返すかエラー
                 Ok(Column::String(crate::column::StringColumn::new(Vec::new()))) // 空のString列
                 // Err(Error::Empty("Cannot build column from empty builder".to_string()))
            }
        }
    }
}