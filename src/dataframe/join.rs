use std::collections::{HashMap, HashSet};
use crate::error::{PandRSError, Result};
use crate::DataFrame;
use crate::dataframe::DataBox;
use crate::series::Series;

/// 結合タイプの列挙型
#[derive(Debug)]
pub enum JoinType {
    /// 内部結合 (両方の表に一致する行のみ)
    Inner,
    /// 左結合 (左側の表の全ての行と、右側の表の一致する行)
    Left,
    /// 右結合 (右側の表の全ての行と、左側の表の一致する行)
    Right,
    /// 外部結合 (両方の表の全ての行)
    Outer,
}

impl DataFrame {
    /// 2つのDataFrameを結合する
    pub fn join(&self, other: &DataFrame, on: &str, join_type: JoinType) -> Result<DataFrame> {
        // 結合列が存在するか確認
        if !self.contains_column(on) {
            return Err(PandRSError::Column(format!(
                "結合列 '{}' が左側のDataFrameに存在しません", on
            )));
        }
        
        if !other.contains_column(on) {
            return Err(PandRSError::Column(format!(
                "結合列 '{}' が右側のDataFrameに存在しません", on
            )));
        }
        
        match join_type {
            JoinType::Inner => self.inner_join(other, on),
            JoinType::Left => self.left_join(other, on),
            JoinType::Right => self.right_join(other, on),
            JoinType::Outer => self.outer_join(other, on),
        }
    }
    
    // 内部結合を実行
    fn inner_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let left_keys = self.get_column_string_values(on)?;
        let right_keys = other.get_column_string_values(on)?;
        
        // 内部結合: 両方に存在するキーのみを使用
        for (left_idx, left_key) in left_keys.iter().enumerate() {
            for (right_idx, right_key) in right_keys.iter().enumerate() {
                if left_key == right_key {
                    // このキーで一致した行を新しいDataFrameに追加
                    self.add_join_row(&mut result, left_idx, other, right_idx, on)?;
                }
            }
        }
        
        Ok(result)
    }
    
    // 左結合を実行
    fn left_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let left_keys = self.get_column_string_values(on)?;
        let right_keys = other.get_column_string_values(on)?;
        
        // キーの一意セットを構築
        let right_keys_set: HashSet<&String> = right_keys.iter().collect();
        
        // 左側の全ての行をループ
        for (left_idx, left_key) in left_keys.iter().enumerate() {
            let mut has_match = false;
            
            // 右側に一致するキーがあるか確認
            if right_keys_set.contains(left_key) {
                for (right_idx, right_key) in right_keys.iter().enumerate() {
                    if left_key == right_key {
                        // このキーで一致した行を新しいDataFrameに追加
                        self.add_join_row(&mut result, left_idx, other, right_idx, on)?;
                        has_match = true;
                    }
                }
            }
            
            // 右側に一致がない場合、左側のデータのみを追加
            if !has_match {
                self.add_left_only_row(&mut result, left_idx, other, on)?;
            }
        }
        
        Ok(result)
    }
    
    // 右結合を実行
    fn right_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        // 右結合は、引数を入れ替えた左結合として実装
        other.left_join(self, on)
    }
    
    // 外部結合を実行
    fn outer_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let left_keys = self.get_column_string_values(on)?;
        let right_keys = other.get_column_string_values(on)?;
        
        // 処理済みの左キーをトラッキング
        let mut processed_left_indices = HashSet::new();
        
        // 左側の全ての行をループ
        for (left_idx, left_key) in left_keys.iter().enumerate() {
            let mut has_match = false;
            
            // 右側の一致するキーを探す
            for (right_idx, right_key) in right_keys.iter().enumerate() {
                if left_key == right_key {
                    // このキーで一致した行を新しいDataFrameに追加
                    self.add_join_row(&mut result, left_idx, other, right_idx, on)?;
                    has_match = true;
                }
            }
            
            // 右側に一致がない場合、左側のデータのみを追加
            if !has_match {
                self.add_left_only_row(&mut result, left_idx, other, on)?;
            }
            
            processed_left_indices.insert(left_key.clone());
        }
        
        // 右側で左側にないキーを処理
        for (right_idx, right_key) in right_keys.iter().enumerate() {
            if !processed_left_indices.contains(right_key) {
                // 左側に一致がない場合、右側のデータのみを追加
                self.add_right_only_row(&mut result, other, right_idx, on)?;
            }
        }
        
        Ok(result)
    }
    
    // 削除: get_column_string_valuesはmod.rsに移動しました
    
    // 結合された行を結果DataFrameに追加
    fn add_join_row(&self, result: &mut DataFrame, left_idx: usize, other: &DataFrame, right_idx: usize, on: &str) -> Result<()> {
        // 結果のDataFrameが空の場合、列を初期化
        if result.column_count() == 0 {
            // 左側の全列名を追加
            for col_name in self.column_names() {
                let empty_series = Series::<DataBox>::new(Vec::new(), Some(col_name.clone()))?;
                result.add_column(col_name.clone(), empty_series)?;
            }
            
            // 右側の列名（結合列を除く）を追加
            for col_name in other.column_names() {
                if col_name != on {
                    let name = format!("{}_{}", col_name, "right");
                    let empty_series = Series::<DataBox>::new(Vec::new(), Some(name.clone()))?;
                    result.add_column(name, empty_series)?;
                }
            }
        }
        
        // 新しい行データを作成
        let mut row_data = HashMap::new();
        
        // 左側の値を追加
        for col_name in self.column_names() {
            if let Some(series) = self.columns.get(col_name) {
                if let Some(value) = series.get(left_idx) {
                    row_data.insert(col_name.clone(), value.clone());
                }
            }
        }
        
        // 右側の値を追加（結合列を除く）
        for col_name in other.column_names() {
            if col_name != on {
                let result_col_name = format!("{}_{}", col_name, "right");
                if let Some(series) = other.columns.get(col_name) {
                    if let Some(value) = series.get(right_idx) {
                        row_data.insert(result_col_name, value.clone());
                    }
                }
            }
        }
        
        // TODO: 行データをDataFrameに追加
        // 現在の実装では行の直接追加はサポートされていないため、一時的なスタブ
        
        Ok(())
    }
    
    // 左側のみの行を結果DataFrameに追加
    fn add_left_only_row(&self, result: &mut DataFrame, left_idx: usize, other: &DataFrame, on: &str) -> Result<()> {
        // 結果のDataFrameが空の場合、列を初期化
        if result.column_count() == 0 {
            // 左側の全列名を追加
            for col_name in self.column_names() {
                let empty_series = Series::<DataBox>::new(Vec::new(), Some(col_name.clone()))?;
                result.add_column(col_name.clone(), empty_series)?;
            }
            
            // 右側の列名（結合列を除く）を追加
            for col_name in other.column_names() {
                if col_name != on {
                    let name = format!("{}_{}", col_name, "right");
                    let empty_series = Series::<DataBox>::new(Vec::new(), Some(name.clone()))?;
                    result.add_column(name, empty_series)?;
                }
            }
        }
        
        // 新しい行データを作成
        let mut row_data = HashMap::new();
        
        // 左側の値を追加
        for col_name in self.column_names() {
            if let Some(series) = self.columns.get(col_name) {
                if let Some(value) = series.get(left_idx) {
                    row_data.insert(col_name.clone(), value.clone());
                }
            }
        }
        
        // 右側の列には空の値を追加
        for col_name in other.column_names() {
            if col_name != on {
                let result_col_name = format!("{}_{}", col_name, "right");
                // 空/NAの値を入れる
                row_data.insert(result_col_name, DataBox(Box::new(String::new())));
            }
        }
        
        // TODO: 行データをDataFrameに追加
        // 現在の実装では行の直接追加はサポートされていないため、一時的なスタブ
        
        Ok(())
    }
    
    // 右側のみの行を結果DataFrameに追加
    fn add_right_only_row(&self, result: &mut DataFrame, other: &DataFrame, right_idx: usize, on: &str) -> Result<()> {
        // 結果のDataFrameが空の場合、列を初期化
        if result.column_count() == 0 {
            // 左側の全列名を追加
            for col_name in self.column_names() {
                let empty_series = Series::<DataBox>::new(Vec::new(), Some(col_name.clone()))?;
                result.add_column(col_name.clone(), empty_series)?;
            }
            
            // 右側の列名（結合列を除く）を追加
            for col_name in other.column_names() {
                if col_name != on {
                    let name = format!("{}_{}", col_name, "right");
                    let empty_series = Series::<DataBox>::new(Vec::new(), Some(name.clone()))?;
                    result.add_column(name, empty_series)?;
                }
            }
        }
        
        // 新しい行データを作成
        let mut row_data = HashMap::new();
        
        // 左側の列には空の値を追加
        for col_name in self.column_names() {
            if col_name == on {
                // 結合キーは右側から取得
                if let Some(series) = other.columns.get(col_name) {
                    if let Some(value) = series.get(right_idx) {
                        row_data.insert(col_name.clone(), value.clone());
                    }
                }
            } else {
                // 他の列は空/NAを入れる
                row_data.insert(col_name.clone(), DataBox(Box::new(String::new())));
            }
        }
        
        // 右側の値を追加
        for col_name in other.column_names() {
            if col_name != on {
                let result_col_name = format!("{}_{}", col_name, "right");
                if let Some(series) = other.columns.get(col_name) {
                    if let Some(value) = series.get(right_idx) {
                        row_data.insert(result_col_name, value.clone());
                    }
                }
            }
        }
        
        // TODO: 行データをDataFrameに追加
        // 現在の実装では行の直接追加はサポートされていないため、一時的なスタブ
        
        Ok(())
    }
}