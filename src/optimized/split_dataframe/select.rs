use std::collections::HashSet;

use crate::column::Column;
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// 指定した列を選択してDataFrameを作成
    ///
    /// # Arguments
    /// * `columns` - 選択する列名の配列
    ///
    /// # Returns
    /// * `Result<Self>` - 選択された列を含む新しいDataFrame
    pub fn select_columns(&self, columns: &[&str]) -> Result<Self> {
        let mut df = Self::new();
        
        // 列名のセットを作成（存在チェック用）
        let column_set: HashSet<&str> = self.column_names.iter()
            .map(|s| s.as_str())
            .collect();
        
        // 指定された列を新しいDataFrameに追加
        for &col_name in columns {
            if !column_set.contains(col_name) {
                // 列が存在しない場合はエラーを返す
                return Err(crate::error::Error::ColumnNotFound(col_name.to_string()));
            }
            
            let col_idx = self.column_indices.get(col_name).unwrap();
            let column = &self.columns[*col_idx];
            
            df.add_column(col_name.to_string(), column.clone())?;
        }
        
        // インデックスをコピー
        if let Some(ref index) = self.index {
            df.index = Some(index.clone());
        }
        
        Ok(df)
    }
    
    /// インデックスによる行選択
    ///
    /// # Arguments
    /// * `indices` - 選択する行インデックスの配列
    ///
    /// # Returns
    /// * `Result<Self>` - 選択された行を含む新しいDataFrame
    /// 
    /// Note: sort.rsにも同名のメソッドが存在しますが、そちらはprivateです
    pub fn select_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let mut df = Self::new();
        
        // 列ごとに処理
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let column = &self.columns[col_idx];
            
            // 選択された行のデータを抽出
            let new_column = match column {
                Column::Int64(col) => {
                    let values: Vec<i64> = indices.iter()
                        .filter_map(|&idx| if idx < self.row_count {
                            col.get(idx).ok().flatten()
                        } else {
                            None
                        })
                        .collect();
                    Column::Int64(crate::column::Int64Column::new(values))
                },
                Column::Float64(col) => {
                    let values: Vec<f64> = indices.iter()
                        .filter_map(|&idx| if idx < self.row_count {
                            col.get(idx).ok().flatten()
                        } else {
                            None
                        })
                        .collect();
                    Column::Float64(crate::column::Float64Column::new(values))
                },
                Column::String(col) => {
                    let values: Vec<String> = indices.iter()
                        .filter_map(|&idx| if idx < self.row_count {
                            col.get(idx).ok().flatten().map(|s| s.to_string())
                        } else {
                            None
                        })
                        .collect();
                    Column::String(crate::column::StringColumn::new(values))
                },
                Column::Boolean(col) => {
                    let values: Vec<bool> = indices.iter()
                        .filter_map(|&idx| if idx < self.row_count {
                            col.get(idx).ok().flatten()
                        } else {
                            None
                        })
                        .collect();
                    Column::Boolean(crate::column::BooleanColumn::new(values))
                },
            };
            
            df.add_column(col_name.clone(), new_column)?;
        }
        
        // 新しいインデックスを作成
        // NOTE: 既存のインデックスから選択行に対応する値を取得することも考えられるが、
        // 単純化のため、ここでは新しい連番インデックスを作成する
        df.set_default_index()?;
        
        Ok(df)
    }
    
    /// 行と列の両方を選択
    ///
    /// # Arguments
    /// * `row_indices` - 選択する行インデックスの配列
    /// * `columns` - 選択する列名の配列
    ///
    /// # Returns
    /// * `Result<Self>` - 選択された行と列を含む新しいDataFrame
    pub fn select_rows_columns(&self, row_indices: &[usize], columns: &[&str]) -> Result<Self> {
        // まず列を選択
        let cols_selected = self.select_columns(columns)?;
        
        // 次に行を選択
        cols_selected.select_rows_by_indices(row_indices)
    }
    
    /// 条件マスクによる行選択
    ///
    /// # Arguments
    /// * `mask` - 選択条件を表すブールベクトル（Trueの行が選択される）
    ///
    /// # Returns
    /// * `Result<Self>` - 条件に一致する行を含む新しいDataFrame
    pub fn select_by_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.row_count {
            return Err(crate::error::Error::Format(
                format!("マスクの長さ({})がDataFrameの行数({})と一致しません", mask.len(), self.row_count)
            ));
        }
        
        // マスクからインデックスのリストを作成
        let indices: Vec<usize> = mask.iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();
        
        // インデックスによる選択を実行
        self.select_rows_by_indices(&indices)
    }
}

/// 行インデックスに基づいて行を選択する実装（他のモジュールで使用）
pub(crate) fn select_rows_by_indices_impl(df: &OptimizedDataFrame, indices: &[usize]) -> Result<OptimizedDataFrame> {
    // 行が0の場合は空のデータフレームを返す
    if indices.is_empty() {
        return Ok(OptimizedDataFrame::new());
    }
    
    let mut result = OptimizedDataFrame::new();
    
    // 各列を処理
    for (name, &column_idx) in &df.column_indices {
        let column = &df.columns[column_idx];
        
        // 列の型に応じて行インデックスから取得
        let selected_col = match column {
            Column::Int64(col) => {
                let selected_data: Vec<i64> = indices.iter()
                    .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                    .collect();
                Column::Int64(crate::column::Int64Column::new(selected_data))
            },
            Column::Float64(col) => {
                let selected_data: Vec<f64> = indices.iter()
                    .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                    .collect();
                Column::Float64(crate::column::Float64Column::new(selected_data))
            },
            Column::String(col) => {
                let selected_data: Vec<String> = indices.iter()
                    .map(|&idx| col.get(idx).ok().flatten().map(|s| s.to_string()).unwrap_or_default())
                    .collect();
                Column::String(crate::column::StringColumn::new(selected_data))
            },
            Column::Boolean(col) => {
                let selected_data: Vec<bool> = indices.iter()
                    .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                    .collect();
                Column::Boolean(crate::column::BooleanColumn::new(selected_data))
            },
        };
        
        result.add_column(name.clone(), selected_col)?;
    }
    
    // インデックスの取得
    if let Some(ref idx) = df.get_index() {
        // TODO: インデックスの選択処理
        match idx {
            crate::index::DataFrameIndex::Simple(simple_idx) => {
                result.set_index_from_simple_index(simple_idx.clone())?;
            },
            _ => {
                // 暫定対応：デフォルトのインデックスを設定
                result.set_default_index()?;
            }
        }
    }
    
    Ok(result)
}