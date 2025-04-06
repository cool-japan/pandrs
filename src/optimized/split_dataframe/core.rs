//! OptimizedDataFrameの核となる構造体定義と基本機能

use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use crate::column::{Column, ColumnTrait, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait, Index};

/// 最適化されたDataFrame実装
/// 列指向ストレージを使用し、高速なデータ処理を実現
#[derive(Clone)]
pub struct OptimizedDataFrame {
    // 列データ
    pub(crate) columns: Vec<Column>,
    // 列名→インデックスのマッピング
    pub(crate) column_indices: HashMap<String, usize>,
    // 列の順序
    pub(crate) column_names: Vec<String>,
    // 行数
    pub(crate) row_count: usize,
    // インデックス (オプション)
    pub(crate) index: Option<DataFrameIndex<String>>,
}

/// 列に対するビュー（参照）を表す構造体
#[derive(Clone)]
pub struct ColumnView {
    pub(crate) column: Column,
}

impl Debug for OptimizedDataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 最大表示行数
        const MAX_ROWS: usize = 10;
        
        if self.columns.is_empty() {
            return write!(f, "OptimizedDataFrame (0 rows x 0 columns)");
        }
        
        writeln!(f, "OptimizedDataFrame ({} rows x {} columns):", self.row_count, self.columns.len())?;
        
        // 列ヘッダーの表示
        write!(f, "{:<5} |", "idx")?;
        for name in &self.column_names {
            write!(f, " {:<15} |", name)?;
        }
        writeln!(f)?;
        
        // 区切り線
        write!(f, "{:-<5}-+", "")?;
        for _ in &self.column_names {
            write!(f, "-{:-<15}-+", "")?;
        }
        writeln!(f)?;
        
        // 最大MAX_ROWS行まで表示
        let display_rows = std::cmp::min(self.row_count, MAX_ROWS);
        for i in 0..display_rows {
            if let Some(ref idx) = self.index {
                let idx_value = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => {
                        if i < simple_idx.len() {
                            simple_idx.get_value(i).map(|s| s.to_string()).unwrap_or_else(|| i.to_string())
                        } else {
                            i.to_string()
                        }
                    },
                    DataFrameIndex::Multi(_) => i.to_string()
                };
                write!(f, "{:<5} |", idx_value)?;
            } else {
                write!(f, "{:<5} |", i)?;
            }
            
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
                            format!("{:.3}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("\"{}\"", val)
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
                write!(f, " {:<15} |", value)?;
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

impl OptimizedDataFrame {
    /// 新しい空のDataFrameを作成
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: 0,
            index: None,
        }
    }
    
    /// 文字列インデックスでDataFrameを作成
    pub fn with_index(index: Index<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: index.len(),
            index: Some(DataFrameIndex::<String>::from_simple(index)),
        }
    }
    
    /// 複合インデックスでDataFrameを作成
    pub fn with_multi_index(index: crate::index::MultiIndex<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: index.len(),
            index: Some(DataFrameIndex::<String>::from_multi(index)),
        }
    }
    
    /// 範囲インデックスでDataFrameを作成
    pub fn with_range_index(range: std::ops::Range<usize>) -> Result<Self> {
        let range_idx = Index::<usize>::from_range(range)?;
        // 数値インデックスを文字列インデックスに変換
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values)?;
        Ok(Self::with_index(string_idx))
    }
    
    /// 行数を取得
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    
    /// 列数を取得
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
    
    /// 列名のリストを取得
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }
    
    /// 指定された列が存在するかチェックします
    pub fn contains_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }
}
