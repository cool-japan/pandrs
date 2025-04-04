use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::column::{Column, ColumnType};
use crate::error::{Error, Result};
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::operations::AggregateOp;

/// 遅延評価のためのデータフレームラッパー
#[derive(Clone)]
pub struct LazyFrame {
    // 元のデータフレーム
    source: Arc<OptimizedDataFrame>,
    // 適用する操作のキュー
    operations: Vec<Operation>,
}

/// 遅延評価するための操作
#[derive(Clone)]
pub enum Operation {
    /// 列を選択
    Select(Vec<String>),
    /// フィルタリング
    Filter(String),
    /// マッピング関数
    Map(Arc<dyn Fn(&Column) -> Result<Column> + Send + Sync>),
    /// 集計
    Aggregate {
        /// グループ化する列
        group_by: Vec<String>,
        /// 集計操作
        aggregations: Vec<(String, AggregateOp, String)>,
    },
    /// 結合
    Join {
        /// 右側のデータフレーム
        right: Arc<OptimizedDataFrame>,
        /// 左側の結合キー
        left_on: String,
        /// 右側の結合キー
        right_on: String,
        /// 結合タイプ
        join_type: crate::optimized::operations::JoinType,
    },
    /// ソート
    Sort {
        /// ソートする列
        by: String,
        /// 昇順かどうか
        ascending: bool,
    },
}

impl Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Select(columns) => {
                write!(f, "Select({})", columns.join(", "))
            },
            Operation::Filter(condition) => {
                write!(f, "Filter({})", condition)
            },
            Operation::Map(_) => {
                write!(f, "Map(...)")
            },
            Operation::Aggregate { group_by, aggregations } => {
                write!(f, "Aggregate(group_by=[{}], aggs=[", group_by.join(", "))?;
                let aggs: Vec<String> = aggregations.iter()
                    .map(|(col, op, alias)| format!("{} {:?} as {}", col, op, alias))
                    .collect();
                write!(f, "{}])", aggs.join(", "))
            },
            Operation::Join { right: _, left_on, right_on, join_type } => {
                write!(f, "Join({:?}, {} = {})", join_type, left_on, right_on)
            },
            Operation::Sort { by, ascending } => {
                write!(f, "Sort({}, {})", by, if *ascending { "asc" } else { "desc" })
            },
        }
    }
}

impl LazyFrame {
    /// 新しいLazyFrameを作成
    pub fn new(df: OptimizedDataFrame) -> Self {
        Self {
            source: Arc::new(df),
            operations: Vec::new(),
        }
    }
    
    /// 列を選択
    pub fn select(mut self, columns: &[&str]) -> Self {
        let columns = columns.iter().map(|&s| s.to_string()).collect();
        self.operations.push(Operation::Select(columns));
        self
    }
    
    /// フィルタリング
    pub fn filter(mut self, condition: &str) -> Self {
        self.operations.push(Operation::Filter(condition.to_string()));
        self
    }
    
    /// マッピング関数
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(&Column) -> Result<Column> + Send + Sync + 'static,
    {
        self.operations.push(Operation::Map(Arc::new(f)));
        self
    }
    
    /// 集計
    pub fn aggregate<I, J>(mut self, group_by: I, aggregations: J) -> Self
    where
        I: IntoIterator<Item = String>,
        J: IntoIterator<Item = (String, AggregateOp, String)>,
    {
        self.operations.push(Operation::Aggregate {
            group_by: group_by.into_iter().collect(),
            aggregations: aggregations.into_iter().collect(),
        });
        self
    }
    
    /// 結合
    pub fn join(
        mut self,
        right: OptimizedDataFrame,
        left_on: &str,
        right_on: &str,
        join_type: crate::optimized::operations::JoinType,
    ) -> Self {
        self.operations.push(Operation::Join {
            right: Arc::new(right),
            left_on: left_on.to_string(),
            right_on: right_on.to_string(),
            join_type,
        });
        self
    }
    
    /// ソート
    pub fn sort(mut self, by: &str, ascending: bool) -> Self {
        self.operations.push(Operation::Sort {
            by: by.to_string(),
            ascending,
        });
        self
    }
    
    /// 計算グラフを実行して結果を取得
    pub fn execute(self) -> Result<OptimizedDataFrame> {
        let mut df = (*self.source).clone();
        
        for op in self.operations {
            match op {
                Operation::Select(columns) => {
                    let columns_slice = columns.iter().map(|s| s.as_str()).collect::<Vec<_>>();
                    df = df.select(&columns_slice)?;
                },
                Operation::Filter(condition) => {
                    // 並列フィルタリング処理を使用
                    df = df.par_filter(&condition)?;
                },
                Operation::Map(f) => {
                    df = df.par_apply(|view| {
                        f(&view.clone().into_column())
                    })?;
                },
                Operation::Aggregate { group_by, aggregations } => {
                    // グループ化の実装
                    let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
                    
                    // グループ化キーの作成
                    for row_idx in 0..df.row_count() {
                        let mut key = Vec::with_capacity(group_by.len());
                        
                        for col_name in &group_by {
                            let col_view = df.column(col_name.as_str())?;
                            let key_part = match col_view.column_type() {
                                ColumnType::Int64 => {
                                    let col = col_view.as_int64().unwrap();
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NULL".to_string()
                                    }
                                },
                                ColumnType::Float64 => {
                                    let col = col_view.as_float64().unwrap();
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NULL".to_string()
                                    }
                                },
                                ColumnType::String => {
                                    let col = col_view.as_string().unwrap();
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NULL".to_string()
                                    }
                                },
                                ColumnType::Boolean => {
                                    let col = col_view.as_boolean().unwrap();
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NULL".to_string()
                                    }
                                },
                            };
                            key.push(key_part);
                        }
                        
                        groups.entry(key).or_default().push(row_idx);
                    }
                    
                    // 集計結果の格納先
                    let mut result = OptimizedDataFrame::new();
                    
                    // グループ化キー列のデータ
                    let mut group_key_data: HashMap<String, Vec<String>> = HashMap::new();
                    for key in group_by.iter() {
                        group_key_data.insert(key.clone(), Vec::new());
                    }
                    
                    // 集計結果列のデータ
                    let mut agg_result_data: HashMap<String, Vec<f64>> = HashMap::new();
                    for (_, _, alias) in &aggregations {
                        agg_result_data.insert(alias.clone(), Vec::new());
                    }
                    
                    // 各グループに対して集計を実行
                    for (key, row_indices) in groups {
                        // グループ化キーの値を追加
                        for (i, col_name) in group_by.iter().enumerate() {
                            group_key_data.get_mut(col_name).unwrap().push(key[i].clone());
                        }
                        
                        // 集計操作を実行
                        for (col_name, op, alias) in &aggregations {
                            let col_view = df.column(col_name.as_str())?;
                            let result = match (col_view.column_type(), op) {
                                (ColumnType::Int64, AggregateOp::Sum) => {
                                    let col = col_view.as_int64().unwrap();
                                    let mut sum = 0;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            sum += val;
                                        }
                                    }
                                    sum as f64
                                },
                                (ColumnType::Int64, AggregateOp::Mean) => {
                                    let col = col_view.as_int64().unwrap();
                                    let mut sum = 0;
                                    let mut count = 0;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            sum += val;
                                            count += 1;
                                        }
                                    }
                                    if count > 0 {
                                        sum as f64 / count as f64
                                    } else {
                                        0.0
                                    }
                                },
                                (ColumnType::Int64, AggregateOp::Min) => {
                                    let col = col_view.as_int64().unwrap();
                                    let mut min = i64::MAX;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            min = min.min(val);
                                        }
                                    }
                                    if min == i64::MAX {
                                        0.0
                                    } else {
                                        min as f64
                                    }
                                },
                                (ColumnType::Int64, AggregateOp::Max) => {
                                    let col = col_view.as_int64().unwrap();
                                    let mut max = i64::MIN;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            max = max.max(val);
                                        }
                                    }
                                    if max == i64::MIN {
                                        0.0
                                    } else {
                                        max as f64
                                    }
                                },
                                (ColumnType::Float64, AggregateOp::Sum) => {
                                    let col = col_view.as_float64().unwrap();
                                    let mut sum = 0.0;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            sum += val;
                                        }
                                    }
                                    sum
                                },
                                (ColumnType::Float64, AggregateOp::Mean) => {
                                    let col = col_view.as_float64().unwrap();
                                    let mut sum = 0.0;
                                    let mut count = 0;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            sum += val;
                                            count += 1;
                                        }
                                    }
                                    if count > 0 {
                                        sum / count as f64
                                    } else {
                                        0.0
                                    }
                                },
                                (ColumnType::Float64, AggregateOp::Min) => {
                                    let col = col_view.as_float64().unwrap();
                                    let mut min = f64::INFINITY;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            min = min.min(val);
                                        }
                                    }
                                    if min == f64::INFINITY {
                                        0.0
                                    } else {
                                        min
                                    }
                                },
                                (ColumnType::Float64, AggregateOp::Max) => {
                                    let col = col_view.as_float64().unwrap();
                                    let mut max = f64::NEG_INFINITY;
                                    for &idx in &row_indices {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            max = max.max(val);
                                        }
                                    }
                                    if max == f64::NEG_INFINITY {
                                        0.0
                                    } else {
                                        max
                                    }
                                },
                                (_, AggregateOp::Count) => {
                                    row_indices.len() as f64
                                },
                                _ => {
                                    return Err(Error::OperationFailed(format!(
                                        "集計操作 {:?} は列タイプ {:?} に対応していません",
                                        op, col_view.column_type()
                                    )));
                                }
                            };
                            
                            agg_result_data.get_mut(alias).unwrap().push(result);
                        }
                    }
                    
                    // グループ化キー列を追加
                    for (col_name, values) in group_key_data {
                        // 文字列列として追加
                        let col = crate::column::StringColumn::new(values);
                        result.add_column(col_name, Column::String(col))?;
                    }
                    
                    // 集計結果列を追加
                    for (_, _, alias) in &aggregations {
                        let values = agg_result_data.get(alias).unwrap();
                        let col = crate::column::Float64Column::new(values.clone());
                        result.add_column(alias.clone(), Column::Float64(col))?;
                    }
                    
                    df = result;
                },
                Operation::Join { right, left_on, right_on, join_type } => {
                    df = match join_type {
                        crate::optimized::operations::JoinType::Inner => {
                            df.inner_join(&right, left_on.as_str(), right_on.as_str())?
                        },
                        crate::optimized::operations::JoinType::Left => {
                            df.left_join(&right, left_on.as_str(), right_on.as_str())?
                        },
                        crate::optimized::operations::JoinType::Right => {
                            df.right_join(&right, left_on.as_str(), right_on.as_str())?
                        },
                        crate::optimized::operations::JoinType::Outer => {
                            df.outer_join(&right, left_on.as_str(), right_on.as_str())?
                        },
                    };
                },
                Operation::Sort { by, ascending } => {
                    // ソートの実装
                    let col_view = df.column(&by)?;
                    
                    // 元の行インデックスとソートキーのペアを作成
                    let mut pairs: Vec<(usize, String)> = Vec::with_capacity(df.row_count());
                    
                    for row_idx in 0..df.row_count() {
                        let key = match col_view.column_type() {
                            ColumnType::Int64 => {
                                let col = col_view.as_int64().unwrap();
                                if let Ok(Some(val)) = col.get(row_idx) {
                                    val.to_string()
                                } else {
                                    "".to_string()
                                }
                            },
                            ColumnType::Float64 => {
                                let col = col_view.as_float64().unwrap();
                                if let Ok(Some(val)) = col.get(row_idx) {
                                    val.to_string()
                                } else {
                                    "".to_string()
                                }
                            },
                            ColumnType::String => {
                                let col = col_view.as_string().unwrap();
                                if let Ok(Some(val)) = col.get(row_idx) {
                                    val.to_string()
                                } else {
                                    "".to_string()
                                }
                            },
                            ColumnType::Boolean => {
                                let col = col_view.as_boolean().unwrap();
                                if let Ok(Some(val)) = col.get(row_idx) {
                                    val.to_string()
                                } else {
                                    "".to_string()
                                }
                            },
                        };
                        pairs.push((row_idx, key));
                    }
                    
                    // ソート
                    if ascending {
                        pairs.sort_by(|a, b| a.1.cmp(&b.1));
                    } else {
                        pairs.sort_by(|a, b| b.1.cmp(&a.1));
                    }
                    
                    // ソート順に従って新しいDataFrameを作成
                    let mut result = OptimizedDataFrame::new();
                    
                    for col_name in df.column_names() {
                        let col_view = df.column(col_name)?;
                        let col_type = col_view.column_type();
                        
                        match col_type {
                            ColumnType::Int64 => {
                                let col = col_view.as_int64().unwrap();
                                let mut new_data = Vec::with_capacity(df.row_count());
                                
                                for &(idx, _) in &pairs {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        new_data.push(val);
                                    } else {
                                        new_data.push(0);
                                    }
                                }
                                
                                let new_col = crate::column::Int64Column::new(new_data);
                                result.add_column(col_name.to_string(), Column::Int64(new_col))?;
                            },
                            ColumnType::Float64 => {
                                let col = col_view.as_float64().unwrap();
                                let mut new_data = Vec::with_capacity(df.row_count());
                                
                                for &(idx, _) in &pairs {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        new_data.push(val);
                                    } else {
                                        new_data.push(0.0);
                                    }
                                }
                                
                                let new_col = crate::column::Float64Column::new(new_data);
                                result.add_column(col_name.to_string(), Column::Float64(new_col))?;
                            },
                            ColumnType::String => {
                                let col = col_view.as_string().unwrap();
                                let mut new_data = Vec::with_capacity(df.row_count());
                                
                                for &(idx, _) in &pairs {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        new_data.push(val.to_string());
                                    } else {
                                        new_data.push(String::new());
                                    }
                                }
                                
                                let new_col = crate::column::StringColumn::new(new_data);
                                result.add_column(col_name.to_string(), Column::String(new_col))?;
                            },
                            ColumnType::Boolean => {
                                let col = col_view.as_boolean().unwrap();
                                let mut new_data = Vec::with_capacity(df.row_count());
                                
                                for &(idx, _) in &pairs {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        new_data.push(val);
                                    } else {
                                        new_data.push(false);
                                    }
                                }
                                
                                let new_col = crate::column::BooleanColumn::new(new_data);
                                result.add_column(col_name.to_string(), Column::Boolean(new_col))?;
                            },
                        }
                    }
                    
                    df = result;
                },
            }
        }
        
        Ok(df)
    }
    
    /// 最適化された計算グラフを表示
    pub fn explain(&self) -> String {
        let mut result = String::new();
        result.push_str("LazyFrame execution plan:\n");
        result.push_str("----------------------\n");
        result.push_str("SOURCE: OptimizedDataFrame\n");
        
        for (i, op) in self.operations.iter().enumerate() {
            result.push_str(&format!("{}: {:?}\n", i + 1, op));
        }
        
        result
    }
}