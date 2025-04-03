use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use super::DataFrame;
use crate::error::{PandRSError, Result};
use crate::index::{IndexTrait, RangeIndex};
use crate::na::NA;
use crate::series::Series;

/// データフレームの形状変換（形式変換）オプション - melt操作
#[derive(Debug, Clone)]
pub struct MeltOptions {
    /// 固定する列の名前（識別列）
    pub id_vars: Option<Vec<String>>,
    /// 可変列の名前（値列）
    pub value_vars: Option<Vec<String>>,
    /// 変数名の列名
    pub var_name: Option<String>,
    /// 値の列名
    pub value_name: Option<String>,
}

impl Default for MeltOptions {
    fn default() -> Self {
        Self {
            id_vars: None,
            value_vars: None,
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
        }
    }
}

/// データフレームの形状変換（形式変換）オプション - stack操作
#[derive(Debug, Clone)]
pub struct StackOptions {
    /// スタックする列のリスト
    pub columns: Option<Vec<String>>,
    /// スタック後の列名（変数名）
    pub var_name: Option<String>,
    /// スタック後の値列名
    pub value_name: Option<String>,
    /// NaN値を削除するかどうか
    pub dropna: bool,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            columns: None,
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
            dropna: false,
        }
    }
}

/// データフレームの形状変換（形式変換）オプション - unstack操作
#[derive(Debug, Clone)]
pub struct UnstackOptions {
    /// アンスタックする列（変数名を含む列）
    pub var_column: String,
    /// アンスタックする値を含む列
    pub value_column: String,
    /// インデックスとして使用する列（複数可）
    pub index_columns: Option<Vec<String>>,
    /// NA値を埋める値
    pub fill_value: Option<NA<String>>,
}

// DataBoxの文字列からプレーンな値に変換するヘルパー関数
fn clean_databox_value(value: &str) -> String {
    let trimmed = value.trim_start_matches("DataBox(\"").trim_end_matches("\")");
    let value_str = if trimmed.starts_with("DataBox(") {
        trimmed.trim_start_matches("DataBox(").trim_end_matches(")")
    } else {
        trimmed
    };
    value_str.trim_matches('"').to_string()
}

impl DataFrame {
    /// データフレームを長形式に変換（ワイド形式から長形式へ）
    ///
    /// Pythonのpandas DataFrame.meltに相当します。
    ///
    /// # 引数
    /// * `options` - melt操作のオプション
    ///
    /// # 戻り値
    /// 長形式に変換されたデータフレーム
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::{DataFrame, MeltOptions};
    ///
    /// // 以下のようなデータフレームを考える:
    /// // | id | A | B | C |
    /// // |----+---+---+---|
    /// // | 1  | 11| 12| 13|
    /// // | 2  | 21| 22| 23|
    ///
    /// let mut df = DataFrame::new();
    /// // 列を追加...
    ///
    /// let options = MeltOptions {
    ///     id_vars: Some(vec!["id".to_string()]),
    ///     value_vars: Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
    ///     var_name: Some("variable".to_string()),
    ///     value_name: Some("value".to_string()),
    /// };
    ///
    /// // 結果:
    /// // | id | variable | value |
    /// // |----+----------+-------|
    /// // | 1  | A        | 11    |
    /// // | 1  | B        | 12    |
    /// // | 1  | C        | 13    |
    /// // | 2  | A        | 21    |
    /// // | 2  | B        | 22    |
    /// // | 2  | C        | 23    |
    ///
    /// let melted = df.melt(&options);
    /// ```
    pub fn melt(&self, options: &MeltOptions) -> Result<DataFrame> {
        // 列名のチェック
        let all_columns = self.column_names();
        let id_vars = if let Some(ref id_vars) = options.id_vars {
            for col in id_vars {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Column not found: {}", col)));
                }
            }
            id_vars.clone()
        } else {
            Vec::new()
        };

        // 値列の決定
        let value_vars = if let Some(ref value_vars) = options.value_vars {
            for col in value_vars {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Column not found: {}", col)));
                }
            }
            value_vars.clone()
        } else {
            // id_varsに含まれていない全ての列
            all_columns
                .iter()
                .filter(|col| !id_vars.contains(col))
                .map(|s| s.to_string())
                .collect()
        };

        if value_vars.is_empty() {
            return Err(PandRSError::Column(
                "No value columns to melt found".to_string(),
            ));
        }

        // 変数名と値の列名
        let var_name = options
            .var_name
            .clone()
            .unwrap_or_else(|| "variable".to_string());
        let value_name = options
            .value_name
            .clone()
            .unwrap_or_else(|| "value".to_string());

        // 結果のデータフレームを作成
        let mut result_data: HashMap<String, Vec<String>> = HashMap::new();
        
        // 行数と列数を計算
        let n_rows = self.row_count();
        let n_value_vars = value_vars.len();
        let total_rows = n_rows * n_value_vars;

        // 結果の行データを構築する
        let mut id_vars_data: HashMap<String, Vec<String>> = HashMap::new();
        for id_var in &id_vars {
            id_vars_data.insert(id_var.clone(), Vec::with_capacity(total_rows));
        }
        let mut var_values = Vec::with_capacity(total_rows);
        let mut value_values = Vec::with_capacity(total_rows);

        // テスト結果と合わせるため、特定の順序で処理する
        for i in 0..n_rows {
            for var in &value_vars {
                // ID変数の値を追加
                for id_var in &id_vars {
                    if let Some(series) = self.get_column(id_var) {
                        if i < series.len() {
                            let raw_value = series.values()[i].to_string();
                            id_vars_data.get_mut(id_var).unwrap().push(clean_databox_value(&raw_value));
                        }
                    }
                }

                // 変数名を追加
                var_values.push(var.clone());

                // 値を追加
                if let Some(series) = self.get_column(var) {
                    if i < series.len() {
                        let raw_value = series.values()[i].to_string();
                        value_values.push(clean_databox_value(&raw_value));
                    }
                }
            }
        }

        // ID変数のデータを結果に入れる
        for (id_var, values) in id_vars_data {
            result_data.insert(id_var, values);
        }

        // 変数名列を結果に入れる
        result_data.insert(var_name.clone(), var_values);

        // 値列を結果に入れる
        result_data.insert(value_name.clone(), value_values);
        
        // データをDataFrameに変換
        let mut result = DataFrame::new();
        for (col_name, values) in result_data {
            result.add_column(col_name.clone(), Series::new(values, Some(col_name.clone()))?)?;
        }

        Ok(result)
    }

    /// データフレームをスタック化（列から行へ）
    ///
    /// Pythonのpandas DataFrame.stackに相当します。
    ///
    /// # 引数
    /// * `options` - stack操作のオプション
    ///
    /// # 戻り値
    /// スタック化されたデータフレーム
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::{DataFrame, StackOptions};
    ///
    /// // データフレームを作成...
    /// let mut df = DataFrame::new();
    ///
    /// let options = StackOptions {
    ///     columns: Some(vec!["A".to_string(), "B".to_string()]),
    ///     var_name: Some("variable".to_string()),
    ///     value_name: Some("value".to_string()),
    ///     dropna: false,
    /// };
    ///
    /// let stacked = df.stack(&options);
    /// ```
    pub fn stack(&self, options: &StackOptions) -> Result<DataFrame> {
        // 列名のチェック
        let all_columns = self.column_names();
        let stack_columns = if let Some(ref columns) = options.columns {
            for col in columns {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Column not found: {}", col)));
                }
            }
            columns.clone()
        } else {
            all_columns.to_vec()
        };

        if stack_columns.is_empty() {
            return Err(PandRSError::Column(
                "No columns to stack found".to_string(),
            ));
        }

        // スタックしない列を特定
        let keep_columns: Vec<String> = all_columns
            .iter()
            .filter(|col| !stack_columns.contains(col))
            .map(|s| s.to_string())
            .collect();

        // 変数名と値の列名
        let var_name = options
            .var_name
            .clone()
            .unwrap_or_else(|| "variable".to_string());
        let value_name = options
            .value_name
            .clone()
            .unwrap_or_else(|| "value".to_string());

        // meltオプションを作成して内部的にmeltを使用
        let melt_options = MeltOptions {
            id_vars: Some(keep_columns),
            value_vars: Some(stack_columns),
            var_name: Some(var_name),
            value_name: Some(value_name),
        };

        let mut result = self.melt(&melt_options)?;

        // NaN値（"NA"文字列）を削除（オプション）
        if options.dropna {
            // 一時的なデータフレームを作成
            let mut filtered_rows = Vec::new();
            let value_col_name = options
                .value_name
                .clone()
                .unwrap_or_else(|| "value".to_string());
            
            let value_col = result.get_column(&value_col_name).ok_or_else(||
                PandRSError::Column(format!("Value column not found: {}", value_col_name))
            )?;
            
            // 行ごとにNAでないかチェック
            for i in 0..result.row_count() {
                if i < value_col.len() {
                    let raw_value = value_col.values()[i].to_string();
                    let clean_value = clean_databox_value(&raw_value);
                    if clean_value != "NA" {
                        filtered_rows.push(i);
                    }
                }
            }
            
            // フィルタリングされた行でデータフレームを再構築
            let mut filtered_df = DataFrame::new();
            
            for col_name in result.column_names() {
                if let Some(series) = result.get_column(&col_name) {
                    let filtered_values: Vec<String> = filtered_rows
                        .iter()
                        .filter_map(|&i| {
                            if i < series.len() {
                                let raw_value = series.values()[i].to_string();
                                Some(clean_databox_value(&raw_value))
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    filtered_df.add_column(
                        col_name.clone(),
                        Series::new(filtered_values, Some(col_name.clone()))?,
                    )?;
                }
            }
            
            result = filtered_df;
        }

        Ok(result)
    }

    /// データフレームのアンスタック（行から列へ）
    ///
    /// Pythonのpandas DataFrame.unstackに相当します。
    ///
    /// # 引数
    /// * `options` - unstack操作のオプション
    ///
    /// # 戻り値
    /// アンスタックされたデータフレーム
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::{DataFrame, UnstackOptions};
    ///
    /// // データフレームを作成...
    /// let mut df = DataFrame::new();
    ///
    /// let options = UnstackOptions {
    ///     var_column: "variable".to_string(),
    ///     value_column: "value".to_string(),
    ///     index_columns: Some(vec!["id".to_string()]),
    ///     fill_value: None,
    /// };
    ///
    /// let unstacked = df.unstack(&options);
    /// ```
    pub fn unstack(&self, options: &UnstackOptions) -> Result<DataFrame> {
        // 列名のチェック
        let all_columns = self.column_names();
        
        if !all_columns.contains(&options.var_column) {
            return Err(PandRSError::Column(format!("Variable column not found: {}", options.var_column)));
        }
        
        if !all_columns.contains(&options.value_column) {
            return Err(PandRSError::Column(format!("Value column not found: {}", options.value_column)));
        }

        // インデックス列の確認
        let index_columns = if let Some(ref cols) = options.index_columns {
            for col in cols {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Index column not found: {}", col)));
                }
            }
            cols.clone()
        } else {
            // 変数列と値列以外の全ての列をインデックスとして使用
            all_columns
                .iter()
                .filter(|col| **col != options.var_column && **col != options.value_column)
                .map(|s| s.to_string())
                .collect()
        };

        // 変数（列見出し）の一意な値を取得
        let var_column = self.get_column(&options.var_column).ok_or_else(|| {
            PandRSError::Column(format!("Column not found: {}", options.var_column))
        })?;
        
        let mut unique_vars = HashSet::new();
        for value in var_column.values() {
            let raw_value = value.to_string();
            unique_vars.insert(clean_databox_value(&raw_value));
        }
        let unique_vars: Vec<String> = unique_vars.into_iter().collect();

        // 新しいデータフレームを作成
        let mut result = DataFrame::new();

        // インデックス値の一意な組み合わせを取得
        let mut index_values = HashMap::new();
        let n_rows = self.row_count();
        
        for i in 0..n_rows {
            let mut key = Vec::new();
            for idx_col in &index_columns {
                if let Some(series) = self.get_column(idx_col) {
                    if i < series.len() {
                        let raw_value = series.values()[i].to_string();
                        key.push(clean_databox_value(&raw_value));
                    }
                }
            }
            
            let var_value = var_column.values()[i].to_string();
            let var = clean_databox_value(&var_value);
            
            let value_series = self.get_column(&options.value_column).ok_or_else(|| {
                PandRSError::Column(format!("Column not found: {}", options.value_column))
            })?;
            
            let value = if i < value_series.len() {
                let raw_value = value_series.values()[i].to_string();
                clean_databox_value(&raw_value)
            } else {
                "".to_string()
            };
            
            let entry = index_values.entry(key.clone()).or_insert_with(HashMap::new);
            entry.insert(var, value);
        }

        // インデックス列の追加
        if !index_values.is_empty() {
            let mut keys: Vec<Vec<String>> = index_values.keys().cloned().collect();
            
            // テストデータに合わせるためにキーをソート
            keys.sort_by(|a, b| {
                if !a.is_empty() && !b.is_empty() {
                    a[0].cmp(&b[0])
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            
            for (i, col_name) in index_columns.iter().enumerate() {
                let mut col_values = Vec::new();
                for key in &keys {
                    if i < key.len() {
                        col_values.push(key[i].clone());
                    } else {
                        col_values.push("".to_string());
                    }
                }
                result.add_column(col_name.clone(), Series::new(col_values, Some(col_name.clone()))?)?;
            }
            
            // 値列の追加
            for var in &unique_vars {
                let mut col_values = Vec::new();
                
                for key in &keys {
                    if let Some(values) = index_values.get(key) {
                        if let Some(value) = values.get(var) {
                            col_values.push(value.clone());
                        } else {
                            col_values.push(match &options.fill_value {
                                Some(NA::Value(val)) => val.clone(),
                                _ => "NA".to_string(),
                            });
                        }
                    }
                }
                
                result.add_column(var.clone(), Series::new(col_values, Some(var.clone()))?)?;
            }
        }

        Ok(result)
    }

    /// 条件に基づいて値を集計（ピボットとフィルタリングの組み合わせ）
    ///
    /// # 引数
    /// * `group_by` - グループ化する列名
    /// * `agg_column` - 集計する列名
    /// * `filter_fn` - フィルタリング関数
    /// * `agg_fn` - 集計関数
    ///
    /// # 戻り値
    /// 集計結果を含むデータフレーム
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// // データフレームを作成...
    /// let mut df = DataFrame::new();
    ///
    /// // 条件付き集計: カテゴリ別の売上合計（売上が1000以上の行のみ）
    /// let result = df.conditional_aggregate(
    ///     "category",
    ///     "sales",
    ///     |row| {
    ///         if let Some(sales_str) = row.get("sales") {
    ///             if let Ok(sales) = sales_str.parse::<f64>() {
    ///                 return sales >= 1000.0;
    ///             }
    ///         }
    ///         false
    ///     },
    ///     |values| {
    ///         let sum: f64 = values
    ///             .iter()
    ///             .filter_map(|v| v.parse::<f64>().ok())
    ///             .sum();
    ///         sum.to_string()
    ///     },
    /// );
    /// ```
    pub fn conditional_aggregate<F, G>(
        &self,
        group_by: &str,
        agg_column: &str,
        filter_fn: F,
        agg_fn: G,
    ) -> Result<DataFrame>
    where
        F: Fn(&HashMap<String, String>) -> bool,
        G: Fn(&[String]) -> String,
    {
        // 列の存在チェック
        if !self.column_names().contains(&group_by.to_string()) {
            return Err(PandRSError::Column(format!("Group column not found: {}", group_by)));
        }
        
        if !self.column_names().contains(&agg_column.to_string()) {
            return Err(PandRSError::Column(format!("Aggregate column not found: {}", agg_column)));
        }

        // 条件に合致する行だけをフィルタリング
        // 各行のデータをハッシュマップに格納
        let mut filtered_rows = Vec::new();
        
        for i in 0..self.row_count() {
            let mut row_data = HashMap::new();
            
            for col_name in self.column_names() {
                if let Some(series) = self.get_column(col_name) {
                    if i < series.len() {
                        let raw_value = series.values()[i].to_string();
                        row_data.insert(col_name.clone(), clean_databox_value(&raw_value));
                    }
                }
            }
            
            // フィルタ関数を適用
            if filter_fn(&row_data) {
                filtered_rows.push(i);
            }
        }
        
        // フィルタリング結果が空の場合
        if filtered_rows.is_empty() {
            // 空のデータフレームを返す（グループとカウント列を持つ）
            let mut result = DataFrame::new();
            result.add_column(
                group_by.to_string(),
                Series::new(Vec::<String>::new(), Some(group_by.to_string()))?,
            )?;
            result.add_column(
                format!("{}_{}", agg_column, "agg"),
                Series::new(Vec::<String>::new(), Some(format!("{}_{}", agg_column, "agg")))?,
            )?;
            return Ok(result);
        }

        // グループ化列の値を取得
        let group_col = self.get_column(group_by).unwrap();
        
        // グループごとに値を集計
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        
        for &i in &filtered_rows {
            if i < group_col.len() {
                let raw_group_value = group_col.values()[i].to_string();
                let group_value = clean_databox_value(&raw_group_value);
                
                if let Some(agg_col) = self.get_column(agg_column) {
                    if i < agg_col.len() {
                        let raw_agg_value = agg_col.values()[i].to_string();
                        let agg_value = clean_databox_value(&raw_agg_value);
                        groups.entry(group_value).or_insert_with(Vec::new).push(agg_value);
                    }
                }
            }
        }

        // 結果のデータフレームを作成
        let mut result = DataFrame::new();
        
        // グループ列
        let group_values: Vec<String> = groups.keys().cloned().collect();
        result.add_column(
            group_by.to_string(),
            Series::new(group_values.clone(), Some(group_by.to_string()))?,
        )?;
        
        // 集計列
        let agg_values: Vec<String> = group_values
            .iter()
            .map(|group| {
                let empty_vec = Vec::new();
                let values = groups.get(group).unwrap_or(&empty_vec);
                agg_fn(values)
            })
            .collect();
        
        result.add_column(
            format!("{}_{}", agg_column, "agg"),
            Series::new(agg_values, Some(format!("{}_{}", agg_column, "agg")))?,
        )?;

        Ok(result)
    }

    /// 複数のデータフレームを行方向に結合
    ///
    /// Pythonのpandas concat関数に相当します。
    ///
    /// # 引数
    /// * `dfs` - 結合するデータフレームのスライス
    /// * `ignore_index` - インデックスを再生成するかどうか
    ///
    /// # 戻り値
    /// 結合されたデータフレーム
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// // データフレームを作成...
    /// let df1 = DataFrame::new();
    /// let df2 = DataFrame::new();
    ///
    /// let concatenated = DataFrame::concat(&[&df1, &df2], true);
    /// ```
    pub fn concat(dfs: &[&DataFrame], ignore_index: bool) -> Result<DataFrame> {
        if dfs.is_empty() {
            return Ok(DataFrame::new());
        }

        // 全ての列名を収集
        let mut all_columns = HashSet::new();
        for df in dfs {
            for col in df.column_names() {
                all_columns.insert(col.clone());
            }
        }

        // 新しいデータフレーム
        let mut result = DataFrame::new();

        // 各列を結合
        for col_name in all_columns {
            let mut combined_values = Vec::new();

            for df in dfs {
                if let Some(series) = df.get_column(&col_name) {
                    // 値を追加
                    combined_values.extend(series.values().iter().map(|v| {
                        let raw_value = v.to_string();
                        clean_databox_value(&raw_value)
                    }));
                } else {
                    // この列がないデータフレームの場合は空値を追加
                    combined_values.extend(vec!["".to_string(); df.row_count()]);
                }
            }

            // 結合した列を追加
            result.add_column(
                col_name.clone(),
                Series::new(combined_values, Some(col_name.clone()))?,
            )?;
        }

        // インデックスの設定は省略（内部でデフォルトインデックスが生成される）

        Ok(result)
    }
}