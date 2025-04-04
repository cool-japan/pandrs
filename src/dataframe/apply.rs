use std::collections::HashMap;
use std::fmt::Debug;

use super::DataFrame;
use crate::error::{PandRSError, Result};
use crate::na::NA;
use crate::series::Series;
use crate::temporal::{TimeSeries, WindowType};

/// 関数適用のAxis（軸）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// 列単位での適用（各列に関数適用）
    Column = 0,
    /// 行単位での適用（各行に関数適用）
    Row = 1,
}

impl DataFrame {
    /// 関数を各列または各行に適用する
    ///
    /// Pythonのpandas DataFrame.applyに相当します。
    ///
    /// # 引数
    /// * `f` - 適用する関数
    /// * `axis` - 関数適用の軸（列または行）
    /// * `result_name` - 結果のSeriesの名前
    ///
    /// # 戻り値
    /// 関数適用の結果を含む新しいSeries
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::{DataFrame};
    /// use pandrs::dataframe::apply::Axis;
    ///
    /// // DataFrameを作成
    /// let mut df = DataFrame::new();
    /// // 列を追加...
    ///
    /// // 各列の最初の要素を取得（例）
    /// let first_elems = df.apply(|series| series.get(0).unwrap().clone(), Axis::Column, Some("first".to_string()));
    /// ```
    pub fn apply<F, R>(&self, f: F, axis: Axis, result_name: Option<String>) -> Result<Series<R>>
    where
        F: Fn(&Series<String>) -> R,
        R: Debug + Clone,
    {
        match axis {
            Axis::Column => {
                // 各列にfを適用
                let mut results = Vec::with_capacity(self.columns.len());
                
                for col_name in self.column_names() {
                    let col = self.get_column(col_name).unwrap();
                    results.push(f(&col));
                }
                
                // 結果からSeriesを構築
                Series::new(results, result_name)
            }
            Axis::Row => {
                // 現時点では行単位の適用はサポートしない
                Err(PandRSError::NotImplemented(
                    "行単位の関数適用は現在サポートされていません".to_string(),
                ))
            }
        }
    }
    
    /// 関数を各要素に適用する
    ///
    /// Pythonのpandas DataFrame.applymap に相当します。
    ///
    /// # 引数
    /// * `f` - 各要素に適用する関数
    ///
    /// # 戻り値
    /// 関数適用の結果を含む新しいDataFrame
    ///
    /// # 例
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// // DataFrameを作成
    /// let mut df = DataFrame::new();
    /// // 列を追加...
    ///
    /// // 各要素を2倍にする（例）
    /// let doubled = df.applymap(|x| x.parse::<i32>().unwrap_or(0) * 2);
    /// ```
    pub fn applymap<F, R>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str) -> R,
        R: Debug + Clone + ToString,
    {
        let mut result_df = DataFrame::new();
        
        // 各列の各要素に関数を適用
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                transformed.push(f(val).to_string());
            }
            
            // 変換した列をデータフレームに追加
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// 条件に基づいて値を置換する
    ///
    /// Pythonのpandas DataFrame.maskに相当します。
    ///
    /// # 引数
    /// * `condition` - 条件を評価する関数（trueなら置換）
    /// * `other` - 置換する値
    ///
    /// # 戻り値
    /// 条件に一致した値が置換された新しいDataFrame
    pub fn mask<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool,
    {
        let mut result_df = DataFrame::new();
        
        // 各列の条件に合致する要素を置換
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                if condition(val) {
                    transformed.push(other.to_string());
                } else {
                    transformed.push(val.clone().to_string());
                }
            }
            
            // 変換した列をデータフレームに追加
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// 条件に基づいて値を置換する（maskの逆）
    ///
    /// Pythonのpandas DataFrame.whereに相当します。
    ///
    /// # 引数
    /// * `condition` - 条件を評価する関数（falseなら置換）
    /// * `other` - 置換する値
    ///
    /// # 戻り値
    /// 条件に一致しない値が置換された新しいDataFrame
    pub fn where_func<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool,
    {
        let mut result_df = DataFrame::new();
        
        // 各列の条件に合致しない要素を置換
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                if !condition(val) {
                    transformed.push(other.to_string());
                } else {
                    transformed.push(val.clone().to_string());
                }
            }
            
            // 変換した列をデータフレームに追加
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// 値を対応する値に置換する
    ///
    /// Pythonのpandas DataFrame.replaceに相当します。
    ///
    /// # 引数
    /// * `replace_map` - 置換する値のマップ
    ///
    /// # 戻り値
    /// 値が置換された新しいDataFrame
    pub fn replace(&self, replace_map: &HashMap<String, String>) -> Result<DataFrame> {
        let mut result_df = DataFrame::new();
        
        // 各列の各要素を置換
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                match replace_map.get(val) {
                    Some(replacement) => transformed.push(replacement.clone()),
                    None => transformed.push(val.clone().to_string()),
                }
            }
            
            // 変換した列をデータフレームに追加
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// 重複行を検出する
    ///
    /// Pythonのpandas DataFrame.duplicatedに相当します。
    ///
    /// # 引数
    /// * `subset` - 重複を確認する列のサブセット（Noneの場合はすべての列）
    /// * `keep` - どの重複行を保持するか（"first"=最初の行のみ、"last"=最後の行のみ、None=すべて重複としてマーク）
    ///
    /// # 戻り値
    /// 各行が重複かどうかを示すブール値のSeries
    pub fn duplicated(
        &self,
        subset: Option<&[String]>,
        keep: Option<&str>,
    ) -> Result<Series<bool>> {
        let columns_to_check = match subset {
            Some(cols) => {
                // 指定された列が存在するか確認
                for col in cols {
                    if !self.contains_column(col) {
                        return Err(PandRSError::Column(format!(
                            "列 '{}' が存在しません",
                            col
                        )));
                    }
                }
                cols.to_vec()
            }
            None => self.column_names().to_vec(),
        };
        
        let row_count = self.row_count();
        let mut duplicated = vec![false; row_count];
        let mut seen = HashMap::new();
        
        // 各行を文字列に変換して一意なキーを作成
        for i in 0..row_count {
            let mut row_key = String::new();
            
            for col in &columns_to_check {
                let value = self.get_column(col).unwrap().values()[i].clone();
                row_key.push_str(&value);
                row_key.push('\0'); // 区切り文字
            }
            
            match keep {
                Some("first") => {
                    // 初出以外をマーク
                    if seen.contains_key(&row_key) {
                        duplicated[i] = true;
                    } else {
                        seen.insert(row_key, i);
                    }
                }
                Some("last") => {
                    // 現在の行をマークし、後で最後の行のマークを外す
                    if seen.contains_key(&row_key) {
                        duplicated[*seen.get(&row_key).unwrap()] = true;
                    }
                    seen.insert(row_key, i);
                }
                _ => {
                    // 2回目以降の出現をすべてマーク
                    if seen.contains_key(&row_key) {
                        duplicated[i] = true;
                        duplicated[*seen.get(&row_key).unwrap()] = true;
                    } else {
                        seen.insert(row_key, i);
                    }
                }
            }
        }
        
        // 結果をSeriesとして返す
        Series::new(duplicated, Some("duplicated".to_string()))
    }
    
    /// 重複行を削除する
    ///
    /// Pythonのpandas DataFrame.drop_duplicatesに相当します。
    ///
    /// # 引数
    /// * `subset` - 重複を確認する列のサブセット（Noneの場合はすべての列）
    /// * `keep` - どの重複行を保持するか（"first"=最初の行のみ、"last"=最後の行のみ、None=すべての重複を削除）
    ///
    /// # 戻り値
    /// 重複行が削除された新しいDataFrame
    pub fn drop_duplicates(
        &self,
        subset: Option<&[String]>,
        keep: Option<&str>,
    ) -> Result<DataFrame> {
        // 重複行を特定
        let is_duplicated = self.duplicated(subset, keep)?;
        
        let mut result_df = DataFrame::new();
        
        // 各列で非重複行のみを抽出
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut filtered = Vec::new();
            
            for i in 0..col.len() {
                if !is_duplicated.values()[i] {
                    filtered.push(col.values()[i].clone());
                }
            }
            
            // フィルタリングした列をデータフレームに追加
            let series = Series::new(filtered, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// 固定長ウィンドウ（移動ウィンドウ）操作を適用
    /// 
    /// # 引数
    /// * `window_size` - ウィンドウサイズ
    /// * `column_name` - 対象の列名
    /// * `operation` - ウィンドウ操作の種類 ("mean", "sum", "std", "min", "max")
    /// * `result_column` - 結果を格納する列名（省略時は "{column_name}_{operation}_{window_size}" という名前になる）
    /// 
    /// # 戻り値
    /// 操作が適用されたDataFrame
    pub fn rolling(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        result_column: Option<&str>,
    ) -> Result<DataFrame> {
        // 対象の列が存在するか確認
        let series = match self.get_column(column_name) {
            Some(s) => s,
            None => return Err(PandRSError::KeyNotFound(column_name.to_string())),
        };
        
        // 結果を格納する列名
        let result_col = result_column.unwrap_or(&format!(
            "{}_{}_{}", column_name, operation, window_size
        )).to_string();
        
        // シリーズから数値データを抽出
        let values: Vec<NA<f64>> = series.values().iter()
            .map(|s| {
                match s.parse::<f64>() {
                    Ok(num) => NA::Value(num),
                    Err(_) => NA::NA,
                }
            })
            .collect();
        
        // タイムスタンプはインデックスから取得
        // （実際のタイムスタンプが無い場合はダミーの日付を使用）
        let timestamps = self.get_index().to_datetime_vec()?;
        
        // TimeSeries を作成
        let time_series = TimeSeries::new(
            values,
            timestamps,
            Some(column_name.to_string()),
        )?;
        
        // ウィンドウ操作を実行
        let window_result = match operation.to_lowercase().as_str() {
            "mean" => time_series.rolling(window_size)?.mean()?,
            "sum" => time_series.rolling(window_size)?.sum()?,
            "std" => time_series.rolling(window_size)?.std(1)?,
            "min" => time_series.rolling(window_size)?.min()?,
            "max" => time_series.rolling(window_size)?.max()?,
            _ => return Err(PandRSError::Operation(format!(
                "サポートされていないウィンドウ操作: {}", operation
            ))),
        };
        
        // 結果をString Seriesに変換
        let result_strings: Vec<String> = window_result.values().iter()
            .map(|v| match v {
                NA::Value(val) => val.to_string(),
                NA::NA => "NA".to_string(),
            })
            .collect();
        
        // 結果のSeriesを作成
        let result_series = Series::new(
            result_strings,
            Some(result_col.clone()),
        )?;
        
        // 元のDataFrameに結果列を追加
        let mut result_df = self.clone();
        result_df.add_column(result_col, result_series)?;
        
        Ok(result_df)
    }
    
    /// 拡大ウィンドウ操作を適用
    /// 
    /// # 引数
    /// * `min_periods` - 最小期間サイズ
    /// * `column_name` - 対象の列名
    /// * `operation` - ウィンドウ操作の種類 ("mean", "sum", "std", "min", "max")
    /// * `result_column` - 結果を格納する列名（省略時は "{column_name}_expanding_{operation}" という名前になる）
    /// 
    /// # 戻り値
    /// 操作が適用されたDataFrame
    pub fn expanding(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        result_column: Option<&str>,
    ) -> Result<DataFrame> {
        // 対象の列が存在するか確認
        let series = match self.get_column(column_name) {
            Some(s) => s,
            None => return Err(PandRSError::KeyNotFound(column_name.to_string())),
        };
        
        // 結果を格納する列名
        let result_col = result_column.unwrap_or(&format!(
            "{}_expanding_{}", column_name, operation
        )).to_string();
        
        // シリーズから数値データを抽出
        let values: Vec<NA<f64>> = series.values().iter()
            .map(|s| {
                match s.parse::<f64>() {
                    Ok(num) => NA::Value(num),
                    Err(_) => NA::NA,
                }
            })
            .collect();
        
        // タイムスタンプはインデックスから取得
        let timestamps = self.get_index().to_datetime_vec()?;
        
        // TimeSeries を作成
        let time_series = TimeSeries::new(
            values,
            timestamps,
            Some(column_name.to_string()),
        )?;
        
        // ウィンドウ操作を実行
        let window_result = match operation.to_lowercase().as_str() {
            "mean" => time_series.expanding(min_periods)?.mean()?,
            "sum" => time_series.expanding(min_periods)?.sum()?,
            "std" => time_series.expanding(min_periods)?.std(1)?,
            "min" => time_series.expanding(min_periods)?.min()?,
            "max" => time_series.expanding(min_periods)?.max()?,
            _ => return Err(PandRSError::Operation(format!(
                "サポートされていないウィンドウ操作: {}", operation
            ))),
        };
        
        // 結果をString Seriesに変換
        let result_strings: Vec<String> = window_result.values().iter()
            .map(|v| match v {
                NA::Value(val) => val.to_string(),
                NA::NA => "NA".to_string(),
            })
            .collect();
        
        // 結果のSeriesを作成
        let result_series = Series::new(
            result_strings,
            Some(result_col.clone()),
        )?;
        
        // 元のDataFrameに結果列を追加
        let mut result_df = self.clone();
        result_df.add_column(result_col, result_series)?;
        
        Ok(result_df)
    }
    
    /// 指数加重ウィンドウ操作を適用
    /// 
    /// # 引数
    /// * `column_name` - 対象の列名
    /// * `operation` - ウィンドウ操作の種類 ("mean", "std")
    /// * `span` - 半減期（alphaとは同時に指定できない）
    /// * `alpha` - 減衰係数 0.0 < alpha <= 1.0（spanとは同時に指定できない）
    /// * `result_column` - 結果を格納する列名（省略時は "{column_name}_ewm_{operation}" という名前になる）
    /// 
    /// # 戻り値
    /// 操作が適用されたDataFrame
    pub fn ewm(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        result_column: Option<&str>,
    ) -> Result<DataFrame> {
        // spanとalphaの両方が指定された場合はエラー
        if span.is_some() && alpha.is_some() {
            return Err(PandRSError::Consistency(
                "span と alpha は同時に指定できません。いずれか一方を指定してください。".to_string(),
            ));
        }
        
        // 対象の列が存在するか確認
        let series = match self.get_column(column_name) {
            Some(s) => s,
            None => return Err(PandRSError::KeyNotFound(column_name.to_string())),
        };
        
        // 結果を格納する列名
        let result_col = result_column.unwrap_or(&format!(
            "{}_ewm_{}", column_name, operation
        )).to_string();
        
        // シリーズから数値データを抽出
        let values: Vec<NA<f64>> = series.values().iter()
            .map(|s| {
                match s.parse::<f64>() {
                    Ok(num) => NA::Value(num),
                    Err(_) => NA::NA,
                }
            })
            .collect();
        
        // タイムスタンプはインデックスから取得
        let timestamps = self.get_index().to_datetime_vec()?;
        
        // TimeSeries を作成
        let time_series = TimeSeries::new(
            values,
            timestamps,
            Some(column_name.to_string()),
        )?;
        
        // ウィンドウ操作を実行
        let window_result = match operation.to_lowercase().as_str() {
            "mean" => time_series.ewm(span, alpha, false)?.mean()?,
            "std" => time_series.ewm(span, alpha, false)?.std(1)?,
            _ => return Err(PandRSError::Operation(format!(
                "サポートされていないEWM操作: {}", operation
            ))),
        };
        
        // 結果をString Seriesに変換
        let result_strings: Vec<String> = window_result.values().iter()
            .map(|v| match v {
                NA::Value(val) => val.to_string(),
                NA::NA => "NA".to_string(),
            })
            .collect();
        
        // 結果のSeriesを作成
        let result_series = Series::new(
            result_strings,
            Some(result_col.clone()),
        )?;
        
        // 元のDataFrameに結果列を追加
        let mut result_df = self.clone();
        result_df.add_column(result_col, result_series)?;
        
        Ok(result_df)
    }
}