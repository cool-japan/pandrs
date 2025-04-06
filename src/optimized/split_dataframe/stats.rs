//! OptimizedDataFrame向け統計関数モジュール
//! 
//! このモジュールは、データ分析のための統計機能を提供します。
//! ANOVA、t検定、カイ二乗検定、Mann-Whitney U検定などをサポートします。

use std::collections::HashMap;
use crate::error::Result;
use crate::stats::{
    self, DescriptiveStats, TTestResult, AnovaResult, 
    MannWhitneyResult, ChiSquareResult, LinearRegressionResult
};
use crate::column::{Column, ColumnTrait};
use crate::optimized::split_dataframe::OptimizedDataFrame;

/// OptimizedDataFrame用の統計結果型
#[derive(Debug, Clone)]
pub enum StatResult {
    /// 記述統計の結果
    Descriptive(DescriptiveStats),
    /// t検定の結果
    TTest(TTestResult),
    /// 分散分析の結果
    Anova(AnovaResult),
    /// Mann-Whitney U検定の結果
    MannWhitneyU(MannWhitneyResult),
    /// カイ二乗検定の結果
    ChiSquare(ChiSquareResult),
    /// 回帰分析の結果
    LinearRegression(LinearRegressionResult),
}

/// 記述統計の結果出力形式
#[derive(Debug, Clone)]
pub struct StatDescribe {
    /// 統計量のマップ
    pub stats: HashMap<String, f64>,
    /// 統計量の列挙
    pub stats_list: Vec<(String, f64)>,
}

/// OptimizedDataFrameの統計機能拡張実装
impl OptimizedDataFrame {
    /// 特定の列に対する基本統計量を計算
    /// 
    /// # 引数
    /// * `column_name` - 統計を計算する列の名前
    /// 
    /// # 戻り値
    /// 記述統計量を含む構造体
    pub fn describe(&self, column_name: &str) -> Result<StatDescribe> {
        let col = self.column(column_name)?;
        
        if let Some(float_col) = col.as_float64() {
            // 浮動小数点列の場合
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| float_col.get(i).ok().flatten())
                .collect();
            
            // statsモジュールを使用
            let stats = stats::describe(&values)?;
            
            // 結果をHashMapに格納
            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);
            
            // 順序付きリストも提供
            let stats_list = vec![
                ("count".to_string(), stats.count as f64),
                ("mean".to_string(), stats.mean),
                ("std".to_string(), stats.std),
                ("min".to_string(), stats.min),
                ("25%".to_string(), stats.q1),
                ("50%".to_string(), stats.median),
                ("75%".to_string(), stats.q3),
                ("max".to_string(), stats.max),
            ];
            
            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);
            
            Ok(StatDescribe { stats: result, stats_list })
        } else if let Some(int_col) = col.as_int64() {
            // 整数列の場合は浮動小数点に変換して計算
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                .collect();
            
            // statsモジュールを使用
            let stats = stats::describe(&values)?;
            
            // 結果をHashMapに格納
            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);
            
            // 順序付きリストも提供
            let stats_list = vec![
                ("count".to_string(), stats.count as f64),
                ("mean".to_string(), stats.mean),
                ("std".to_string(), stats.std),
                ("min".to_string(), stats.min),
                ("25%".to_string(), stats.q1),
                ("50%".to_string(), stats.median),
                ("75%".to_string(), stats.q3),
                ("max".to_string(), stats.max),
            ];
            
            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);
            
            Ok(StatDescribe { stats: result, stats_list })
        } else {
            Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", column_name)))
        }
    }
    
    /// 複数の列に対する記述統計をまとめて計算
    /// 
    /// # 戻り値
    /// 列名から統計結果へのマッピング
    pub fn describe_all(&self) -> Result<HashMap<String, StatDescribe>> {
        let mut results = HashMap::new();
        
        for col_name in self.column_names() {
            // 数値列のみを対象に
            let col = self.column(col_name)?;
            if col.as_float64().is_some() || col.as_int64().is_some() {
                if let Ok(desc) = self.describe(col_name) {
                    results.insert(col_name.to_string(), desc);
                }
            }
        }
        
        Ok(results)
    }
    
    /// 2つの列に対してt検定を実行
    /// 
    /// # 引数
    /// * `col1` - 1つ目の列名
    /// * `col2` - 2つ目の列名
    /// * `alpha` - 有意水準 (デフォルト: 0.05)
    /// * `equal_var` - 等分散を仮定するか (デフォルト: true)
    /// 
    /// # 戻り値
    /// t検定の結果
    pub fn ttest(&self, col1: &str, col2: &str, alpha: Option<f64>, equal_var: Option<bool>) -> Result<TTestResult> {
        let alpha = alpha.unwrap_or(0.05);
        let equal_var = equal_var.unwrap_or(true);
        
        // 列データを取得
        let column1 = self.column(col1)?;
        let column2 = self.column(col2)?;
        
        // 浮動小数点ベクトルに変換
        let values1: Vec<f64> = match column1 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            },
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            },
            _ => return Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", col1))),
        };
        
        let values2: Vec<f64> = match column2 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            },
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            },
            _ => return Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", col2))),
        };
        
        // t検定を実行
        stats::ttest(&values1, &values2, alpha, equal_var)
    }
    
    /// 一元配置分散分析（ANOVA）を実行
    /// 
    /// # 引数
    /// * `value_col` - 測定値を含む列名
    /// * `group_col` - グループ分けの基準となる列名
    /// * `alpha` - 有意水準 (デフォルト: 0.05)
    /// 
    /// # 戻り値
    /// 分散分析の結果
    pub fn anova(&self, value_col: &str, group_col: &str, alpha: Option<f64>) -> Result<AnovaResult> {
        let alpha = alpha.unwrap_or(0.05);
        
        // 値の列を取得
        let value_column = self.column(value_col)?;
        
        // グループの列を取得
        let group_column = self.column(group_col)?;
        let group_col_string = group_column.as_string().ok_or_else(|| 
            crate::error::Error::Type(format!("列 '{}' は文字列型である必要があります", group_col)))?;
        
        // 値を浮動小数点に変換
        let values: Vec<(f64, String)> = match value_column {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| {
                        let val = float_col.get(i).ok().flatten()?;
                        let group = group_col_string.get(i).ok().flatten()?;
                        Some((val, group.to_string()))
                    })
                    .collect()
            },
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| {
                        let val = int_col.get(i).ok().flatten()? as f64;
                        let group = group_col_string.get(i).ok().flatten()?;
                        Some((val, group.to_string()))
                    })
                    .collect()
            },
            _ => return Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", value_col))),
        };
        
        // グループごとにデータを整理
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        for (val, group) in values {
            groups.entry(group).or_insert_with(Vec::new).push(val);
        }
        
        // グループが少なくとも2つあることを確認
        if groups.len() < 2 {
            return Err(crate::error::Error::InsufficientData(
                "分散分析には少なくとも2つのグループが必要です".to_string()
            ));
        }
        
        // &strのグループマップに変換
        let str_groups: HashMap<&str, Vec<f64>> = groups.iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        
        // 分散分析を実行
        stats::anova(&str_groups, alpha)
    }
    
    /// Mann-Whitney U検定（ノンパラメトリック検定）を実行
    /// 
    /// # 引数
    /// * `col1` - 1つ目の列名
    /// * `col2` - 2つ目の列名
    /// * `alpha` - 有意水準 (デフォルト: 0.05)
    /// 
    /// # 戻り値
    /// Mann-Whitney U検定の結果
    pub fn mann_whitney_u(&self, col1: &str, col2: &str, alpha: Option<f64>) -> Result<MannWhitneyResult> {
        let alpha = alpha.unwrap_or(0.05);
        
        // 列データを取得
        let column1 = self.column(col1)?;
        let column2 = self.column(col2)?;
        
        // 浮動小数点ベクトルに変換
        let values1: Vec<f64> = match column1 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            },
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            },
            _ => return Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", col1))),
        };
        
        let values2: Vec<f64> = match column2 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            },
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            },
            _ => return Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", col2))),
        };
        
        // Mann-Whitney U検定を実行
        stats::mann_whitney_u(&values1, &values2, alpha)
    }
    
    /// カイ二乗検定を実行
    /// 
    /// # 引数
    /// * `row_col` - 行を決定する列名
    /// * `col_col` - 列を決定する列名
    /// * `count_col` - カウント/度数の列名
    /// * `alpha` - 有意水準 (デフォルト: 0.05)
    /// 
    /// # 戻り値
    /// カイ二乗検定の結果
    pub fn chi_square_test(&self, row_col: &str, col_col: &str, count_col: &str, alpha: Option<f64>) -> Result<ChiSquareResult> {
        let alpha = alpha.unwrap_or(0.05);
        
        // 列データを取得
        let row_column = self.column(row_col)?;
        let col_column = self.column(col_col)?;
        let count_column = self.column(count_col)?;
        
        // 文字列列の取得
        let row_strings = row_column.as_string().ok_or_else(|| 
            crate::error::Error::Type(format!("列 '{}' は文字列型である必要があります", row_col)))?;
        
        let col_strings = col_column.as_string().ok_or_else(|| 
            crate::error::Error::Type(format!("列 '{}' は文字列型である必要があります", col_col)))?;
        
        // カウント値の取得
        let count_values: Vec<f64> = match count_column {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            },
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            },
            _ => return Err(crate::error::Error::Type(format!("列 '{}' は数値型ではありません", count_col))),
        };
        
        // クロス集計表の生成
        // 一意な行と列の値を抽出
        let mut unique_rows = vec![];
        let mut unique_cols = vec![];
        
        for i in 0..self.row_count() {
            if let Ok(Some(row_val)) = row_strings.get(i) {
                if !unique_rows.contains(&row_val) {
                    unique_rows.push(row_val.clone());
                }
            }
            
            if let Ok(Some(col_val)) = col_strings.get(i) {
                if !unique_cols.contains(&col_val) {
                    unique_cols.push(col_val.clone());
                }
            }
        }
        
        // 観測データ行列の構築
        let mut observed = vec![vec![0.0; unique_cols.len()]; unique_rows.len()];
        
        for i in 0..self.row_count() {
            if let (Ok(Some(row_val)), Ok(Some(col_val)), count) = (
                row_strings.get(i),
                col_strings.get(i),
                count_values.get(i)
            ) {
                if let (Some(row_idx), Some(col_idx)) = (
                    unique_rows.iter().position(|r| r == &row_val),
                    unique_cols.iter().position(|c| c == &col_val)
                ) {
                    // カウント値がある場合はそれを、なければ1.0を追加
                    if let Some(cnt) = count {
                        observed[row_idx][col_idx] += *cnt;
                    } else {
                        observed[row_idx][col_idx] += 1.0;
                    }
                }
            }
        }
        
        // カイ二乗検定を実行
        stats::chi_square_test(&observed, alpha)
    }
    
    /// 線形回帰分析を実行
    /// 
    /// # 引数
    /// * `y_col` - 目的変数（被説明変数）の列名
    /// * `x_cols` - 説明変数の列名リスト
    /// 
    /// # 戻り値
    /// 線形回帰分析の結果
    pub fn linear_regression(&self, y_col: &str, x_cols: &[&str]) -> Result<LinearRegressionResult> {
        // DataFrame形式に変換
        let mut df = crate::dataframe::DataFrame::new();
        
        // 目的変数を追加
        let y_column = self.column(y_col)?;
        if let Some(float_col) = y_column.as_float64() {
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| float_col.get(i).ok().flatten())
                .collect();
            
            let series = crate::series::Series::new(values, Some(y_col.to_string()))?;
            df.add_column(y_col.to_string(), series)?;
        } else if let Some(int_col) = y_column.as_int64() {
            // 整数列は浮動小数点に変換して追加
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                .collect();
            
            let series = crate::series::Series::new(values, Some(y_col.to_string()))?;
            df.add_column(y_col.to_string(), series)?;
        } else {
            return Err(crate::error::Error::Type(format!("列 '{}' は数値型である必要があります", y_col)));
        }
        
        // 説明変数列を追加
        for &x_col in x_cols {
            let x_column = self.column(x_col)?;
            if let Some(float_col) = x_column.as_float64() {
                let values: Vec<f64> = (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect();
                
                let series = crate::series::Series::new(values, Some(x_col.to_string()))?;
                df.add_column(x_col.to_string(), series)?;
            } else if let Some(int_col) = x_column.as_int64() {
                // 整数列は浮動小数点に変換して追加
                let values: Vec<f64> = (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect();
                
                let series = crate::series::Series::new(values, Some(x_col.to_string()))?;
                df.add_column(x_col.to_string(), series)?;
            } else {
                return Err(crate::error::Error::Type(format!("列 '{}' は数値型である必要があります", x_col)));
            }
        }
        
        // 線形回帰モデルを構築
        stats::linear_regression(&df, y_col, x_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{Float64Column, StringColumn, Column};
    use crate::optimized::split_dataframe::OptimizedDataFrame;
    
    #[test]
    fn test_describe() {
        let mut df = OptimizedDataFrame::new();
        
        // テストデータの作成
        let values = Float64Column::with_name(vec![1.0, 2.0, 3.0, 4.0, 5.0], "values");
        df.add_column("values", Column::Float64(values)).unwrap();
        
        // describe関数のテスト
        let desc = df.describe("values").unwrap();
        
        // 結果の検証
        assert_eq!(desc.stats.get("count").unwrap().clone() as usize, 5);
        assert!((desc.stats.get("mean").unwrap() - 3.0).abs() < 1e-10);
        assert!((desc.stats.get("min").unwrap() - 1.0).abs() < 1e-10);
        assert!((desc.stats.get("max").unwrap() - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_ttest() {
        let mut df = OptimizedDataFrame::new();
        
        // テストデータの作成
        let values1 = Float64Column::with_name(vec![1.0, 2.0, 3.0, 4.0, 5.0], "sample1");
        let values2 = Float64Column::with_name(vec![2.0, 3.0, 4.0, 5.0, 6.0], "sample2");
        
        df.add_column("sample1", Column::Float64(values1)).unwrap();
        df.add_column("sample2", Column::Float64(values2)).unwrap();
        
        // t検定の実行
        let result = df.ttest("sample1", "sample2", Some(0.05), Some(true)).unwrap();
        
        // 結果の検証
        assert!(result.statistic < 0.0);  // サンプル2の方が大きい値を持つため
        assert_eq!(result.df, 8);  // 自由度は合計サンプル数 - 2
    }
    
    #[test]
    fn test_anova() {
        let mut df = OptimizedDataFrame::new();
        
        // テストデータの作成
        let values = Float64Column::with_name(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
            "values"
        );
        
        let groups = StringColumn::with_name(
            vec![
                "A".to_string(), "A".to_string(), "A".to_string(), "A".to_string(), "A".to_string(),
                "B".to_string(), "B".to_string(), "B".to_string(), "B".to_string(), "B".to_string(),
                "C".to_string(), "C".to_string(), "C".to_string(), "C".to_string(), "C".to_string(),
            ],
            "group"
        );
        
        df.add_column("values", Column::Float64(values)).unwrap();
        df.add_column("group", Column::String(groups)).unwrap();
        
        // ANOVA実行
        let result = df.anova("values", "group", Some(0.05)).unwrap();
        
        // 結果の検証
        assert!(result.f_statistic > 0.0);
        assert_eq!(result.df_between, 2);  // グループ数 - 1
        assert_eq!(result.df_within, 12);  // 全サンプル数 - グループ数
        assert_eq!(result.df_total, 14);   // 全サンプル数 - 1
    }
}