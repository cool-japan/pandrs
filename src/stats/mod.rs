// PandRS 統計モジュール
//
// このモジュールは、データ分析のための統計機能を提供します。
// 記述統計、推測統計、仮説検定、回帰分析など、幅広い統計手法が実装されています。

pub mod descriptive;
pub mod inference;
pub mod regression;
pub mod sampling;

use crate::dataframe::DataFrame;
use crate::series::Series;
use crate::error::{Result, Error};

/// データの基本統計量を計算
///
/// # 説明
/// この関数は、Series、DataFrame、またはその他の数値データに対する
/// 基本的な記述統計量（平均、標準偏差、最小値、最大値など）を計算します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
/// use pandrs::series::Series;
///
/// let series = Series::from_vec(vec![1, 2, 3, 4, 5]);
/// let stats = stats::describe(&series).unwrap();
/// println!("平均: {}", stats.mean);
/// println!("標準偏差: {}", stats.std);
/// ```
pub fn describe<T: AsRef<[f64]>>(data: T) -> Result<DescriptiveStats> {
    descriptive::describe_impl(data.as_ref())
}

/// 記述統計量の結果を保持する構造体
#[derive(Debug, Clone)]
pub struct DescriptiveStats {
    /// データの件数
    pub count: usize,
    /// 平均値
    pub mean: f64,
    /// 標準偏差（不偏推定量）
    pub std: f64,
    /// 最小値
    pub min: f64,
    /// 25%分位点
    pub q1: f64,
    /// 中央値（50%分位点）
    pub median: f64,
    /// 75%分位点
    pub q3: f64,
    /// 最大値
    pub max: f64,
}

/// 相関係数を計算
///
/// # 説明
/// 2つの数値配列間のピアソン相関係数を計算します。
/// 相関係数は-1から1の範囲で、1は完全な正の相関、-1は完全な負の相関、
/// 0は相関がないことを示します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let corr = stats::correlation(&x, &y).unwrap();
/// println!("相関係数: {}", corr);
/// ```
pub fn correlation<T: AsRef<[f64]>, U: AsRef<[f64]>>(x: T, y: U) -> Result<f64> {
    descriptive::correlation_impl(x.as_ref(), y.as_ref())
}

/// 共分散を計算
///
/// # 説明
/// 2つの数値配列間の共分散を計算します。
/// 共分散は2つの変数がどの程度一緒に変動するかを示す指標です。
///
/// # 例
/// ```rust
/// use pandrs::stats;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let cov = stats::covariance(&x, &y).unwrap();
/// println!("共分散: {}", cov);
/// ```
pub fn covariance<T: AsRef<[f64]>, U: AsRef<[f64]>>(x: T, y: U) -> Result<f64> {
    descriptive::covariance_impl(x.as_ref(), y.as_ref())
}

/// tテストの結果
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// t統計量
    pub statistic: f64,
    /// p値
    pub pvalue: f64,
    /// 有意水準で有意か
    pub significant: bool,
    /// 自由度
    pub df: usize,
}

/// 2標本のt検定を実行
///
/// # 説明
/// 2つの独立した標本の平均値に有意差があるかを検定します。
/// 等分散を仮定するかどうかで検定方法が異なります。
///
/// # 例
/// ```rust
/// use pandrs::stats;
/// use pandrs::series::Series;
///
/// let sample1 = Series::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let sample2 = Series::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
/// // 等分散を仮定した検定、有意水準0.05
/// let result = stats::ttest(&sample1, &sample2, 0.05, true).unwrap();
/// println!("t統計量: {}", result.statistic);
/// println!("p値: {}", result.pvalue);
/// println!("有意差あり: {}", result.significant);
/// ```
pub fn ttest<T: AsRef<[f64]>, U: AsRef<[f64]>>(
    sample1: T,
    sample2: U,
    alpha: f64,
    equal_var: bool,
) -> Result<TTestResult> {
    inference::ttest_impl(sample1.as_ref(), sample2.as_ref(), alpha, equal_var)
}

/// 線形回帰モデルの結果
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    /// 切片
    pub intercept: f64,
    /// 係数（重回帰の場合は複数）
    pub coefficients: Vec<f64>,
    /// 決定係数（R²）
    pub r_squared: f64,
    /// 調整済み決定係数
    pub adj_r_squared: f64,
    /// 各係数のp値
    pub p_values: Vec<f64>,
    /// モデルの当てはめ値
    pub fitted_values: Vec<f64>,
    /// 残差
    pub residuals: Vec<f64>,
}

/// 線形回帰分析を実行
///
/// # 説明
/// 単回帰または重回帰分析を実行して、最小二乗法による線形モデルを構築します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// // DataFrameを作成
/// let mut df = DataFrame::new();
/// // （データを追加する処理）
///
/// // y列を目的変数、x1とx2列を説明変数として回帰分析
/// let model = stats::linear_regression(&df, "y", &["x1", "x2"]).unwrap();
/// println!("切片: {}", model.intercept);
/// println!("係数: {:?}", model.coefficients);
/// println!("決定係数: {}", model.r_squared);
/// ```
pub fn linear_regression(
    df: &DataFrame,
    y_column: &str,
    x_columns: &[&str],
) -> Result<LinearRegressionResult> {
    regression::linear_regression_impl(df, y_column, x_columns)
}

/// 無作為抽出を行う
///
/// # 説明
/// 指定したサイズのランダムサンプルを取得します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // データを含むDataFrame
/// // 10%のランダムサンプルを取得
/// let sampled_df = stats::sample(&df, 0.1, true).unwrap();
/// ```
pub fn sample(
    df: &DataFrame,
    fraction: f64,
    replace: bool,
) -> Result<DataFrame> {
    sampling::sample_impl(df, fraction, replace)
}

/// ブートストラップサンプルを生成
///
/// # 説明
/// ブートストラップ法によるリサンプリングを行います。
/// 統計量の信頼区間推定などに使用します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
/// use pandrs::series::Series;
///
/// let data = Series::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// // 1000サンプルのブートストラップ
/// let bootstrap_samples = stats::bootstrap(&data, 1000).unwrap();
/// ```
pub fn bootstrap<T: AsRef<[f64]>>(
    data: T,
    n_samples: usize,
) -> Result<Vec<Vec<f64>>> {
    sampling::bootstrap_impl(data.as_ref(), n_samples)
}