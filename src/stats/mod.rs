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
use std::collections::HashMap;

/// データの基本統計量を計算
///
/// # 説明
/// この関数は、Series、DataFrame、またはその他の数値データに対する
/// 基本的な記述統計量（平均、標準偏差、最小値、最大値など）を計算します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let stats = stats::describe(&data).unwrap();
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
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
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
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
/// use pandrs::series::Series;
///
/// // DataFrameを作成
/// let mut df = DataFrame::new();
/// // データを追加
/// df.add_column("x1".to_string(), Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x1".to_string())).unwrap()).unwrap();
/// df.add_column("x2".to_string(), Series::new(vec![2.0, 3.0, 4.0, 5.0, 6.0], Some("x2".to_string())).unwrap()).unwrap();
/// df.add_column("y".to_string(), Series::new(vec![3.0, 5.0, 7.0, 9.0, 11.0], Some("y".to_string())).unwrap()).unwrap();
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
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// // 1000サンプルのブートストラップ
/// let bootstrap_samples = stats::bootstrap(&data, 1000).unwrap();
/// ```
pub fn bootstrap<T: AsRef<[f64]>>(
    data: T,
    n_samples: usize,
) -> Result<Vec<Vec<f64>>> {
    sampling::bootstrap_impl(data.as_ref(), n_samples)
}

/// 一元配置分散分析（ANOVA）の結果
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F統計量
    pub f_statistic: f64,
    /// p値
    pub p_value: f64,
    /// グループ間平方和
    pub ss_between: f64,
    /// グループ内平方和
    pub ss_within: f64,
    /// 総平方和
    pub ss_total: f64,
    /// グループ間自由度
    pub df_between: usize,
    /// グループ内自由度
    pub df_within: usize,
    /// 総自由度
    pub df_total: usize,
    /// グループ間平均平方
    pub ms_between: f64,
    /// グループ内平均平方
    pub ms_within: f64,
    /// 有意水準で有意か
    pub significant: bool,
}

/// 一元配置分散分析（ANOVA）を実行
///
/// # 説明
/// 3つ以上のグループの平均値に有意差があるかを検定します。
/// グループ間の分散とグループ内の分散の比率（F値）を使用します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
/// use std::collections::HashMap;
///
/// let mut groups = HashMap::new();
/// groups.insert("グループA", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// groups.insert("グループB", vec![2.0, 3.0, 4.0, 5.0, 6.0]);
/// groups.insert("グループC", vec![3.0, 4.0, 5.0, 6.0, 7.0]);
///
/// // 有意水準0.05でANOVA検定
/// let result = stats::anova(&groups, 0.05).unwrap();
/// println!("F統計量: {}", result.f_statistic);
/// println!("p値: {}", result.p_value);
/// println!("有意差あり: {}", result.significant);
/// ```
pub fn anova<T: AsRef<[f64]>>(
    groups: &HashMap<&str, T>,
    alpha: f64,
) -> Result<AnovaResult> {
    if groups.len() < 2 {
        return Err(Error::InsufficientData("分散分析には少なくとも2つのグループが必要です".into()));
    }
    
    // 実装をinferenceモジュールに委譲
    let groups_converted: HashMap<&str, &[f64]> = groups.iter()
        .map(|(k, v)| (*k, v.as_ref()))
        .collect();
    
    inference::anova_impl(&groups_converted, alpha)
}

/// Mann-Whitney U検定（ノンパラメトリック検定）の結果
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// U統計量
    pub u_statistic: f64,
    /// p値
    pub p_value: f64,
    /// 有意水準で有意か
    pub significant: bool,
}

/// Mann-Whitney U検定を実行（ノンパラメトリック検定）
///
/// # 説明
/// データが正規分布に従わない場合にt検定の代わりに使用できる
/// ノンパラメトリック検定です。2つの独立したサンプルの分布を比較します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
/// // 有意水準0.05でMann-Whitney U検定
/// let result = stats::mann_whitney_u(&sample1, &sample2, 0.05).unwrap();
/// println!("U統計量: {}", result.u_statistic);
/// println!("p値: {}", result.p_value);
/// println!("有意差あり: {}", result.significant);
/// ```
pub fn mann_whitney_u<T: AsRef<[f64]>, U: AsRef<[f64]>>(
    sample1: T,
    sample2: U,
    alpha: f64,
) -> Result<MannWhitneyResult> {
    inference::mann_whitney_u_impl(sample1.as_ref(), sample2.as_ref(), alpha)
}

/// カイ二乗検定の結果
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// カイ二乗統計量
    pub chi2_statistic: f64,
    /// p値
    pub p_value: f64,
    /// 自由度
    pub df: usize,
    /// 有意水準で有意か
    pub significant: bool,
    /// 期待度数
    pub expected_freq: Vec<Vec<f64>>,
}

/// カイ二乗検定を実行
///
/// # 説明
/// カテゴリ変数間の関連性を検定します。
/// 観測値と期待値の差に基づいて、変数が独立しているかを評価します。
///
/// # 例
/// ```rust
/// use pandrs::stats;
///
/// // 2x2の分割表（観測値）
/// let observed = vec![
///     vec![20.0, 30.0],
///     vec![25.0, 25.0]
/// ];
/// // 有意水準0.05でカイ二乗検定
/// let result = stats::chi_square_test(&observed, 0.05).unwrap();
/// println!("カイ二乗統計量: {}", result.chi2_statistic);
/// println!("p値: {}", result.p_value);
/// println!("有意差あり: {}", result.significant);
/// ```
pub fn chi_square_test(
    observed: &[Vec<f64>],
    alpha: f64,
) -> Result<ChiSquareResult> {
    inference::chi_square_test_impl(observed, alpha)
}