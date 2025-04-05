//! データ可視化機能を提供するモジュール
//!
//! このモジュールには、テキストベース（textplots）と高品質な可視化（plotters）の
//! 両方の機能が含まれています。

use std::fs::File;
use std::io::Write;
use std::path::Path;
use textplots::{Chart, Plot, Shape};

use crate::error::{PandRSError, Result};
use crate::temporal::TimeSeries;
use crate::DataFrame;
use crate::Series;

// 高品質な可視化モジュールをエクスポート（一時的に無効化）
// pub mod plotters_ext;

/// プロットの種類
#[derive(Debug, Clone, Copy)]
pub enum PlotType {
    /// 折れ線グラフ
    Line,
    /// 散布図
    Scatter,
    /// ポイントプロット
    Points,
}

/// プロットの出力形式
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// ターミナル出力
    Terminal,
    /// ファイル出力（テキスト形式）
    TextFile,
}

/// プロットの設定
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// タイトル
    pub title: String,
    /// X軸のラベル
    pub x_label: String,
    /// Y軸のラベル
    pub y_label: String,
    /// 幅（文字数）
    pub width: u32,
    /// 高さ（行数）
    pub height: u32,
    /// プロットの種類
    pub plot_type: PlotType,
    /// 出力形式
    pub format: OutputFormat,
}

impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig {
            title: "Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 80,
            height: 25,
            plot_type: PlotType::Line,
            format: OutputFormat::Terminal,
        }
    }
}

/// 可視化機能の拡張: Seriesの可視化
impl<T> Series<T>
where
    T: Clone + Copy + Into<f32> + std::fmt::Debug,
{
    /// Seriesをプロットしてファイルに保存またはターミナルに表示
    pub fn plot<P: AsRef<Path>>(&self, path: P, config: PlotConfig) -> Result<()> {
        let values: Vec<f32> = self.values().iter().map(|v| (*v).into()).collect();
        let indices: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();

        plot_xy(&indices, &values, path, config)
    }
}

/// 可視化機能の拡張: DataFrameの可視化
impl DataFrame {
    /// 2つの列をXY座標としてプロット
    pub fn plot_xy<P: AsRef<Path>>(
        &self,
        x_col: &str,
        y_col: &str,
        path: P,
        config: PlotConfig,
    ) -> Result<()> {
        // 列の存在チェック
        if !self.contains_column(x_col) {
            return Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                x_col
            )));
        }
        if !self.contains_column(y_col) {
            return Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                y_col
            )));
        }

        // 列データを数値に変換
        let x_values = self.get_column_numeric_values(x_col)?;
        let y_values = self.get_column_numeric_values(y_col)?;

        // f32に変換
        let x_f32: Vec<f32> = x_values.iter().map(|&v| v as f32).collect();
        let y_f32: Vec<f32> = y_values.iter().map(|&v| v as f32).collect();

        plot_xy(&x_f32, &y_f32, path, config)
    }

    /// 複数列の折れ線グラフを描画
    pub fn plot_lines<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        config: PlotConfig,
    ) -> Result<()> {
        // 各列の存在チェック
        for col in columns {
            if !self.contains_column(col) {
                return Err(PandRSError::Column(format!("列 '{}' が存在しません", col)));
            }
        }

        // インデックスを取得
        let indices: Vec<f32> = (0..self.row_count()).map(|i| i as f32).collect();

        // 最初の列だけを使用（textplotsでは複数系列の表示に制限がある）
        if let Some(&first_col) = columns.first() {
            let values = self.get_column_numeric_values(first_col)?;
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

            let mut custom_config = config;
            custom_config.title = format!("{} ({})", custom_config.title, first_col);

            return plot_xy(&indices, &values_f32, path, custom_config);
        }

        Err(PandRSError::Empty("プロットする列がありません".to_string()))
    }
}

/// 可視化機能の拡張: 時系列データの可視化
impl<T> TimeSeries<T>
where
    T: crate::temporal::Temporal,
{
    /// 時系列データを折れ線グラフとしてプロット
    pub fn plot<P: AsRef<Path>>(&self, path: P, config: PlotConfig) -> Result<()> {
        // 値データを取得
        let values: Vec<f32> = self
            .values()
            .iter()
            .map(|v| match v {
                crate::na::NA::Value(val) => (*val as i32) as f32, // 数値への変換
                crate::na::NA::NA => 0.0, // NAは0として扱う (実際はより適切な処理が必要)
            })
            .collect();

        // 日付をインデックスとして使用
        let indices: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();

        // プロット設定を更新
        let mut custom_config = config;
        if custom_config.x_label == "X" {
            custom_config.x_label = "Date".to_string();
        }

        plot_xy(&indices, &values, path, custom_config)
    }
}

/// XY座標の基本プロット関数
fn plot_xy<P: AsRef<Path>>(x: &[f32], y: &[f32], path: P, config: PlotConfig) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "XとYの長さが一致しません".to_string(),
        ));
    }

    // データが空の場合は何もしない
    if x.is_empty() {
        return Err(PandRSError::Empty(
            "プロットするデータがありません".to_string(),
        ));
    }

    // ポイントの作成
    let points: Vec<(f32, f32)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

    // チャートの作成
    let mut chart_string = String::new();
    chart_string.push_str(&format!("=== {} ===\n", config.title));
    chart_string.push_str(&format!(
        "X軸: {}, Y軸: {}\n\n",
        config.x_label, config.y_label
    ));

    // プロット描画
    let chart_result = match config.plot_type {
        PlotType::Line => Chart::new(
            config.width as u32,
            config.height as u32,
            x[0],
            x[x.len() - 1],
        )
        .lineplot(&Shape::Lines(&points))
        .to_string(),
        PlotType::Scatter | PlotType::Points => Chart::new(
            config.width as u32,
            config.height as u32,
            x[0],
            x[x.len() - 1],
        )
        .lineplot(&Shape::Points(&points))
        .to_string(),
    };

    chart_string.push_str(&chart_result);

    // 出力
    match config.format {
        OutputFormat::Terminal => {
            println!("{}", chart_string);
            Ok(())
        }
        OutputFormat::TextFile => {
            let mut file = File::create(path).map_err(PandRSError::Io)?;
            file.write_all(chart_string.as_bytes())
                .map_err(PandRSError::Io)?;
            Ok(())
        }
    }
}
