//! Plottersを使用した高度な可視化機能の実装
//!
//! このモジュールはDataFrameやSeriesのためのPlottersベースの可視化機能を提供します。
//! テキストベースの可視化（textplots）に加えて、より高品質なグラフや可視化を生成できます。

use std::path::Path;
use plotters::prelude::*;
use crate::error::{PandRSError, Result};
use crate::DataFrame;
use crate::Series;
use crate::temporal::TimeSeries;

/// プロットの種類（拡張バージョン）
#[derive(Debug, Clone, Copy)]
pub enum PlotKind {
    /// 折れ線グラフ
    Line,
    /// 散布図
    Scatter,
    /// 棒グラフ
    Bar,
    /// ヒストグラム
    Histogram,
    /// 箱ひげ図
    BoxPlot,
    /// 面グラフ
    Area,
}

/// プロットの出力形式（拡張バージョン）
#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    /// PNG画像
    PNG,
    /// SVG形式
    SVG,
}

/// 拡張プロットの設定
#[derive(Debug, Clone)]
pub struct PlotSettings {
    /// タイトル
    pub title: String,
    /// X軸のラベル
    pub x_label: String,
    /// Y軸のラベル
    pub y_label: String,
    /// グラフの幅（ピクセル）
    pub width: u32,
    /// グラフの高さ（ピクセル）
    pub height: u32,
    /// プロットの種類
    pub plot_kind: PlotKind,
    /// 出力形式
    pub output_type: OutputType,
    /// 凡例の表示
    pub show_legend: bool,
    /// グリッドの表示
    pub show_grid: bool,
    /// 色のパレット
    pub color_palette: Vec<(u8, u8, u8)>,
}

impl Default for PlotSettings {
    fn default() -> Self {
        PlotSettings {
            title: "Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 800,
            height: 600,
            plot_kind: PlotKind::Line,
            output_type: OutputType::PNG,
            show_legend: true,
            show_grid: true,
            color_palette: vec![
                (0, 123, 255),    // 青
                (255, 99, 71),    // 赤
                (46, 204, 113),   // 緑
                (255, 193, 7),    // 黄
                (142, 68, 173),   // 紫
                (52, 152, 219),   // 水色
                (243, 156, 18),   // オレンジ
                (211, 84, 0),     // 茶色
            ],
        }
    }
}

/// Series型の拡張機能
impl<T> Series<T>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    /// Seriesを高品質なグラフとして出力
    ///
    /// # 引数
    ///
    /// * `path` - 出力ファイルのパス
    /// * `settings` - プロット設定
    ///
    /// # 例
    ///
    /// ```no_run
    /// use pandrs::{Series, vis::plotters_ext::{PlotSettings, PlotKind, OutputType}};
    ///
    /// let series = Series::new(vec![1, 2, 3, 4, 5], Some("data".to_string())).unwrap();
    /// let settings = PlotSettings {
    ///     title: "My Plot".to_string(),
    ///     plot_kind: PlotKind::Line,
    ///     ..PlotSettings::default()
    /// };
    /// series.plotters_plot("my_plot.png", settings).unwrap();
    /// ```
    pub fn plotters_plot<P: AsRef<Path>>(&self, path: P, mut settings: PlotSettings) -> Result<()> {
        let values: Vec<f64> = self.values().iter().map(|v| (*v).into()).collect();
        let indices: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        // シリーズ名をタイトルに反映（設定されていない場合）
        if settings.title == "Plot" {
            if let Some(name) = self.name() {
                settings.title = format!("{} のプロット", name);
            }
        }

        // シリーズ名を取得（凡例用）
        let series_name = self.name().map_or_else(|| "Series".to_string(), |s| s.clone());

        match settings.output_type {
            OutputType::PNG => plot_series_xy_png(&indices, &values, path, &settings, &series_name),
            OutputType::SVG => plot_series_xy_svg(&indices, &values, path, &settings, &series_name),
        }
    }

    /// Seriesからヒストグラムを作成
    ///
    /// # 引数
    ///
    /// * `path` - 出力ファイルのパス
    /// * `bins` - ビン（区間）の数
    /// * `settings` - プロット設定
    ///
    /// # 例
    ///
    /// ```no_run
    /// use pandrs::{Series, vis::plotters_ext::{PlotSettings, OutputType}};
    ///
    /// let series = Series::new(vec![1, 2, 3, 4, 5, 1, 2, 3, 2, 1], Some("data".to_string())).unwrap();
    /// let settings = PlotSettings {
    ///     title: "Histogram".to_string(),
    ///     ..PlotSettings::default()
    /// };
    /// series.plotters_histogram("histogram.png", 5, settings).unwrap();
    /// ```
    pub fn plotters_histogram<P: AsRef<Path>>(
        &self,
        path: P,
        bins: usize,
        mut settings: PlotSettings,
    ) -> Result<()> {
        let values: Vec<f64> = self.values().iter().map(|v| (*v).into()).collect();
        
        if values.is_empty() {
            return Err(PandRSError::Empty("データが空です".to_string()));
        }
        
        if bins == 0 {
            return Err(PandRSError::InvalidInput("ビン数は1以上である必要があります".to_string()));
        }

        // シリーズ名をタイトルに反映（設定されていない場合）
        if settings.title == "Plot" {
            if let Some(name) = self.name() {
                settings.title = format!("{} のヒストグラム", name);
            } else {
                settings.title = "ヒストグラム".to_string();
            }
        }
        
        // シリーズ名を取得（凡例用）
        let series_name = self.name().map_or_else(|| "Histogram".to_string(), |s| s.clone());
        
        match settings.output_type {
            OutputType::PNG => plot_histogram_png(&values, path, bins, &settings, &series_name),
            OutputType::SVG => plot_histogram_svg(&values, path, bins, &settings, &series_name),
        }
    }
}

/// DataFrame型の拡張機能
impl DataFrame {
    /// DataFrameの2つの列をXY座標として高品質なグラフを出力
    ///
    /// # 引数
    ///
    /// * `x_col` - X軸の列名
    /// * `y_col` - Y軸の列名
    /// * `path` - 出力ファイルのパス
    /// * `settings` - プロット設定
    ///
    /// # 例
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, Series, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("x".to_string(), Series::new(vec![1, 2, 3, 4, 5], Some("x".to_string())).unwrap()).unwrap();
    /// df.add_column("y".to_string(), Series::new(vec![5, 3, 1, 4, 2], Some("y".to_string())).unwrap()).unwrap();
    ///
    /// let settings = PlotSettings {
    ///     title: "Scatter Plot".to_string(),
    ///     plot_kind: PlotKind::Scatter,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_xy("x", "y", "scatter.png", settings).unwrap();
    /// ```
    pub fn plotters_xy<P: AsRef<Path>>(
        &self,
        x_col: &str,
        y_col: &str,
        path: P,
        mut settings: PlotSettings,
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

        // f64に変換
        let x_f64: Vec<f64> = x_values.iter().map(|&v| v as f64).collect();
        let y_f64: Vec<f64> = y_values.iter().map(|&v| v as f64).collect();

        // タイトルとラベルが指定されていない場合はデフォルトを設定
        if settings.title == "Plot" {
            settings.title = format!("{} vs {}", y_col, x_col);
        }
        if settings.x_label == "X" {
            settings.x_label = x_col.to_string();
        }
        if settings.y_label == "Y" {
            settings.y_label = y_col.to_string();
        }

        // 出力形式に応じてプロット関数を選択
        let series_name = format!("{} vs {}", y_col, x_col);
        match settings.output_type {
            OutputType::PNG => plot_series_xy_png(&x_f64, &y_f64, path, &settings, &series_name),
            OutputType::SVG => plot_series_xy_svg(&x_f64, &y_f64, path, &settings, &series_name),
        }
    }

    /// DataFrameの複数列を比較する折れ線グラフや棒グラフを描画
    ///
    /// # 引数
    ///
    /// * `columns` - プロットする列名のリスト
    /// * `path` - 出力ファイルのパス
    /// * `settings` - プロット設定
    ///
    /// # 例
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, Series, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("A".to_string(), Series::new(vec![1, 2, 3, 4, 5], Some("A".to_string())).unwrap()).unwrap();
    /// df.add_column("B".to_string(), Series::new(vec![5, 3, 1, 4, 2], Some("B".to_string())).unwrap()).unwrap();
    /// df.add_column("C".to_string(), Series::new(vec![2, 4, 6, 3, 1], Some("C".to_string())).unwrap()).unwrap();
    ///
    /// let settings = PlotSettings {
    ///     title: "Multiple Lines".to_string(),
    ///     plot_kind: PlotKind::Line,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_multi(&["A", "B", "C"], "multi_line.png", settings).unwrap();
    /// ```
    pub fn plotters_multi<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        mut settings: PlotSettings,
    ) -> Result<()> {
        if columns.is_empty() {
            return Err(PandRSError::Empty("プロットする列が指定されていません".to_string()));
        }

        // 各列の存在チェック
        for col in columns {
            if !self.contains_column(col) {
                return Err(PandRSError::Column(format!("列 '{}' が存在しません", col)));
            }
        }

        // インデックスを取得
        let indices: Vec<f64> = (0..self.row_count()).map(|i| i as f64).collect();

        // 各列のデータを収集
        let mut series_data = Vec::with_capacity(columns.len());
        for &col in columns {
            let values = self.get_column_numeric_values(col)?;
            let values_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            series_data.push((col, values_f64));
        }

        // タイトルの設定
        if settings.title == "Plot" {
            settings.title = "複数系列の比較".to_string();
        }

        // series_dataをStringベースに変換
        let string_series_data: Vec<(&str, Vec<f64>)> = series_data;

        // 出力形式に応じてプロット関数を選択
        match settings.output_type {
            OutputType::PNG => plot_multi_series_png(&indices, &string_series_data, path, &settings),
            OutputType::SVG => plot_multi_series_svg(&indices, &string_series_data, path, &settings),
        }
    }

    /// DataFrameのある列を基準に他の列のグループごとの集計をボックスプロットで表示
    ///
    /// # 引数
    ///
    /// * `category_col` - カテゴリ列名（x軸）
    /// * `value_col` - 値の列名（y軸）
    /// * `path` - 出力ファイルのパス
    /// * `settings` - プロット設定
    ///
    /// # 例
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, Series, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// // ここにデータを追加...
    ///
    /// let settings = PlotSettings {
    ///     title: "Box Plot by Category".to_string(),
    ///     plot_kind: PlotKind::BoxPlot,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_boxplot("category", "value", "boxplot.png", settings).unwrap();
    /// ```
    pub fn plotters_boxplot<P: AsRef<Path>>(
        &self,
        category_col: &str,
        value_col: &str,
        path: P,
        mut settings: PlotSettings,
    ) -> Result<()> {
        // 列の存在チェック
        if !self.contains_column(category_col) {
            return Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                category_col
            )));
        }
        if !self.contains_column(value_col) {
            return Err(PandRSError::Column(format!(
                "列 '{}' が存在しません",
                value_col
            )));
        }

        // カテゴリとその値のマッピングを作成
        let categories = self.get_column_string_values(category_col)?;
        let values = self.get_column_numeric_values(value_col)?;

        // カテゴリごとに値をグループ化
        let mut category_map: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
        for (cat, val) in categories.iter().zip(values.iter()) {
            let entry = category_map.entry(cat.clone()).or_insert_with(Vec::new);
            entry.push(*val as f64);
        }

        // タイトルとラベルの設定
        if settings.title == "Plot" {
            settings.title = format!("{} by {}", value_col, category_col);
        }
        if settings.x_label == "X" {
            settings.x_label = category_col.to_string();
        }
        if settings.y_label == "Y" {
            settings.y_label = value_col.to_string();
        }

        // TODO: 実際の箱ひげ図の実装
        // 現時点ではこの機能はスケルトン実装のみ
        Err(PandRSError::NotImplemented("箱ひげ図の実装は今後のアップデートで提供予定です".to_string()))
    }
}

/// PNG形式でのシリーズXYプロット実装
fn plot_series_xy_png<P: AsRef<Path>>(
    x: &[f64],
    y: &[f64],
    path: P,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "XとYの長さが一致しません".to_string(),
        ));
    }

    if x.is_empty() {
        return Err(PandRSError::Empty("プロットするデータがありません".to_string()));
    }

    // データの最小値と最大値を計算
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // マージン計算
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // PNG出力用のバックエンドを作成
    let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // グリッド線を追加
    if settings.show_grid {
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    }

    // ポイントの作成
    let points: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

    // 色情報の取得
    let rgb = &settings.color_palette[0];
    let color = RGBColor(rgb.0, rgb.1, rgb.2);

    // プロットの種類によって描画方法を変える
    match settings.plot_kind {
        PlotKind::Line => {
            let series = LineSeries::new(points, color);
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)], 
                        RGBColor(rgb.0, rgb.1, rgb.2)
                    )
                });
        }
        PlotKind::Scatter => {
            let series = points.iter().map(|&(x, y)| {
                Circle::new((x, y), 3, color.filled())
            });
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Circle::new(
                        (x + 10, y), 3, 
                        RGBColor(rgb.0, rgb.1, rgb.2).filled()
                    )
                });
        }
        PlotKind::Bar => {
            let bar_width = (x_max - x_min) / (x.len() as f64) * 0.8;
            let y_baseline = if y_min < 0.0 { 0.0 } else { y_min };
            
            let series = points.iter().map(|&(x, y)| {
                Rectangle::new(
                    [(x - bar_width / 2.0, y_baseline), (x + bar_width / 2.0, y)],
                    color.filled(),
                )
            });
            
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Rectangle::new(
                        [(x, y - 5), (x + 20, y + 5)], 
                        RGBColor(rgb.0, rgb.1, rgb.2).filled()
                    )
                });
        }
        PlotKind::Area => {
            let baseline = y_min.min(0.0);
            let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);
            
            let series = AreaSeries::new(
                points.iter().map(|&(x, y)| (x, y)),
                baseline,
                area_color,
            );
            
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)], 
                        RGBColor(rgb.0, rgb.1, rgb.2)
                    )
                });
        }
        _ => {
            return Err(PandRSError::NotImplemented(
                "指定されたプロット種類はこの関数ではサポートされていません".to_string(),
            ));
        }
    }

    // 凡例を表示
    if settings.show_legend {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// SVG形式でのシリーズXYプロット実装
fn plot_series_xy_svg<P: AsRef<Path>>(
    x: &[f64],
    y: &[f64],
    path: P,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "XとYの長さが一致しません".to_string(),
        ));
    }

    if x.is_empty() {
        return Err(PandRSError::Empty("プロットするデータがありません".to_string()));
    }

    // データの最小値と最大値を計算
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // マージン計算
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // SVG出力用のバックエンドを作成
    let root = SVGBackend::new(path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // グリッド線を追加
    if settings.show_grid {
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    }

    // ポイントの作成
    let points: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

    // 色情報の取得
    let rgb = &settings.color_palette[0];
    let color = RGBColor(rgb.0, rgb.1, rgb.2);

    // プロットの種類によって描画方法を変える
    match settings.plot_kind {
        PlotKind::Line => {
            let series = LineSeries::new(points, color);
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)], 
                        RGBColor(rgb.0, rgb.1, rgb.2)
                    )
                });
        }
        PlotKind::Scatter => {
            let series = points.iter().map(|&(x, y)| {
                Circle::new((x, y), 3, color.filled())
            });
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Circle::new(
                        (x + 10, y), 3, 
                        RGBColor(rgb.0, rgb.1, rgb.2).filled()
                    )
                });
        }
        PlotKind::Bar => {
            let bar_width = (x_max - x_min) / (x.len() as f64) * 0.8;
            let y_baseline = if y_min < 0.0 { 0.0 } else { y_min };
            
            let series = points.iter().map(|&(x, y)| {
                Rectangle::new(
                    [(x - bar_width / 2.0, y_baseline), (x + bar_width / 2.0, y)],
                    color.filled(),
                )
            });
            
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Rectangle::new(
                        [(x, y - 5), (x + 20, y + 5)], 
                        RGBColor(rgb.0, rgb.1, rgb.2).filled()
                    )
                });
        }
        PlotKind::Area => {
            let baseline = y_min.min(0.0);
            let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);
            
            let series = AreaSeries::new(
                points.iter().map(|&(x, y)| (x, y)),
                baseline,
                area_color,
            );
            
            chart.draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)], 
                        RGBColor(rgb.0, rgb.1, rgb.2)
                    )
                });
        }
        _ => {
            return Err(PandRSError::NotImplemented(
                "指定されたプロット種類はこの関数ではサポートされていません".to_string(),
            ));
        }
    }

    // 凡例を表示
    if settings.show_legend {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// PNG形式での複数系列プロット実装
fn plot_multi_series_png<P: AsRef<Path>>(
    x: &[f64],
    series: &[(&str, Vec<f64>)],
    path: P,
    settings: &PlotSettings,
) -> Result<()> {
    if series.is_empty() {
        return Err(PandRSError::Empty("プロットするデータがありません".to_string()));
    }

    // 各系列の長さをチェック
    for (name, data) in series {
        if data.len() != x.len() {
            return Err(PandRSError::Consistency(format!(
                "系列 '{}' の長さがX軸の長さと一致しません",
                name
            )));
        }
    }

    // Xの最小値と最大値
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // すべての系列のYの最小値と最大値
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, data) in series {
        let series_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let series_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        y_min = y_min.min(series_min);
        y_max = y_max.max(series_max);
    }

    // マージン計算
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // PNG出力用のバックエンドを作成
    let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // グリッド線を追加
    if settings.show_grid {
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    }

    // 各系列をプロット
    for (i, (name, data)) in series.iter().enumerate() {
        let palette_idx = i % settings.color_palette.len();
        let rgb = &settings.color_palette[palette_idx];
        let color = RGBColor(rgb.0, rgb.1, rgb.2);
        let points: Vec<(f64, f64)> = x.iter().zip(data.iter()).map(|(&x, &y)| (x, y)).collect();

        match settings.plot_kind {
            PlotKind::Line => {
                let series = LineSeries::new(points, color);
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)], 
                            RGBColor(rgb.0, rgb.1, rgb.2)
                        )
                    });
            }
            PlotKind::Scatter => {
                let series = points.iter().map(|&(x, y)| {
                    Circle::new((x, y), 3, color.filled())
                });
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        Circle::new(
                            (x + 10, y), 3, 
                            RGBColor(rgb.0, rgb.1, rgb.2).filled()
                        )
                    });
            }
            PlotKind::Bar => {
                // 棒グラフは系列ごとに位置をずらす
                let bar_width = (x_max - x_min) / (x.len() as f64) * 0.8 / (series.len() as f64);
                let offset = (i as f64 - (series.len() as f64 - 1.0) / 2.0) * bar_width;
                let y_baseline = if y_min < 0.0 { 0.0 } else { y_min };
                
                let series = points.iter().map(move |&(x, y)| {
                    Rectangle::new(
                        [(x + offset - bar_width / 2.0, y_baseline), (x + offset + bar_width / 2.0, y)],
                        color.filled(),
                    )
                });
                
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        Rectangle::new(
                            [(x, y - 5), (x + 20, y + 5)], 
                            RGBColor(rgb.0, rgb.1, rgb.2).filled()
                        )
                    });
            }
            PlotKind::Area => {
                let baseline = y_min.min(0.0);
                let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);
                
                let series = AreaSeries::new(
                    points.iter().map(|&(x, y)| (x, y)),
                    baseline,
                    area_color,
                );
                
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)], 
                            RGBColor(rgb.0, rgb.1, rgb.2)
                        )
                    });
            }
            _ => {
                return Err(PandRSError::NotImplemented(
                    "指定されたプロット種類はこの関数ではサポートされていません".to_string(),
                ));
            }
        }
    }

    // 凡例を表示
    if settings.show_legend {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// SVG形式での複数系列プロット実装
fn plot_multi_series_svg<P: AsRef<Path>>(
    x: &[f64],
    series: &[(&str, Vec<f64>)],
    path: P,
    settings: &PlotSettings,
) -> Result<()> {
    if series.is_empty() {
        return Err(PandRSError::Empty("プロットするデータがありません".to_string()));
    }

    // 各系列の長さをチェック
    for (name, data) in series {
        if data.len() != x.len() {
            return Err(PandRSError::Consistency(format!(
                "系列 '{}' の長さがX軸の長さと一致しません",
                name
            )));
        }
    }

    // Xの最小値と最大値
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // すべての系列のYの最小値と最大値
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, data) in series {
        let series_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let series_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        y_min = y_min.min(series_min);
        y_max = y_max.max(series_max);
    }

    // マージン計算
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // SVG出力用のバックエンドを作成
    let root = SVGBackend::new(path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // グリッド線を追加
    if settings.show_grid {
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    }

    // 各系列をプロット
    for (i, (name, data)) in series.iter().enumerate() {
        let palette_idx = i % settings.color_palette.len();
        let rgb = &settings.color_palette[palette_idx];
        let color = RGBColor(rgb.0, rgb.1, rgb.2);
        let points: Vec<(f64, f64)> = x.iter().zip(data.iter()).map(|(&x, &y)| (x, y)).collect();

        match settings.plot_kind {
            PlotKind::Line => {
                let series = LineSeries::new(points, color);
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)], 
                            RGBColor(rgb.0, rgb.1, rgb.2)
                        )
                    });
            }
            PlotKind::Scatter => {
                let series = points.iter().map(|&(x, y)| {
                    Circle::new((x, y), 3, color.filled())
                });
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        Circle::new(
                            (x + 10, y), 3, 
                            RGBColor(rgb.0, rgb.1, rgb.2).filled()
                        )
                    });
            }
            PlotKind::Bar => {
                // 棒グラフは系列ごとに位置をずらす
                let bar_width = (x_max - x_min) / (x.len() as f64) * 0.8 / (series.len() as f64);
                let offset = (i as f64 - (series.len() as f64 - 1.0) / 2.0) * bar_width;
                let y_baseline = if y_min < 0.0 { 0.0 } else { y_min };
                
                let series = points.iter().map(move |&(x, y)| {
                    Rectangle::new(
                        [(x + offset - bar_width / 2.0, y_baseline), (x + offset + bar_width / 2.0, y)],
                        color.filled(),
                    )
                });
                
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        Rectangle::new(
                            [(x, y - 5), (x + 20, y + 5)], 
                            RGBColor(rgb.0, rgb.1, rgb.2).filled()
                        )
                    });
            }
            PlotKind::Area => {
                let baseline = y_min.min(0.0);
                let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);
                
                let series = AreaSeries::new(
                    points.iter().map(|&(x, y)| (x, y)),
                    baseline,
                    area_color,
                );
                
                chart.draw_series(series)?
                    .label((*name).to_owned())
                    .legend(move |(x, y)| {
                        PathElement::new(
                            vec![(x, y), (x + 20, y)], 
                            RGBColor(rgb.0, rgb.1, rgb.2)
                        )
                    });
            }
            _ => {
                return Err(PandRSError::NotImplemented(
                    "指定されたプロット種類はこの関数ではサポートされていません".to_string(),
                ));
            }
        }
    }

    // 凡例を表示
    if settings.show_legend {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// PNG形式でのヒストグラム実装
fn plot_histogram_png<P: AsRef<Path>>(
    data: &[f64],
    path: P,
    bins: usize,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if data.is_empty() {
        return Err(PandRSError::Empty("データが空です".to_string()));
    }

    if bins == 0 {
        return Err(PandRSError::InvalidInput("ビン数は1以上である必要があります".to_string()));
    }

    // データの最小値と最大値を計算
    let min_value = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_value = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // ビンの幅を計算
    let bin_width = (max_value - min_value) / (bins as f64);
    
    // ヒストグラムのビンを作成
    let mut histogram = vec![0; bins];
    for &value in data {
        let bin_index = ((value - min_value) / bin_width).floor() as usize;
        // 最大値の場合は最後のビンに入れる
        let index = if bin_index >= bins { bins - 1 } else { bin_index };
        histogram[index] += 1;
    }

    // ヒストグラムの最大値
    let max_count = *histogram.iter().max().unwrap_or(&0) as f64;
    
    // PNG出力用のバックエンドを作成
    let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (min_value - bin_width * 0.1)..(max_value + bin_width * 0.1),
            0.0..(max_count * 1.1),
        )?;

    // グリッド線を追加
    if settings.show_grid {
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .x_desc(&settings.x_label)
            .y_desc("Frequency")
            .draw()?;
    }

    // 色情報の取得
    let rgb = &settings.color_palette[0];
    let color = RGBColor(rgb.0, rgb.1, rgb.2);

    // ヒストグラムを描画
    let series = histogram.iter().enumerate().map(|(i, &count)| {
        let x0 = min_value + (i as f64) * bin_width;
        let x1 = min_value + ((i + 1) as f64) * bin_width;
        Rectangle::new(
            [(x0, 0.0), (x1, count as f64)],
            color.filled(),
        )
    });
    
    chart.draw_series(series)?
        .label(series_name.to_owned())
        .legend(move |(x, y)| {
            Rectangle::new(
                [(x, y - 5), (x + 20, y + 5)], 
                RGBColor(rgb.0, rgb.1, rgb.2).filled()
            )
        });

    // 凡例を表示
    if settings.show_legend {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// SVG形式でのヒストグラム実装
fn plot_histogram_svg<P: AsRef<Path>>(
    data: &[f64],
    path: P,
    bins: usize,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if data.is_empty() {
        return Err(PandRSError::Empty("データが空です".to_string()));
    }

    if bins == 0 {
        return Err(PandRSError::InvalidInput("ビン数は1以上である必要があります".to_string()));
    }

    // データの最小値と最大値を計算
    let min_value = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_value = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // ビンの幅を計算
    let bin_width = (max_value - min_value) / (bins as f64);
    
    // ヒストグラムのビンを作成
    let mut histogram = vec![0; bins];
    for &value in data {
        let bin_index = ((value - min_value) / bin_width).floor() as usize;
        // 最大値の場合は最後のビンに入れる
        let index = if bin_index >= bins { bins - 1 } else { bin_index };
        histogram[index] += 1;
    }

    // ヒストグラムの最大値
    let max_count = *histogram.iter().max().unwrap_or(&0) as f64;
    
    // SVG出力用のバックエンドを作成
    let root = SVGBackend::new(path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (min_value - bin_width * 0.1)..(max_value + bin_width * 0.1),
            0.0..(max_count * 1.1),
        )?;

    // グリッド線を追加
    if settings.show_grid {
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .x_desc(&settings.x_label)
            .y_desc("Frequency")
            .draw()?;
    }

    // 色情報の取得
    let rgb = &settings.color_palette[0];
    let color = RGBColor(rgb.0, rgb.1, rgb.2);

    // ヒストグラムを描画
    let series = histogram.iter().enumerate().map(|(i, &count)| {
        let x0 = min_value + (i as f64) * bin_width;
        let x1 = min_value + ((i + 1) as f64) * bin_width;
        Rectangle::new(
            [(x0, 0.0), (x1, count as f64)],
            color.filled(),
        )
    });
    
    chart.draw_series(series)?
        .label(series_name.to_owned())
        .legend(move |(x, y)| {
            Rectangle::new(
                [(x, y - 5), (x + 20, y + 5)], 
                RGBColor(rgb.0, rgb.1, rgb.2).filled()
            )
        });

    // 凡例を表示
    if settings.show_legend {
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}