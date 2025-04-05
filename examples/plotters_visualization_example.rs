//! Plottersによる可視化機能の例
//!
//! このサンプルは、Plottersを使用した高品質な可視化機能の基本的な使い方を示します。

use pandrs::{DataFrame, Series};
use pandrs::vis::plotters_ext::{PlotSettings, PlotKind, OutputType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Series単体のプロット例
    println!("サンプル1: Series単体のプロット作成");
    let values = vec![15.0, 23.5, 18.2, 29.8, 32.1, 28.5, 19.2, 22.3, 25.6, 21.9];
    
    // 線グラフの作成
    let series = Series::new(values.clone(), Some("温度変化".to_string()))?;
    let line_settings = PlotSettings {
        title: "時間ごとの温度変化".to_string(),
        x_label: "時間".to_string(),
        y_label: "温度 (°C)".to_string(),
        plot_kind: PlotKind::Line,
        ..PlotSettings::default()
    };
    series.plotters_plot("examples/temp_line.png", line_settings)?;
    println!("  ✓ 線グラフを生成しました: examples/temp_line.png");
    
    // 棒グラフの作成
    let bar_settings = PlotSettings {
        title: "時間ごとの温度変化".to_string(),
        x_label: "時間".to_string(),
        y_label: "温度 (°C)".to_string(),
        plot_kind: PlotKind::Bar,
        ..PlotSettings::default()
    };
    series.plotters_plot("examples/temp_bar.png", bar_settings)?;
    println!("  ✓ 棒グラフを生成しました: examples/temp_bar.png");
    
    // SVG形式の散布図の作成
    let scatter_settings = PlotSettings {
        title: "時間ごとの温度変化".to_string(),
        x_label: "時間".to_string(),
        y_label: "温度 (°C)".to_string(),
        plot_kind: PlotKind::Scatter,
        output_type: OutputType::SVG,
        ..PlotSettings::default()
    };
    series.plotters_plot("examples/temp_scatter.svg", scatter_settings)?;
    println!("  ✓ 散布図（SVG）を生成しました: examples/temp_scatter.svg");
    
    // ヒストグラムの作成
    let hist_settings = PlotSettings {
        title: "温度分布のヒストグラム".to_string(),
        x_label: "温度帯 (°C)".to_string(),
        y_label: "頻度".to_string(),
        ..PlotSettings::default()
    };
    series.plotters_histogram("examples/temp_histogram.png", 5, hist_settings)?;
    println!("  ✓ ヒストグラムを生成しました: examples/temp_histogram.png");
    
    // 2. DataFrameを使った可視化例
    println!("\nサンプル2: DataFrameを使った可視化");
    let mut df = DataFrame::new();
    
    // データの準備
    let days = Series::new(vec![1, 2, 3, 4, 5, 6, 7], Some("日".to_string()))?;
    let temp = Series::new(vec![22.5, 25.1, 23.8, 27.2, 26.5, 24.9, 29.1], Some("気温".to_string()))?;
    let humidity = Series::new(vec![67.0, 72.3, 69.5, 58.2, 62.1, 71.5, 55.8], Some("湿度".to_string()))?;
    let pressure = Series::new(vec![1013.2, 1010.5, 1009.8, 1014.5, 1018.2, 1015.7, 1011.3], Some("気圧".to_string()))?;
    
    // DataFrameに列を追加
    df.add_column("日".to_string(), days)?;
    df.add_column("気温".to_string(), temp)?;
    df.add_column("湿度".to_string(), humidity)?;
    df.add_column("気圧".to_string(), pressure)?;
    
    // XY散布図の作成（気温と湿度の関係）
    let xy_settings = PlotSettings {
        title: "気温と湿度の関係".to_string(),
        plot_kind: PlotKind::Scatter,
        ..PlotSettings::default()
    };
    df.plotters_xy("気温", "湿度", "examples/temp_humidity.png", xy_settings)?;
    println!("  ✓ 散布図（気温と湿度）を生成しました: examples/temp_humidity.png");
    
    // 複数系列の比較
    let multi_settings = PlotSettings {
        title: "気象データの推移".to_string(),
        x_label: "日数".to_string(),
        plot_kind: PlotKind::Line,
        ..PlotSettings::default()
    };
    df.plotters_multi(&["気温", "湿度", "気圧"], "examples/weather_multi.png", multi_settings)?;
    println!("  ✓ 複数系列のプロットを生成しました: examples/weather_multi.png");
    
    println!("\n全てのサンプルが正常に実行されました。生成されたファイルを確認してください。");
    Ok(())
}