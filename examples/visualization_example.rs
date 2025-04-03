use pandrs::{DataFrame, OutputFormat, PlotConfig, PlotType, Series};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== 可視化機能の例 ===\n");

    // サンプルデータの作成
    let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string()))?;
    let y = Series::new(vec![2.0, 3.5, 4.2, 4.8, 7.0], Some("y".to_string()))?;
    let z = Series::new(vec![1.5, 2.2, 3.1, 5.3, 8.5], Some("z".to_string()))?;

    // DataFrameに追加する前にクローンを作成
    let y_for_plot = y.clone();

    // Seriesのプロット - ターミナル出力
    let config = PlotConfig {
        title: "サンプルシリーズのプロット".to_string(),
        x_label: "インデックス".to_string(),
        y_label: "値".to_string(),
        width: 80,
        height: 25,
        plot_type: PlotType::Line,
        format: OutputFormat::Terminal,
    };

    println!("y シリーズをプロット:");
    y_for_plot.plot("", config.clone())?;

    // 散布図のプロット
    let scatter_config = PlotConfig {
        title: "X vs Y 散布図".to_string(),
        x_label: "X値".to_string(),
        y_label: "Y値".to_string(),
        plot_type: PlotType::Scatter,
        ..config.clone()
    };

    // データフレームの作成
    let mut df = DataFrame::new();
    df.add_column("x".to_string(), x)?;
    df.add_column("y".to_string(), y)?;
    df.add_column("z".to_string(), z)?;

    println!("\nXY 散布図をプロット:");
    df.plot_xy("x", "y", "", scatter_config)?;

    // 単一系列の折れ線グラフ（テキストプロットでは複数系列の表示は制限がある）
    let line_config = PlotConfig {
        title: "単一系列のプロット".to_string(),
        x_label: "X".to_string(),
        y_label: "値".to_string(),
        plot_type: PlotType::Line,
        ..config.clone()
    };

    println!("\n折れ線グラフをプロット:");
    df.plot_lines(&["z"], "", line_config)?;

    // テキストファイルとして保存
    let file_config = PlotConfig {
        title: "ファイル出力プロット".to_string(),
        x_label: "インデックス".to_string(),
        y_label: "値".to_string(),
        plot_type: PlotType::Line,
        format: OutputFormat::TextFile,
        ..config.clone()
    };

    println!("\nプロットをファイルに保存中...");
    let y_from_df = df.get_column_numeric_values("y")?;
    // f64をf32に変換
    let y_f32: Vec<f32> = y_from_df.iter().map(|&val| val as f32).collect();
    let y_series = Series::new(y_f32, Some("y".to_string()))?;
    y_series.plot("examples/plot.txt", file_config)?;
    println!("プロットを保存しました: examples/plot.txt");

    println!("\n=== 可視化機能の例完了 ===");
    Ok(())
}
