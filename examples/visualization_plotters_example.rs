use pandrs::{DataFrame, Series, vis::plotters_ext::{PlotSettings, PlotKind, OutputType}};
use rand::{thread_rng, Rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ランダムデータの生成
    let mut rng = thread_rng();
    let x: Vec<i32> = (0..100).collect();
    let y1: Vec<f64> = (0..100).map(|i| i as f64 + rng.gen_range(-5.0..5.0)).collect();
    let y2: Vec<f64> = (0..100).map(|i| i as f64 * 0.8 + 10.0 + rng.gen_range(-3.0..3.0)).collect();
    let y3: Vec<f64> = (0..100).map(|i| 50.0 + 30.0 * (i as f64 * 0.1).sin()).collect();

    // 単一系列の折れ線グラフ
    let series1 = Series::new(y1.clone(), Some("データ1".to_string()))?;
    
    let line_settings = PlotSettings {
        title: "折れ線グラフ".to_string(),
        x_label: "X軸".to_string(),
        y_label: "Y値".to_string(),
        plot_kind: PlotKind::Line,
        ..PlotSettings::default()
    };
    
    println!("折れ線グラフを作成中...");
    series1.plotters_plot("line_chart.png", line_settings)?;
    println!("-> line_chart.png に保存しました");

    // ヒストグラム
    let hist_data: Vec<f64> = (0..1000).map(|_| rng.gen_range(-50.0..50.0)).collect();
    let hist_series = Series::new(hist_data, Some("分布".to_string()))?;
    
    let hist_settings = PlotSettings {
        title: "ヒストグラム".to_string(),
        x_label: "値".to_string(),
        y_label: "頻度".to_string(),
        plot_kind: PlotKind::Histogram,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };
    
    println!("ヒストグラムを作成中...");
    hist_series.plotters_histogram("histogram.png", 20, hist_settings)?;
    println!("-> histogram.png に保存しました");

    // データフレームを使った複数系列のグラフ
    let mut df = DataFrame::new();
    df.add_column("X".to_string(), Series::new(x, Some("X".to_string()))?)?;
    df.add_column("データ1".to_string(), Series::new(y1, Some("データ1".to_string()))?)?;
    df.add_column("データ2".to_string(), Series::new(y2, Some("データ2".to_string()))?)?;
    df.add_column("データ3".to_string(), Series::new(y3.clone(), Some("データ3".to_string()))?)?;

    // 散布図
    let scatter_settings = PlotSettings {
        title: "散布図".to_string(),
        x_label: "X値".to_string(),
        y_label: "Y値".to_string(),
        plot_kind: PlotKind::Scatter,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };
    
    println!("散布図を作成中...");
    df.plotters_xy("X", "データ1", "scatter_chart.png", scatter_settings)?;
    println!("-> scatter_chart.png に保存しました");

    // 複数系列の折れ線グラフ
    let multi_line_settings = PlotSettings {
        title: "複数系列の折れ線グラフ".to_string(),
        x_label: "X値".to_string(),
        y_label: "Y値".to_string(),
        plot_kind: PlotKind::Line,
        output_type: OutputType::SVG, // SVG形式で保存
        ..PlotSettings::default()
    };
    
    println!("複数系列の折れ線グラフを作成中...");
    df.plotters_multi(&["データ1", "データ2", "データ3"], "multi_line_chart.svg", multi_line_settings)?;
    println!("-> multi_line_chart.svg に保存しました");

    // 棒グラフ
    let bar_values = vec![15, 30, 25, 40, 20];
    let categories = vec!["A", "B", "C", "D", "E"];
    
    let mut bar_df = DataFrame::new();
    bar_df.add_column("カテゴリ".to_string(), Series::new(categories, Some("カテゴリ".to_string()))?)?;
    bar_df.add_column("値".to_string(), Series::new(bar_values.clone(), Some("値".to_string()))?)?;
    
    let bar_settings = PlotSettings {
        title: "棒グラフ".to_string(),
        x_label: "カテゴリ".to_string(),
        y_label: "値".to_string(),
        plot_kind: PlotKind::Bar,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };
    
    println!("棒グラフを作成中...");
    // インデックスを使用して棒グラフを作成
    let bar_series = Series::new(bar_values, Some("値".to_string()))?;
    bar_series.plotters_plot("bar_chart.png", bar_settings)?;
    println!("-> bar_chart.png に保存しました");

    // 面グラフ
    let area_settings = PlotSettings {
        title: "面グラフ".to_string(),
        x_label: "時間".to_string(),
        y_label: "値".to_string(),
        plot_kind: PlotKind::Area,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };
    
    println!("面グラフを作成中...");
    let area_series = Series::new(y3, Some("波形".to_string()))?;
    area_series.plotters_plot("area_chart.png", area_settings)?;
    println!("-> area_chart.png に保存しました");

    println!("すべてのグラフの生成が完了しました。");
    Ok(())
}