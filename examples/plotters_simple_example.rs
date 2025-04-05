// 直接plottersクレートを使用するシンプルな可視化サンプル
use plotters::prelude::*;
use rand::{rng, Rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ランダムデータの生成
    let mut rng = rng();
    
    // 折れ線グラフのデータ
    let x: Vec<f64> = (0..100).map(|x| x as f64 / 10.0).collect();
    let y1: Vec<f64> = x.iter().map(|&x| x.sin()).collect();
    let y2: Vec<f64> = x.iter().map(|&x| x.cos()).collect();
    let y3: Vec<f64> = x.iter().map(|&x| x.sin() * 0.5 + x.cos() * 0.5).collect();
    
    // 1. 折れ線グラフ作成
    {
        let root = BitMapBackend::new("line_chart.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Plotters折れ線グラフサンプル", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..10.0, -1.2..1.2)?;
            
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("X軸")
            .y_desc("Y軸")
            .draw()?;
            
        // 複数の系列を描画
        chart.draw_series(LineSeries::new(
            x.iter().zip(y1.iter()).map(|(&x, &y)| (x, y)),
            &RED.mix(0.8),
        ))?
        .label("sin(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.8)));
        
        chart.draw_series(LineSeries::new(
            x.iter().zip(y2.iter()).map(|(&x, &y)| (x, y)), 
            &BLUE.mix(0.8),
        ))?
        .label("cos(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.8)));
        
        chart.draw_series(LineSeries::new(
            x.iter().zip(y3.iter()).map(|(&x, &y)| (x, y)),
            &GREEN.mix(0.8),
        ))?
        .label("sin(x)*0.5 + cos(x)*0.5")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN.mix(0.8)));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
            
        println!("-> line_chart.png に保存しました");
    }
    
    // 2. 散布図作成
    {
        let root = BitMapBackend::new("scatter_chart.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let scatter_data: Vec<(f64, f64)> = (0..100)
            .map(|_| (rng.random_range(0.0..10.0), rng.random_range(-1.0..1.0)))
            .collect();
            
        let mut chart = ChartBuilder::on(&root)
            .caption("Plotters散布図サンプル", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..10.0, -1.2..1.2)?;
            
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("X値")
            .y_desc("Y値")
            .draw()?;
            
        chart.draw_series(
            scatter_data.iter().map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled()))
        )?
        .label("ランダムデータ")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
            
        println!("-> scatter_chart.png に保存しました");
    }
    
    // 3. ヒストグラム作成
    {
        let root = BitMapBackend::new("histogram.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        // 正規分布に似たランダムデータ生成
        let normal_data: Vec<f64> = (0..1000)
            .map(|_| {
                // 中心極限定理で近似正規分布生成
                (0..12).map(|_| rng.random_range(0.0..1.0)).sum::<f64>() - 6.0
            })
            .collect();
            
        // データ範囲計算
        let min_val = normal_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = normal_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // ヒストグラム用のビンを作成
        let bin_count = 20;
        let bin_width = (max_val - min_val) / bin_count as f64;
        let mut bins = vec![0; bin_count];
        
        for &val in &normal_data {
            let bin_idx = ((val - min_val) / bin_width).floor() as usize;
            // 境界処理
            let idx = bin_idx.min(bin_count - 1);
            bins[idx] += 1;
        }
        
        // 最大頻度を計算
        let max_freq = *bins.iter().max().unwrap_or(&0) as f64;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Plottersヒストグラムサンプル", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(min_val..max_val, 0.0..max_freq * 1.1)?;
            
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("値")
            .y_desc("頻度")
            .draw()?;
            
        // ヒストグラムを描画
        chart.draw_series(
            bins.iter().enumerate().map(|(i, &count)| {
                let left = min_val + i as f64 * bin_width;
                let right = left + bin_width;
                
                Rectangle::new(
                    [(left, 0.0), (right, count as f64)],
                    BLUE.mix(0.5).filled(),
                )
            })
        )?;
        
        println!("-> histogram.png に保存しました");
    }
    
    // 4. 棒グラフ作成
    {
        let root = BitMapBackend::new("bar_chart.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        // カテゴリとデータ
        let values = [25, 37, 15, 42, 30];
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Plotters棒グラフサンプル", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..5.0, 0.0..50.0)?;
            
        chart.configure_mesh()
            .x_labels(5)
            .x_label_formatter(&|x| {
                let idx = *x as usize;
                let labels = ["A", "B", "C", "D", "E"];
                if idx < 5 {
                    labels[idx].to_string()
                } else {
                    "".to_string()
                }
            })
            .y_labels(10)
            .y_desc("値")
            .draw()?;
            
        // 棒グラフ描画
        chart.draw_series(
            values.iter().enumerate().map(|(i, &v)| {
                let bar_width = 0.7;
                let x = i as f64 + 0.5;
                Rectangle::new(
                    [(x - bar_width/2.0, 0.0), (x + bar_width/2.0, v as f64)],
                    RGBColor(46, 204, 113).filled(),
                )
            })
        )?;
        
        println!("-> bar_chart.png に保存しました");
    }
    
    // 5. 面グラフ作成 (SVG形式)
    {
        let root = SVGBackend::new("area_chart.svg", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        // データ生成
        let x: Vec<f64> = (0..100).map(|x| x as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&x| x.sin() * 0.5 + 0.5).collect();
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Plotters面グラフサンプル", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..10.0, 0.0..1.2)?;
            
        chart.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("X軸")
            .y_desc("Y軸")
            .draw()?;
            
        // 面グラフ描画
        chart.draw_series(
            AreaSeries::new(
                x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
                0.0,
                &RGBColor(46, 204, 113).mix(0.2),
            )
        )?
        .label("0.5*sin(x) + 0.5")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(46, 204, 113)));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
            
        println!("-> area_chart.svg に保存しました");
    }
    
    println!("すべてのグラフの生成が完了しました。");
    Ok(())
}