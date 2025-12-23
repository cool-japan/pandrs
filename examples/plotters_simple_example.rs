// Directly using the plotters crate for a simple visualization sample
#[cfg(feature = "visualization")]
use plotters::prelude::*;
#[cfg(feature = "visualization")]
use rand::{rng, Rng};

// Translated Japanese comments and strings into English
#[cfg(feature = "visualization")]
#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate random data
    let mut rng = rng();

    // Data for line chart
    let x: Vec<f64> = (0..100).map(|x| x as f64 / 10.0).collect();
    let y1: Vec<f64> = x.iter().map(|&x| x.sin()).collect();
    let y2: Vec<f64> = x.iter().map(|&x| x.cos()).collect();
    let y3: Vec<f64> = x.iter().map(|&x| x.sin() * 0.5 + x.cos() * 0.5).collect();

    // 1. Create line chart
    {
        let root = BitMapBackend::new("line_chart.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Plotters Line Chart Example",
                ("sans-serif", 30).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..10.0, -1.2..1.2)?;

        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("X Axis")
            .y_desc("Y Axis")
            .draw()?;

        // Draw multiple series
        chart
            .draw_series(LineSeries::new(
                x.iter().zip(y1.iter()).map(|(&x, &y)| (x, y)),
                &RED.mix(0.8),
            ))?
            .label("sin(x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.8)));

        chart
            .draw_series(LineSeries::new(
                x.iter().zip(y2.iter()).map(|(&x, &y)| (x, y)),
                &BLUE.mix(0.8),
            ))?
            .label("cos(x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.8)));

        chart
            .draw_series(LineSeries::new(
                x.iter().zip(y3.iter()).map(|(&x, &y)| (x, y)),
                &GREEN.mix(0.8),
            ))?
            .label("sin(x)*0.5 + cos(x)*0.5")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.mix(0.8)));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;

        println!("-> Saved to line_chart.png");
    }

    // 2. Create scatter plot
    {
        let root = BitMapBackend::new("scatter_chart.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let scatter_data: Vec<(f64, f64)> = (0..100)
            .map(|_| (rng.random_range(0.0..10.0), rng.random_range(-1.0..1.0)))
            .collect();

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Plotters Scatter Plot Example",
                ("sans-serif", 30).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..10.0, -1.2..1.2)?;

        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("X Value")
            .y_desc("Y Value")
            .draw()?;

        chart
            .draw_series(
                scatter_data
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled())),
            )?
            .label("Random Data")
            .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        println!("-> Saved to scatter_chart.png");
    }

    // 3. Create histogram
    {
        let root = BitMapBackend::new("histogram.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Generate random data resembling a normal distribution
        let normal_data: Vec<f64> = (0..1000)
            .map(|_| {
                // Approximate normal distribution using the central limit theorem
                (0..12).map(|_| rng.random_range(0.0..1.0)).sum::<f64>() - 6.0
            })
            .collect();

        // Calculate data range
        let min_val = normal_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = normal_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Create bins for histogram
        let bin_count = 20;
        let bin_width = (max_val - min_val) / bin_count as f64;
        let mut bins = vec![0; bin_count];

        for &val in &normal_data {
            let bin_idx = ((val - min_val) / bin_width).floor() as usize;
            // Handle boundary cases
            let idx = bin_idx.min(bin_count - 1);
            bins[idx] += 1;
        }

        // Calculate maximum frequency
        let max_freq = *bins.iter().max().unwrap_or(&0) as f64;

        let mut chart = ChartBuilder::on(&root)
            .caption("Plotters Histogram Example", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(min_val..max_val, 0.0..max_freq * 1.1)?;

        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("Value")
            .y_desc("Frequency")
            .draw()?;

        // Draw histogram
        chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
            let left = min_val + i as f64 * bin_width;
            let right = left + bin_width;

            Rectangle::new([(left, 0.0), (right, count as f64)], BLUE.mix(0.5).filled())
        }))?;

        println!("-> Saved to histogram.png");
    }

    // 4. Create bar chart
    {
        let root = BitMapBackend::new("bar_chart.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Categories and data
        let values = [25, 37, 15, 42, 30];

        let mut chart = ChartBuilder::on(&root)
            .caption("Plotters Bar Chart Example", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..5.0, 0.0..50.0)?;

        chart
            .configure_mesh()
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
            .y_desc("Value")
            .draw()?;

        // Draw bar chart
        chart.draw_series(values.iter().enumerate().map(|(i, &v)| {
            let bar_width = 0.7;
            let x = i as f64 + 0.5;
            Rectangle::new(
                [(x - bar_width / 2.0, 0.0), (x + bar_width / 2.0, v as f64)],
                RGBColor(46, 204, 113).filled(),
            )
        }))?;

        println!("-> Saved to bar_chart.png");
    }

    // 5. Create area chart (SVG format)
    {
        let root = SVGBackend::new("area_chart.svg", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Generate data
        let x: Vec<f64> = (0..100).map(|x| x as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&x| x.sin() * 0.5 + 0.5).collect();

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Plotters Area Chart Example",
                ("sans-serif", 30).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..10.0, 0.0..1.2)?;

        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc("X Axis")
            .y_desc("Y Axis")
            .draw()?;

        // Draw area chart
        chart
            .draw_series(AreaSeries::new(
                x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
                0.0,
                RGBColor(46, 204, 113).mix(0.2),
            ))?
            .label("0.5*sin(x) + 0.5")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(46, 204, 113)));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        println!("-> Saved to area_chart.svg");
    }

    println!("All chart generation completed.");
    Ok(())
}

#[cfg(not(feature = "visualization"))]
fn main() {
    println!("This example requires the 'visualization' feature to be enabled.");
    println!("Run with: cargo run --example plotters_simple_example --features \"visualization\"");
}
