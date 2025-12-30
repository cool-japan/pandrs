use pandrs::column::{
    BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn,
};
use pandrs::{AggregateOp, LazyFrame, OptimizedDataFrame};
use std::time::Instant;

fn main() {
    println!("Parallel Processing Performance Benchmark");
    println!("============================");

    // Data size
    const ROWS: usize = 1_000_000;

    // ====================
    // Create Large DataFrame
    // ====================
    println!("\n[1] Create Large DataFrame ({ROWS} rows)");

    // --- Data Generation ---
    let data_gen_start = Instant::now();

    let mut int_data = Vec::with_capacity(ROWS);
    let mut float_data = Vec::with_capacity(ROWS);
    let mut str_data = Vec::with_capacity(ROWS);
    let mut bool_data = Vec::with_capacity(ROWS);

    for i in 0..ROWS {
        int_data.push(i as i64);
        float_data.push(i as f64 / 100.0);
        str_data.push(format!("value_{}", i % 1000)); // Limited string set
        bool_data.push(i % 2 == 0);
    }

    let data_gen_time = data_gen_start.elapsed();
    println!("Data generation time: {data_gen_time:?}");

    // --- DataFrame Construction ---
    let df_construct_start = Instant::now();
    let mut df = OptimizedDataFrame::new();

    df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data)))
        .unwrap();
    df.add_column(
        "value".to_string(),
        Column::Float64(Float64Column::new(float_data)),
    )
    .unwrap();
    df.add_column(
        "category".to_string(),
        Column::String(StringColumn::new(str_data)),
    )
    .unwrap();
    df.add_column(
        "flag".to_string(),
        Column::Boolean(BooleanColumn::new(bool_data)),
    )
    .unwrap();

    let df_construct_time = df_construct_start.elapsed();
    println!("DataFrame construction time: {df_construct_time:?}");
    println!(
        "Total DataFrame creation time: {:?}",
        data_gen_time + df_construct_time
    );

    // ====================
    // Serial vs Parallel Filtering
    // ====================
    println!("\n[2] Filtering Performance (id > 500000)");

    // Add a condition column
    let condition_data: Vec<bool> = (0..ROWS).map(|i| i > ROWS / 2).collect();
    df.add_column(
        "filter_condition".to_string(),
        Column::Boolean(BooleanColumn::new(condition_data)),
    )
    .unwrap();

    // Serial filtering
    let mut serial_total = std::time::Duration::new(0, 0);
    for _ in 0..3 {
        let start = Instant::now();
        let filtered_df = df.filter("filter_condition").unwrap();
        let duration = start.elapsed();
        serial_total += duration;
        println!("Serial filtering time (1 run): {duration:?}");
        println!("Filtered row count: {}", filtered_df.row_count());
    }
    let serial_time = serial_total / 3;
    println!("Average serial filtering time: {serial_time:?}");

    // Parallel filtering
    let mut parallel_total = std::time::Duration::new(0, 0);
    for _ in 0..3 {
        let start = Instant::now();
        let par_filtered_df = df.par_filter("filter_condition").unwrap();
        let duration = start.elapsed();
        parallel_total += duration;
        println!("Parallel filtering time (1 run): {duration:?}");
        println!("Filtered row count: {}", par_filtered_df.row_count());
    }
    let parallel_time = parallel_total / 3;
    println!("Average parallel filtering time: {parallel_time:?}");

    println!(
        "Speedup: {:.2}x",
        serial_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    // ====================
    // Grouping and Aggregation
    // ====================
    println!("\n[3] Grouping and Aggregation (average by category)");

    // Serial grouping and aggregation
    let small_df = df.select(&["category", "value"]).unwrap();

    let mut serial_total = std::time::Duration::new(0, 0);
    for _ in 0..3 {
        let start = Instant::now();
        let lazy_df = LazyFrame::new(small_df.clone());
        let grouped_df = lazy_df
            .aggregate(
                vec!["category".to_string()],
                vec![(
                    "value".to_string(),
                    AggregateOp::Mean,
                    "value_mean".to_string(),
                )],
            )
            .execute()
            .unwrap();
        let duration = start.elapsed();
        serial_total += duration;
        println!("Serial grouping and aggregation time (1 run): {duration:?}");
        println!("Group count: {}", grouped_df.row_count());
    }
    let serial_time = serial_total / 3;
    println!("Average serial grouping and aggregation time: {serial_time:?}");

    // Parallel grouping and aggregation
    let mut parallel_total = std::time::Duration::new(0, 0);
    for _ in 0..3 {
        let start = Instant::now();
        let grouped_map = small_df.par_groupby(&["category"]).unwrap();

        let mut result_df = OptimizedDataFrame::new();
        let mut categories = Vec::with_capacity(grouped_map.len());
        let mut means = Vec::with_capacity(grouped_map.len());

        for (category, group_df) in &grouped_map {
            categories.push(category.clone());

            let value_col = group_df.column("value").unwrap();
            if let Some(float_col) = value_col.as_float64() {
                let mut sum = 0.0;
                let mut count = 0;

                for i in 0..float_col.len() {
                    if let Ok(Some(val)) = float_col.get(i) {
                        sum += val;
                        count += 1;
                    }
                }

                let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                means.push(mean);
            }
        }

        result_df
            .add_column(
                "category".to_string(),
                Column::String(StringColumn::new(categories)),
            )
            .unwrap();
        result_df
            .add_column(
                "value_mean".to_string(),
                Column::Float64(Float64Column::new(means)),
            )
            .unwrap();

        let duration = start.elapsed();
        parallel_total += duration;
        println!("Parallel grouping and aggregation time (1 run): {duration:?}");
        println!("Group count: {}", result_df.row_count());
    }
    let parallel_time = parallel_total / 3;
    println!("Average parallel grouping and aggregation time: {parallel_time:?}");

    println!(
        "Speedup: {:.2}x",
        serial_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    // ====================
    // Computation (double all values)
    // ====================
    println!("\n[4] Computation (double the values in the 'value' column)");

    // Serial computation
    let start = Instant::now();
    let mut computed_df = OptimizedDataFrame::new();

    for name in df.column_names() {
        let col_view = df.column(name).unwrap();

        let new_col = if name == "value" {
            let float_col = col_view.as_float64().unwrap();
            let mut doubled_values = Vec::with_capacity(float_col.len());

            for i in 0..float_col.len() {
                if let Ok(Some(val)) = float_col.get(i) {
                    doubled_values.push(val * 2.0);
                } else {
                    doubled_values.push(0.0);
                }
            }

            Column::Float64(Float64Column::new(doubled_values))
        } else {
            col_view.into_column()
        };

        computed_df.add_column(name.to_string(), new_col).unwrap();
    }

    let serial_time = start.elapsed();
    println!("Serial computation time: {serial_time:?}");

    // Parallel computation
    let start = Instant::now();
    let _par_computed_df = df
        .par_apply(|view| {
            if view.as_float64().is_some() {
                if let Some(float_col) = view.as_float64() {
                    use rayon::prelude::*;

                    let values = (0..float_col.len())
                        .into_par_iter()
                        .map(|i| {
                            if let Ok(Some(val)) = float_col.get(i) {
                                val * 2.0
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>();

                    Ok(Column::Float64(Float64Column::new(values)))
                } else {
                    Ok(view.clone().into_column())
                }
            } else {
                Ok(view.clone().into_column())
            }
        })
        .unwrap();

    let parallel_time = start.elapsed();
    println!("Parallel computation time: {parallel_time:?}");

    println!(
        "Speedup: {:.2}x",
        serial_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    println!("\nParallel Benchmark Complete");
}
