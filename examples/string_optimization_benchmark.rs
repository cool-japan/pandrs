use pandrs::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use pandrs::OptimizedDataFrame;
use std::time::Instant;
// すべての最適化モードにアクセスするためのインポート
use pandrs::column::string_column_impl::{
    StringColumnOptimizationMode as OptMode, DEFAULT_OPTIMIZATION_MODE,
};

fn main() {
    println!("String Column Optimization Benchmark");
    println!("============================");

    // Data size
    const ROWS: usize = 1_000_000;

    // --- Data Generation ---
    println!("\n[1] Data Generation");
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

    // Benchmark string column creation for each optimization mode
    println!("\n[2] String Column Creation Benchmark");

    // Legacy Mode
    {
        let start = Instant::now();
        let _column = StringColumn::new_legacy(str_data.clone());
        let time = start.elapsed();
        println!("Legacy Mode Creation Time: {time:?}");
    }

    // Global Pool Mode
    {
        let start = Instant::now();
        let _column = StringColumn::new_with_global_pool(str_data.clone());
        let time = start.elapsed();
        println!("Global Pool Mode Creation Time: {time:?}");
    }

    // Categorical Mode
    {
        let start = Instant::now();
        let _column = StringColumn::new_categorical(str_data.clone());
        let time = start.elapsed();
        println!("Categorical Mode Creation Time: {time:?}");
    }

    // Optimized Implementation
    {
        let start = Instant::now();
        let _column = StringColumn::new_categorical(str_data.clone());
        let time = start.elapsed();
        println!("Optimized Implementation Creation Time: {time:?}");
    }

    // Benchmark DataFrame creation for each mode
    println!("\n[3] DataFrame Creation Benchmark for Each Optimization Mode");

    // Legacy Mode
    {
        // Set to Legacy Mode
        unsafe {
            DEFAULT_OPTIMIZATION_MODE = OptMode::Legacy;
        }

        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column(
            "id".to_string(),
            Column::Int64(Int64Column::new(int_data.clone())),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Column::Float64(Float64Column::new(float_data.clone())),
        )
        .unwrap();
        df.add_column(
            "category".to_string(),
            Column::String(StringColumn::new(str_data.clone())),
        )
        .unwrap();
        df.add_column(
            "flag".to_string(),
            Column::Boolean(BooleanColumn::new(bool_data.clone())),
        )
        .unwrap();

        let time = start.elapsed();
        println!("Legacy Mode DataFrame Creation Time: {time:?}");
    }

    // Global Pool Mode
    {
        // Set to Global Pool Mode
        unsafe {
            DEFAULT_OPTIMIZATION_MODE = OptMode::GlobalPool;
        }

        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column(
            "id".to_string(),
            Column::Int64(Int64Column::new(int_data.clone())),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Column::Float64(Float64Column::new(float_data.clone())),
        )
        .unwrap();
        df.add_column(
            "category".to_string(),
            Column::String(StringColumn::new(str_data.clone())),
        )
        .unwrap();
        df.add_column(
            "flag".to_string(),
            Column::Boolean(BooleanColumn::new(bool_data.clone())),
        )
        .unwrap();

        let time = start.elapsed();
        println!("Global Pool Mode DataFrame Creation Time: {time:?}");
    }

    // Categorical Mode
    {
        // Set to Categorical Mode
        unsafe {
            DEFAULT_OPTIMIZATION_MODE = OptMode::Categorical;
        }

        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column(
            "id".to_string(),
            Column::Int64(Int64Column::new(int_data.clone())),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Column::Float64(Float64Column::new(float_data.clone())),
        )
        .unwrap();
        df.add_column(
            "category".to_string(),
            Column::String(StringColumn::new(str_data.clone())),
        )
        .unwrap();
        df.add_column(
            "flag".to_string(),
            Column::Boolean(BooleanColumn::new(bool_data.clone())),
        )
        .unwrap();

        let time = start.elapsed();
        println!("Categorical Mode DataFrame Creation Time: {time:?}");
    }

    // Optimized Implementation
    {
        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column(
            "id".to_string(),
            Column::Int64(Int64Column::new(int_data.clone())),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Column::Float64(Float64Column::new(float_data.clone())),
        )
        .unwrap();
        df.add_column(
            "category".to_string(),
            Column::String(StringColumn::new_categorical(str_data.clone())),
        )
        .unwrap();
        df.add_column(
            "flag".to_string(),
            Column::Boolean(BooleanColumn::new(bool_data.clone())),
        )
        .unwrap();

        let time = start.elapsed();
        println!(
            "Optimized Implementation DataFrame Creation Time: {time:?}"
        );
    }

    println!("\nString Column Optimization Benchmark Completed");
}
