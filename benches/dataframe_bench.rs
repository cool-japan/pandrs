// Rust標準的なベンチマークツール
#![feature(test)]
extern crate test;

use pandrs::{DataFrame, Series};
use std::collections::HashMap;
use test::Bencher;

// 10行DataFrameの作成ベンチマーク
#[bench]
fn bench_create_small_dataframe(b: &mut Bencher) {
    b.iter(|| {
        let col_a = Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("A".to_string())).unwrap();
        let col_b = Series::new(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1], Some("B".to_string())).unwrap();
        let col_c = Series::new(
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
}

// 1,000行DataFrameの作成ベンチマーク
#[bench]
fn bench_create_medium_dataframe(b: &mut Bencher) {
    b.iter(|| {
        let col_a = Series::new((0..1000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new((0..1000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let col_c = Series::new(
            (0..1000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
}

// 10,000行DataFrameの作成ベンチマーク
#[bench]
fn bench_create_large_dataframe(b: &mut Bencher) {
    b.iter(|| {
        let col_a = Series::new((0..10_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new((0..10_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let col_c = Series::new(
            (0..10_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
}

// 100,000行DataFrameの作成ベンチマーク（注：繰り返し回数が少なくなります）
#[bench]
fn bench_create_huge_dataframe(b: &mut Bencher) {
    b.iter(|| {
        let col_a = Series::new((0..100_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new((0..100_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let col_c = Series::new(
            (0..100_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
}

// from_mapメソッドによるDataFrame作成
#[bench]
fn bench_create_medium_dataframe_from_map(b: &mut Bencher) {
    b.iter(|| {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..1000).map(|n| n.to_string()).collect());
        data.insert("B".to_string(), (0..1000).map(|n| format!("{:.1}", n as f64 * 0.5)).collect());
        data.insert("C".to_string(), (0..1000).map(|i| format!("val_{}", i)).collect());
        
        let _ = DataFrame::from_map(data, None).unwrap();
    });
}

// カラムアクセス
#[bench]
fn bench_column_access(b: &mut Bencher) {
    // 事前準備
    let col_a = Series::new((0..10_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
    let col_b = Series::new((0..10_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
    let col_c = Series::new(
        (0..10_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
        Some("C".to_string()),
    ).unwrap();
    
    let mut df = DataFrame::new();
    df.add_column("A".to_string(), col_a).unwrap();
    df.add_column("B".to_string(), col_b).unwrap();
    df.add_column("C".to_string(), col_c).unwrap();
    
    // ベンチマーク実行
    b.iter(|| {
        let _ = df.get_column("A");
        let _ = df.get_column("B");
        let _ = df.get_column("C");
    });
}

// CSVシリアライズ（小）
#[bench]
fn bench_to_csv_small(b: &mut Bencher) {
    // 事前準備
    let col_a = Series::new((0..100).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
    let col_b = Series::new((0..100).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
    let col_c = Series::new(
        (0..100).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
        Some("C".to_string()),
    ).unwrap();
    
    let mut df = DataFrame::new();
    df.add_column("A".to_string(), col_a).unwrap();
    df.add_column("B".to_string(), col_b).unwrap();
    df.add_column("C".to_string(), col_c).unwrap();
    
    // 一時ファイル
    let temp_path = std::env::temp_dir().join("bench_test.csv");
    let path_str = temp_path.to_str().unwrap();
    
    // ベンチマーク実行
    b.iter(|| {
        df.to_csv(path_str).unwrap();
    });
}

// JSONシリアライズ
#[bench]
fn bench_to_json_small(b: &mut Bencher) {
    // 事前準備
    let col_a = Series::new((0..100).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
    let col_b = Series::new((0..100).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
    let col_c = Series::new(
        (0..100).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
        Some("C".to_string()),
    ).unwrap();
    
    let mut df = DataFrame::new();
    df.add_column("A".to_string(), col_a).unwrap();
    df.add_column("B".to_string(), col_b).unwrap();
    df.add_column("C".to_string(), col_c).unwrap();
    
    // ベンチマーク実行
    b.iter(|| {
        let _ = df.to_json().unwrap();
    });
}