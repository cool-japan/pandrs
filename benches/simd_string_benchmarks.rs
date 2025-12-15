//! SIMD String Operations Benchmarks
//!
//! Benchmarks comparing SIMD-accelerated string operations against scalar implementations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pandrs::optimized::jit::simd_string::{
    batch_lowercase, batch_uppercase, count_alpha_simd, count_byte_simd, count_digits_simd,
    count_whitespace_simd, find_byte_simd, is_ascii_simd, parallel_batch_lowercase,
    parallel_batch_uppercase, to_lowercase_simd, to_uppercase_simd, SimdStringStats,
};

/// Generate test strings of various sizes
fn generate_test_strings(count: usize, len: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            let base = format!("Hello World {} Test String ", i);
            base.repeat((len / base.len()).max(1))[..len.min(base.len() * 10)].to_string()
        })
        .collect()
}

/// Generate ASCII strings for benchmarking
fn generate_ascii_strings(count: usize, len: usize) -> Vec<String> {
    (0..count)
        .map(|_| {
            (0..len)
                .map(|i| (b'A' + (i % 26) as u8) as char)
                .collect::<String>()
        })
        .collect()
}

/// Benchmark ASCII detection
fn bench_is_ascii(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_ascii");

    for size in [100, 1000, 10000].iter() {
        let ascii_string: String = (0..*size)
            .map(|i| (b'a' + (i % 26) as u8) as char)
            .collect();
        let non_ascii_string = format!("{}世界", "a".repeat(*size - 6));

        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_ascii", size),
            &ascii_string,
            |b, s| b.iter(|| is_ascii_simd(std::hint::black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_ascii", size),
            &ascii_string,
            |b, s| b.iter(|| s.is_ascii()),
        );

        group.bench_with_input(
            BenchmarkId::new("simd_non_ascii", size),
            &non_ascii_string,
            |b, s| b.iter(|| is_ascii_simd(std::hint::black_box(s))),
        );
    }

    group.finish();
}

/// Benchmark case conversion
fn bench_case_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("case_conversion");

    for size in [100, 1000, 10000].iter() {
        let lower_string: String = (0..*size)
            .map(|i| (b'a' + (i % 26) as u8) as char)
            .collect();
        let upper_string: String = (0..*size)
            .map(|i| (b'A' + (i % 26) as u8) as char)
            .collect();

        group.throughput(Throughput::Bytes(*size as u64));

        // Uppercase benchmarks
        group.bench_with_input(
            BenchmarkId::new("simd_uppercase", size),
            &lower_string,
            |b, s| b.iter(|| to_uppercase_simd(std::hint::black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_uppercase", size),
            &lower_string,
            |b, s| b.iter(|| s.to_uppercase()),
        );

        // Lowercase benchmarks
        group.bench_with_input(
            BenchmarkId::new("simd_lowercase", size),
            &upper_string,
            |b, s| b.iter(|| to_lowercase_simd(std::hint::black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_lowercase", size),
            &upper_string,
            |b, s| b.iter(|| s.to_lowercase()),
        );
    }

    group.finish();
}

/// Benchmark character classification
fn bench_char_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_classification");

    for size in [100, 1000, 10000].iter() {
        let mixed_string: String = (0..*size)
            .map(|i| match i % 4 {
                0 => (b'a' + (i % 26) as u8) as char,
                1 => (b'0' + (i % 10) as u8) as char,
                2 => ' ',
                _ => (b'A' + (i % 26) as u8) as char,
            })
            .collect();

        group.throughput(Throughput::Bytes(*size as u64));

        // Count digits
        group.bench_with_input(
            BenchmarkId::new("simd_count_digits", size),
            &mixed_string,
            |b, s| b.iter(|| count_digits_simd(std::hint::black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_count_digits", size),
            &mixed_string,
            |b, s| b.iter(|| s.chars().filter(|c| c.is_ascii_digit()).count()),
        );

        // Count alpha
        group.bench_with_input(
            BenchmarkId::new("simd_count_alpha", size),
            &mixed_string,
            |b, s| b.iter(|| count_alpha_simd(std::hint::black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_count_alpha", size),
            &mixed_string,
            |b, s| b.iter(|| s.chars().filter(|c| c.is_ascii_alphabetic()).count()),
        );

        // Count whitespace
        group.bench_with_input(
            BenchmarkId::new("simd_count_whitespace", size),
            &mixed_string,
            |b, s| b.iter(|| count_whitespace_simd(std::hint::black_box(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_count_whitespace", size),
            &mixed_string,
            |b, s| b.iter(|| s.chars().filter(|c| c.is_ascii_whitespace()).count()),
        );
    }

    group.finish();
}

/// Benchmark byte search
fn bench_byte_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_search");

    for size in [100, 1000, 10000].iter() {
        let test_string: String = (0..*size)
            .map(|i| (b'a' + (i % 26) as u8) as char)
            .collect();
        let bytes = test_string.as_bytes();

        group.throughput(Throughput::Bytes(*size as u64));

        // Find byte (early)
        group.bench_with_input(
            BenchmarkId::new("simd_find_early", size),
            bytes,
            |b, data| b.iter(|| find_byte_simd(std::hint::black_box(data), b'c')),
        );

        group.bench_with_input(
            BenchmarkId::new("std_find_early", size),
            bytes,
            |b, data| b.iter(|| data.iter().position(|&x| x == b'c')),
        );

        // Find byte (late)
        group.bench_with_input(
            BenchmarkId::new("simd_find_late", size),
            bytes,
            |b, data| b.iter(|| find_byte_simd(std::hint::black_box(data), b'z')),
        );

        group.bench_with_input(BenchmarkId::new("std_find_late", size), bytes, |b, data| {
            b.iter(|| data.iter().position(|&x| x == b'z'))
        });

        // Count byte
        group.bench_with_input(
            BenchmarkId::new("simd_count_byte", size),
            bytes,
            |b, data| b.iter(|| count_byte_simd(std::hint::black_box(data), b'a')),
        );

        group.bench_with_input(
            BenchmarkId::new("std_count_byte", size),
            bytes,
            |b, data| b.iter(|| data.iter().filter(|&&x| x == b'a').count()),
        );
    }

    group.finish();
}

/// Benchmark batch operations
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for count in [100, 1000].iter() {
        let strings = generate_ascii_strings(*count, 100);

        group.throughput(Throughput::Elements(*count as u64));

        // Batch uppercase
        group.bench_with_input(
            BenchmarkId::new("simd_batch_upper", count),
            &strings,
            |b, data| b.iter(|| batch_uppercase(std::hint::black_box(data))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_batch_upper", count),
            &strings,
            |b, data| {
                b.iter(|| {
                    data.iter()
                        .map(|s| s.to_uppercase())
                        .collect::<Vec<String>>()
                })
            },
        );

        // Batch lowercase
        group.bench_with_input(
            BenchmarkId::new("simd_batch_lower", count),
            &strings,
            |b, data| b.iter(|| batch_lowercase(std::hint::black_box(data))),
        );

        group.bench_with_input(
            BenchmarkId::new("std_batch_lower", count),
            &strings,
            |b, data| {
                b.iter(|| {
                    data.iter()
                        .map(|s| s.to_lowercase())
                        .collect::<Vec<String>>()
                })
            },
        );

        // Parallel batch (only for larger counts)
        if *count >= 1000 {
            group.bench_with_input(
                BenchmarkId::new("parallel_batch_upper", count),
                &strings,
                |b, data| b.iter(|| parallel_batch_uppercase(std::hint::black_box(data))),
            );

            group.bench_with_input(
                BenchmarkId::new("parallel_batch_lower", count),
                &strings,
                |b, data| b.iter(|| parallel_batch_lowercase(std::hint::black_box(data))),
            );
        }
    }

    group.finish();
}

/// Print SIMD feature detection info
fn print_simd_info() {
    let stats = SimdStringStats::new();
    println!("\n=== SIMD String Operations Info ===");
    println!("SIMD Level: {}", stats.simd_level());
    println!("AVX2 Available: {}", stats.avx2_available);
    println!("SSE2 Available: {}", stats.sse2_available);
    println!("===================================\n");
}

criterion_group!(
    benches,
    bench_is_ascii,
    bench_case_conversion,
    bench_char_classification,
    bench_byte_search,
    bench_batch_operations,
);

criterion_main!(benches);
