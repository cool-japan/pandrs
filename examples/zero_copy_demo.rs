//! Zero-Copy Data Views and Cache-Aware Memory Management Example
//!
//! This example demonstrates the zero-copy data views and cache-aware memory
//! management capabilities in PandRS, showing how to achieve optimal performance
//! through efficient memory usage and cache-friendly operations.

use pandrs::storage::{
    CacheAwareOps, CacheTopology, MemoryMappedView, ZeroCopyManager, CACHE_LINE_SIZE, PAGE_SIZE,
};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 PandRS Zero-Copy Data Views and Cache-Aware Memory Management Example");
    println!("=========================================================================\n");

    // Demonstrate cache topology detection
    demonstrate_cache_topology()?;

    // Demonstrate zero-copy views
    demonstrate_zero_copy_views()?;

    // Demonstrate memory-mapped views
    demonstrate_memory_mapped_views()?;

    // Demonstrate cache-aware operations
    demonstrate_cache_aware_operations()?;

    // Demonstrate performance optimizations
    demonstrate_performance_optimizations()?;

    println!("\n🎯 Zero-copy data views example completed successfully!");
    Ok(())
}

fn demonstrate_cache_topology() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Cache Topology Detection");
    println!("----------------------------");

    let topology = CacheTopology::detect()?;

    println!("🔍 System Cache Information:");
    println!(
        "   L1 Cache Size: {:.1} KB",
        topology.l1_cache_size as f64 / 1024.0
    );
    println!(
        "   L2 Cache Size: {:.1} KB",
        topology.l2_cache_size as f64 / 1024.0
    );
    println!(
        "   L3 Cache Size: {:.1} MB",
        topology.l3_cache_size as f64 / (1024.0 * 1024.0)
    );
    println!("   Cache Line Size: {} bytes", topology.cache_line_size);
    println!("   CPU Cores: {}", topology.cpu_cores);
    if let Some(numa_node) = topology.numa_node {
        println!("   NUMA Node: {}", numa_node);
    }

    // Test optimal cache level selection
    let test_sizes = vec![
        16 * 1024,        // 16KB
        128 * 1024,       // 128KB
        2 * 1024 * 1024,  // 2MB
        32 * 1024 * 1024, // 32MB
    ];

    println!("\n📊 Optimal Cache Level Selection:");
    for size in test_sizes {
        let level = topology.optimal_cache_level(size);
        println!("   {} -> {:?}", format_size(size), level);
    }

    println!();
    Ok(())
}

fn demonstrate_zero_copy_views() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Zero-Copy Views");
    println!("------------------");

    let manager = ZeroCopyManager::new()?;

    // Create test data
    let data = (0..10000).map(|i| i as f64).collect::<Vec<_>>();
    println!(
        "🔬 Created test data: {} elements ({} KB)",
        data.len(),
        (data.len() * std::mem::size_of::<f64>()) / 1024
    );

    // Create zero-copy view
    let view = manager.create_view(data)?;
    println!("✅ Created zero-copy view: {} elements", view.len());
    println!("   Memory address: 0x{:x}", view.memory_address());
    println!("   Cache aligned: {}", view.is_cache_aligned());
    println!("   Capacity: {}", view.capacity());

    // Test view operations
    println!("\n🔍 View Operations:");
    println!("   First 5 elements: {:?}", &view.as_slice()[0..5]);
    println!(
        "   Last 5 elements: {:?}",
        &view.as_slice()[view.len() - 5..]
    );

    // Create subview
    let subview = view.subview(1000..2000)?;
    println!("   Subview (1000..2000): {} elements", subview.len());
    println!("   Subview first 5: {:?}", &subview.as_slice()[0..5]);

    // Memory layout information
    let layout = view.layout();
    println!("\n📐 Memory Layout:");
    println!("   Element size: {} bytes", layout.element_size);
    println!("   Stride: {} bytes", layout.stride);
    println!("   Alignment: {} bytes", layout.alignment);
    println!("   Cache aligned: {}", layout.cache_aligned);

    // Get manager statistics
    let stats = manager.stats()?;
    println!("\n📈 Manager Statistics:");
    println!("   Views created: {}", stats.views_created);
    println!("   Total memory: {}", format_size(stats.total_memory));

    println!();
    Ok(())
}

fn demonstrate_memory_mapped_views() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗺️  Memory-Mapped Views");
    println!("------------------------");

    // Create a temporary file with test data
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("test_data.bin");

    {
        let mut file = File::create(&file_path)?;
        let data: Vec<f64> = (0..10000).map(|i| i as f64 * 0.5).collect();

        // Write binary data to file
        for value in &data {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }

    println!("📄 Created test file: {}", file_path.display());
    println!(
        "   File size: {}",
        format_size(std::fs::metadata(&file_path)?.len() as usize)
    );

    // Create memory-mapped view
    let manager = ZeroCopyManager::new()?;
    let mmap_view: MemoryMappedView<f64> =
        manager.create_mmap_view(file_path.to_str().unwrap(), 10000)?;

    println!(
        "✅ Created memory-mapped view: {} elements",
        mmap_view.len()
    );

    // Test memory-mapped operations
    println!("\n🔍 Memory-Mapped Operations:");
    println!("   First 5 elements: {:?}", &mmap_view.as_slice()[0..5]);
    println!(
        "   Last 5 elements: {:?}",
        &mmap_view.as_slice()[mmap_view.len() - 5..]
    );

    // Layout information
    let layout = mmap_view.layout();
    println!("\n📐 Memory-Mapped Layout:");
    println!("   Start address: 0x{:x}", layout.start_address);
    println!("   Element size: {} bytes", layout.element_size);
    println!("   Alignment: {} bytes", layout.alignment);

    // Statistics
    let stats = manager.stats()?;
    println!("\n📈 Updated Statistics:");
    println!("   Memory-mapped views: {}", stats.mmap_views_created);
    println!(
        "   Total memory managed: {}",
        format_size(stats.total_memory)
    );

    println!();
    Ok(())
}

fn demonstrate_cache_aware_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Cache-Aware Operations");
    println!("-------------------------");

    let manager = ZeroCopyManager::new()?;

    // Create large dataset for cache-aware operations
    let size = 100000;
    let data = (0..size).map(|i| (i as f64).sin()).collect::<Vec<_>>();
    let view = manager.create_view(data)?;

    println!(
        "🔬 Created large dataset: {} elements ({} MB)",
        view.len(),
        (view.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0)
    );

    // Test optimal block size calculation
    let block_size = view.optimal_block_size();
    println!(
        "🧮 Optimal block size: {} elements ({} KB)",
        block_size,
        (block_size * std::mem::size_of::<f64>()) / 1024
    );

    // Perform cache-friendly linear scan
    let start_time = std::time::Instant::now();
    let positive_indices = view.linear_scan(|&x| x > 0.0);
    let scan_duration = start_time.elapsed();

    println!("\n🔍 Linear Scan Results:");
    println!("   Positive values found: {}", positive_indices.len());
    println!(
        "   Scan duration: {:.2} ms",
        scan_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   Throughput: {:.1} MB/s",
        (view.len() * std::mem::size_of::<f64>()) as f64
            / (1024.0 * 1024.0 * scan_duration.as_secs_f64())
    );

    // Test blocked operations
    let other_data = vec![2.0f64; view.len()];
    let start_time = std::time::Instant::now();
    let results = view.blocked_operation(&other_data, block_size, |&a, &b| a * b);
    let blocked_duration = start_time.elapsed();

    println!("\n🔄 Blocked Operation Results:");
    println!("   Results computed: {}", results.len());
    println!(
        "   Operation duration: {:.2} ms",
        blocked_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   Throughput: {:.1} MB/s",
        (results.len() * std::mem::size_of::<f64>() * 2) as f64
            / (1024.0 * 1024.0 * blocked_duration.as_secs_f64())
    );

    // Test prefetching
    let prefetch_indices: Vec<usize> = (0..view.len()).step_by(1000).collect();
    println!("\n🚀 Prefetching {} cache lines", prefetch_indices.len());
    view.prefetch(&prefetch_indices);

    println!();
    Ok(())
}

fn demonstrate_performance_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏎️  Performance Optimizations");
    println!("------------------------------");

    let manager = ZeroCopyManager::new()?;

    // Compare different data sizes and their cache behavior
    let test_sizes = vec![
        1024,        // 1K elements - fits in L1
        32 * 1024,   // 32K elements - fits in L2
        1024 * 1024, // 1M elements - requires L3/Memory
    ];

    for &size in &test_sizes {
        println!(
            "\n📊 Testing size: {} elements ({} KB)",
            size,
            (size * std::mem::size_of::<f64>()) / 1024
        );

        // Create data and view
        let data = (0..size).map(|i| i as f64).collect::<Vec<_>>();
        let view = manager.create_view(data)?;

        // Determine expected cache level
        let topology = CacheTopology::detect()?;
        let expected_level = topology.optimal_cache_level(size * std::mem::size_of::<f64>());
        println!("   Expected cache level: {:?}", expected_level);

        // Performance test: sum all elements
        let iterations = 100;
        let mut total_time = std::time::Duration::ZERO;

        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _sum: f64 = view.as_slice().iter().sum();
            total_time += start.elapsed();
        }

        let avg_time = total_time / iterations;
        let throughput =
            (size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0 * avg_time.as_secs_f64());

        println!(
            "   Average time: {:.2} μs",
            avg_time.as_secs_f64() * 1_000_000.0
        );
        println!("   Throughput: {:.1} MB/s", throughput);
        println!("   Cache aligned: {}", view.is_cache_aligned());
    }

    // Memory alignment demonstration
    println!("\n🎯 Memory Alignment:");
    println!("   Cache line size: {} bytes", CACHE_LINE_SIZE);
    println!("   Page size: {} bytes", PAGE_SIZE);

    let data = vec![42i32; 1000];
    let view = manager.create_view(data)?;
    let address = view.memory_address();

    println!("   View address: 0x{:x}", address);
    println!("   Cache line aligned: {}", address % CACHE_LINE_SIZE == 0);
    println!("   Page aligned: {}", address % PAGE_SIZE == 0);

    // Final statistics
    let final_stats = manager.stats()?;
    println!("\n📈 Final Statistics:");
    println!("   Total views created: {}", final_stats.views_created);
    println!("   Memory-mapped views: {}", final_stats.mmap_views_created);
    println!(
        "   Total memory managed: {}",
        format_size(final_stats.total_memory)
    );

    println!();
    Ok(())
}

/// Format byte size in human-readable format
fn format_size(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_example() {
        // This test ensures the example compiles and basic functionality works
        let manager = ZeroCopyManager::new().unwrap();
        let data = vec![1, 2, 3, 4, 5];
        let view = manager.create_view(data).unwrap();

        assert_eq!(view.len(), 5);
        assert_eq!(view.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_cache_topology() {
        let topology = CacheTopology::detect().unwrap();
        assert!(topology.l1_cache_size > 0);
        assert!(topology.l2_cache_size > 0);
        assert!(topology.l3_cache_size > 0);
        assert!(topology.cpu_cores > 0);
    }

    #[test]
    fn test_cache_aware_operations() {
        let manager = ZeroCopyManager::new().unwrap();
        let data = (0..1000).collect::<Vec<i32>>();
        let view = manager.create_view(data).unwrap();

        // Test linear scan
        let evens = view.linear_scan(|&x| x % 2 == 0);
        assert_eq!(evens.len(), 500);

        // Test blocked operation
        let other = vec![2; 1000];
        let results = view.blocked_operation(&other, 100, |&a, &b| a * b);
        assert_eq!(results.len(), 1000);
    }

    #[test]
    fn test_subview_operations() {
        let manager = ZeroCopyManager::new().unwrap();
        let data = (0..100).collect::<Vec<i32>>();
        let view = manager.create_view(data).unwrap();

        let subview = view.subview(10..20).unwrap();
        assert_eq!(subview.len(), 10);
        assert_eq!(subview.as_slice()[0], 10);
        assert_eq!(subview.as_slice()[9], 19);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1536), "1.5 KB");
    }
}
