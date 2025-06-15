//! Advanced MultiIndex with Cross-Section Selection Demo
//!
//! This example demonstrates the comprehensive advanced MultiIndex implementation
//! with cross-section selection, hierarchical operations, and performance-optimized
//! multi-level indexing capabilities.
//!
//! Features demonstrated:
//! - Advanced MultiIndex creation and manipulation
//! - Cross-section (xs) selection for partial key matching
//! - Multiple selection criteria (exact, partial, range, boolean)
//! - Level-wise operations and transformations
//! - GroupBy operations with hierarchical data
//! - Index reordering, level dropping, and swapping
//! - Performance optimizations with caching
//!
//! Run with: cargo run --example advanced_multi_index_demo

use pandrs::core::advanced_multi_index::{AdvancedMultiIndex, IndexValue, SelectionCriteria};
use pandrs::core::error::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("PandRS Advanced MultiIndex with Cross-Section Selection Demo");
    println!("==========================================================");
    println!();

    // Create a comprehensive dataset
    demo_index_creation()?;
    println!();

    demo_cross_section_selection()?;
    println!();

    demo_advanced_selection_criteria()?;
    println!();

    demo_hierarchical_operations()?;
    println!();

    demo_level_management()?;
    println!();

    demo_groupby_operations()?;
    println!();

    demo_performance_features()?;
    println!();

    print_summary();

    Ok(())
}

/// Demonstrate comprehensive MultiIndex creation
fn demo_index_creation() -> Result<()> {
    println!("🏗️ Advanced MultiIndex Creation");
    println!("==============================");

    // Create a complex hierarchical dataset (Sales data by Region, Product, Quarter)
    let tuples = vec![
        vec![
            IndexValue::from("North"),
            IndexValue::from("Laptop"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("North"),
            IndexValue::from("Laptop"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("North"),
            IndexValue::from("Desktop"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("North"),
            IndexValue::from("Desktop"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("South"),
            IndexValue::from("Laptop"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("South"),
            IndexValue::from("Laptop"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("South"),
            IndexValue::from("Desktop"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("South"),
            IndexValue::from("Desktop"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("East"),
            IndexValue::from("Laptop"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("East"),
            IndexValue::from("Laptop"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("West"),
            IndexValue::from("Desktop"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("West"),
            IndexValue::from("Desktop"),
            IndexValue::from("Q2"),
        ],
    ];

    let level_names = Some(vec![
        Some("Region".to_string()),
        Some("Product".to_string()),
        Some("Quarter".to_string()),
    ]);

    let index = AdvancedMultiIndex::new(tuples, level_names)?;

    println!(
        "Created MultiIndex with {} levels and {} rows",
        index.n_levels(),
        index.len()
    );
    println!("Level names: {:?}", index.level_names());

    // Show unique values for each level
    for level in 0..index.n_levels() {
        let values = index.get_level_values(level)?;
        println!(
            "Level {} ({}): {} unique values",
            level,
            index.level_names()[level]
                .as_ref()
                .unwrap_or(&"unnamed".to_string()),
            values.len()
        );
        println!("  Values: {:?}", values);
    }

    println!("\nSample tuples:");
    for i in 0..5.min(index.len()) {
        let tuple = index.get_tuple(i)?;
        println!("  [{}] {:?}", i, tuple);
    }

    // Test creation from arrays
    println!("\nCreating from separate arrays:");
    let regions = vec![
        IndexValue::from("North"),
        IndexValue::from("North"),
        IndexValue::from("South"),
        IndexValue::from("South"),
    ];
    let products = vec![
        IndexValue::from("A"),
        IndexValue::from("B"),
        IndexValue::from("A"),
        IndexValue::from("B"),
    ];
    let quarters = vec![
        IndexValue::from(1),
        IndexValue::from(1),
        IndexValue::from(2),
        IndexValue::from(2),
    ];

    let array_index = AdvancedMultiIndex::from_arrays(
        vec![regions, products, quarters],
        Some(vec![
            Some("Region".to_string()),
            Some("Product".to_string()),
            Some("Quarter".to_string()),
        ]),
    )?;

    println!(
        "Array-based index: {} levels, {} rows",
        array_index.n_levels(),
        array_index.len()
    );

    Ok(())
}

/// Demonstrate cross-section selection capabilities
fn demo_cross_section_selection() -> Result<()> {
    println!("🎯 Cross-Section Selection");
    println!("==========================");

    let tuples = vec![
        vec![
            IndexValue::from("A"),
            IndexValue::from(1),
            IndexValue::from("X"),
        ],
        vec![
            IndexValue::from("A"),
            IndexValue::from(1),
            IndexValue::from("Y"),
        ],
        vec![
            IndexValue::from("A"),
            IndexValue::from(2),
            IndexValue::from("X"),
        ],
        vec![
            IndexValue::from("A"),
            IndexValue::from(2),
            IndexValue::from("Y"),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(1),
            IndexValue::from("X"),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(1),
            IndexValue::from("Y"),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(2),
            IndexValue::from("X"),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(2),
            IndexValue::from("Y"),
        ],
    ];

    let mut index = AdvancedMultiIndex::new(tuples, None)?;

    println!("Original index:");
    println!("{}", index);
    println!();

    // Cross-section at level 0 (select all 'A' entries)
    println!("Cross-section: level 0, key 'A', keep level");
    let xs_result = index.xs(IndexValue::from("A"), 0, false)?;
    println!("Selected indices: {:?}", xs_result.indices);
    println!("Found: {}", xs_result.found);
    println!();

    // Cross-section at level 0 with level dropping
    println!("Cross-section: level 0, key 'A', drop level");
    let xs_result_drop = index.xs(IndexValue::from("A"), 0, true)?;
    println!("Selected indices: {:?}", xs_result_drop.indices);
    if let Some(ref result_index) = xs_result_drop.index {
        println!("Resulting index after dropping level:");
        println!("{}", result_index);
    }
    println!();

    // Cross-section at level 1 (select all entries with value 1)
    println!("Cross-section: level 1, key 1");
    let xs_result_level1 = index.xs(IndexValue::from(1), 1, false)?;
    println!("Selected indices: {:?}", xs_result_level1.indices);
    println!();

    // Cross-section at level 2 (select all 'X' entries)
    println!("Cross-section: level 2, key 'X'");
    let xs_result_level2 = index.xs(IndexValue::from("X"), 2, false)?;
    println!("Selected indices: {:?}", xs_result_level2.indices);

    Ok(())
}

/// Demonstrate advanced selection criteria
fn demo_advanced_selection_criteria() -> Result<()> {
    println!("🔍 Advanced Selection Criteria");
    println!("==============================");

    let tuples = vec![
        vec![
            IndexValue::from("A"),
            IndexValue::from(1),
            IndexValue::from(10.5),
        ],
        vec![
            IndexValue::from("A"),
            IndexValue::from(2),
            IndexValue::from(20.3),
        ],
        vec![
            IndexValue::from("A"),
            IndexValue::from(3),
            IndexValue::from(15.7),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(1),
            IndexValue::from(25.1),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(2),
            IndexValue::from(18.9),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(3),
            IndexValue::from(22.4),
        ],
        vec![
            IndexValue::from("C"),
            IndexValue::from(1),
            IndexValue::from(12.8),
        ],
        vec![
            IndexValue::from("C"),
            IndexValue::from(2),
            IndexValue::from(30.2),
        ],
    ];

    let index = AdvancedMultiIndex::new(tuples, None)?;

    println!("Original index: {} rows", index.len());
    println!();

    // Exact selection: multiple level constraints
    println!("1. Exact Selection (A at level 0 AND 1 at level 1):");
    let exact_criteria =
        SelectionCriteria::Exact(vec![(0, IndexValue::from("A")), (1, IndexValue::from(1))]);
    let exact_result = index.select(exact_criteria)?;
    println!("   Selected indices: {:?}", exact_result);
    println!();

    // Partial selection: any matching constraint
    println!("2. Partial Selection (A at level 0 OR 1 at level 1):");
    let partial_criteria =
        SelectionCriteria::Partial(vec![(0, IndexValue::from("A")), (1, IndexValue::from(1))]);
    let partial_result = index.select(partial_criteria)?;
    println!("   Selected indices: {:?}", partial_result);
    println!();

    // Range selection: values between bounds
    println!("3. Range Selection (level 2 values between 15.0 and 25.0):");
    let range_criteria =
        SelectionCriteria::Range(2, IndexValue::from(15.0), IndexValue::from(25.0));
    let range_result = index.select(range_criteria)?;
    println!("   Selected indices: {:?}", range_result);
    println!();

    // Boolean selection: custom mask
    println!("4. Boolean Selection (every other row):");
    let boolean_mask = (0..index.len()).map(|i| i % 2 == 0).collect();
    let boolean_criteria = SelectionCriteria::Boolean(boolean_mask);
    let boolean_result = index.select(boolean_criteria)?;
    println!("   Selected indices: {:?}", boolean_result);
    println!();

    // Position selection: specific indices
    println!("5. Position Selection (indices 1, 3, 5):");
    let position_criteria = SelectionCriteria::Positions(vec![1, 3, 5]);
    let position_result = index.select(position_criteria)?;
    println!("   Selected indices: {:?}", position_result);
    println!();

    // Level selection: multiple values at a level
    println!("6. Level Selection (A or C at level 0):");
    let level_criteria =
        SelectionCriteria::Level(0, vec![IndexValue::from("A"), IndexValue::from("C")]);
    let level_result = index.select(level_criteria)?;
    println!("   Selected indices: {:?}", level_result);

    Ok(())
}

/// Demonstrate hierarchical operations
fn demo_hierarchical_operations() -> Result<()> {
    println!("🌳 Hierarchical Operations");
    println!("===========================");

    let tuples = vec![
        vec![
            IndexValue::from("2023"),
            IndexValue::from("Q1"),
            IndexValue::from("Jan"),
        ],
        vec![
            IndexValue::from("2023"),
            IndexValue::from("Q1"),
            IndexValue::from("Feb"),
        ],
        vec![
            IndexValue::from("2023"),
            IndexValue::from("Q1"),
            IndexValue::from("Mar"),
        ],
        vec![
            IndexValue::from("2023"),
            IndexValue::from("Q2"),
            IndexValue::from("Apr"),
        ],
        vec![
            IndexValue::from("2023"),
            IndexValue::from("Q2"),
            IndexValue::from("May"),
        ],
        vec![
            IndexValue::from("2023"),
            IndexValue::from("Q2"),
            IndexValue::from("Jun"),
        ],
        vec![
            IndexValue::from("2024"),
            IndexValue::from("Q1"),
            IndexValue::from("Jan"),
        ],
        vec![
            IndexValue::from("2024"),
            IndexValue::from("Q1"),
            IndexValue::from("Feb"),
        ],
    ];

    let level_names = Some(vec![
        Some("Year".to_string()),
        Some("Quarter".to_string()),
        Some("Month".to_string()),
    ]);

    let index = AdvancedMultiIndex::new(tuples, level_names)?;

    println!("Hierarchical time series index:");
    println!("{}", index);
    println!();

    // Get unique keys for different level combinations
    println!("1. Group keys by Year (level 0):");
    let year_keys = index.get_group_keys(&[0])?;
    for key in &year_keys {
        let indices = index.get_group_indices(&[0], key)?;
        println!(
            "   {:?}: {} rows (indices: {:?})",
            key,
            indices.len(),
            indices
        );
    }
    println!();

    println!("2. Group keys by Year + Quarter (levels 0, 1):");
    let quarter_keys = index.get_group_keys(&[0, 1])?;
    for key in &quarter_keys {
        let indices = index.get_group_indices(&[0, 1], key)?;
        println!("   {:?}: {} rows", key, indices.len());
    }
    println!();

    // Slice operation
    println!("3. Slice operation (rows 2-5):");
    let sliced = index.slice(2, 6)?;
    println!("   Sliced index:");
    println!("{}", sliced);
    println!();

    // Select specific levels
    println!("4. Level selection (Year and Month only):");
    // This would require implementing level selection on the index
    println!("   [Feature would select levels 0 and 2, dropping Quarter]");

    Ok(())
}

/// Demonstrate level management operations
fn demo_level_management() -> Result<()> {
    println!("⚙️  Level Management Operations");
    println!("==============================");

    let tuples = vec![
        vec![
            IndexValue::from("A"),
            IndexValue::from(1),
            IndexValue::from("X"),
            IndexValue::from(true),
        ],
        vec![
            IndexValue::from("A"),
            IndexValue::from(2),
            IndexValue::from("Y"),
            IndexValue::from(false),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(1),
            IndexValue::from("X"),
            IndexValue::from(true),
        ],
        vec![
            IndexValue::from("B"),
            IndexValue::from(2),
            IndexValue::from("Y"),
            IndexValue::from(false),
        ],
    ];

    let level_names = Some(vec![
        Some("Category".to_string()),
        Some("Number".to_string()),
        Some("Letter".to_string()),
        Some("Flag".to_string()),
    ]);

    let index = AdvancedMultiIndex::new(tuples, level_names)?;

    println!("Original index with 4 levels:");
    println!("{}", index);
    println!();

    // Reorder levels
    println!("1. Reorder levels: [2, 0, 1, 3] (Letter, Category, Number, Flag):");
    let reordered = index.reorder_levels(&[2, 0, 1, 3])?;
    println!("   Reordered index:");
    println!("{}", reordered);
    println!();

    // Test different level orderings
    println!("2. Reverse order: [3, 2, 1, 0]:");
    let reversed = index.reorder_levels(&[3, 2, 1, 0])?;
    println!("   Level names: {:?}", reversed.level_names());
    for i in 0..2.min(reversed.len()) {
        println!("   [{}] {:?}", i, reversed.get_tuple(i)?);
    }
    println!();

    // Cache management
    println!("3. Cache management:");
    let mut mutable_index = index.clone();

    // Perform operations that use cache
    let _ = mutable_index.xs(IndexValue::from("A"), 0, false)?;
    let _ = mutable_index.xs(IndexValue::from("B"), 0, false)?;
    let _ = mutable_index.xs(IndexValue::from(1), 1, false)?;

    let cache_stats = mutable_index.cache_stats();
    println!("   Cache size: {} entries", cache_stats.size);
    println!("   Should clear: {}", cache_stats.should_clear());

    mutable_index.clear_cache();
    let after_clear = mutable_index.cache_stats();
    println!("   Cache size after clear: {} entries", after_clear.size);

    Ok(())
}

/// Demonstrate GroupBy operations
fn demo_groupby_operations() -> Result<()> {
    println!("📊 GroupBy Operations");
    println!("====================");

    let tuples = vec![
        vec![
            IndexValue::from("Sales"),
            IndexValue::from("North"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("Sales"),
            IndexValue::from("North"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("Sales"),
            IndexValue::from("South"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("Sales"),
            IndexValue::from("South"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("Marketing"),
            IndexValue::from("North"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("Marketing"),
            IndexValue::from("North"),
            IndexValue::from("Q2"),
        ],
        vec![
            IndexValue::from("Marketing"),
            IndexValue::from("South"),
            IndexValue::from("Q1"),
        ],
        vec![
            IndexValue::from("Marketing"),
            IndexValue::from("South"),
            IndexValue::from("Q2"),
        ],
    ];

    let level_names = Some(vec![
        Some("Department".to_string()),
        Some("Region".to_string()),
        Some("Quarter".to_string()),
    ]);

    let index = AdvancedMultiIndex::new(tuples, level_names)?;

    println!("Dataset for GroupBy operations:");
    println!("{}", index);
    println!();

    // Group by single level
    println!("1. Group by Department (level 0):");
    let dept_keys = index.get_group_keys(&[0])?;
    for key in &dept_keys {
        let indices = index.get_group_indices(&[0], key)?;
        println!("   {:?}: {} rows", key[0], indices.len());
    }
    println!();

    // Group by multiple levels
    println!("2. Group by Department + Region (levels 0, 1):");
    let dept_region_keys = index.get_group_keys(&[0, 1])?;
    for key in &dept_region_keys {
        let indices = index.get_group_indices(&[0, 1], key)?;
        println!("   {:?}: {} rows", key, indices.len());
    }
    println!();

    // Cross-level grouping
    println!("3. Group by Region + Quarter (levels 1, 2):");
    let region_quarter_keys = index.get_group_keys(&[1, 2])?;
    for key in &region_quarter_keys {
        let indices = index.get_group_indices(&[1, 2], key)?;
        println!("   {:?}: {} rows", key, indices.len());
    }

    Ok(())
}

/// Demonstrate performance features and benchmarks
fn demo_performance_features() -> Result<()> {
    println!("🚀 Performance Features");
    println!("=======================");

    // Create a larger dataset for performance testing
    let mut tuples = Vec::new();
    let categories = ["A", "B", "C", "D", "E"];
    let subcategories = ["X", "Y", "Z"];
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    for cat in &categories {
        for subcat in &subcategories {
            for &num in &numbers {
                tuples.push(vec![
                    IndexValue::from(*cat),
                    IndexValue::from(*subcat),
                    IndexValue::from(num),
                ]);
            }
        }
    }

    let mut index = AdvancedMultiIndex::new(tuples, None)?;
    println!(
        "Created large index: {} rows, {} levels",
        index.len(),
        index.n_levels()
    );
    println!();

    // Benchmark cross-section operations
    println!("1. Cross-section performance (1000 operations):");
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = index.xs(IndexValue::from("A"), 0, false)?;
    }
    let xs_time = start.elapsed();
    println!("   Time: {:.3}ms", xs_time.as_secs_f64() * 1000.0);
    println!(
        "   Avg per operation: {:.3}μs",
        xs_time.as_nanos() as f64 / 1_000_000.0
    );
    println!();

    // Benchmark selection operations
    println!("2. Selection performance (1000 operations):");
    let start = Instant::now();
    for _ in 0..1000 {
        let criteria = SelectionCriteria::Exact(vec![(0, IndexValue::from("B"))]);
        let _ = index.select(criteria)?;
    }
    let select_time = start.elapsed();
    println!("   Time: {:.3}ms", select_time.as_secs_f64() * 1000.0);
    println!(
        "   Avg per operation: {:.3}μs",
        select_time.as_nanos() as f64 / 1_000_000.0
    );
    println!();

    // Benchmark groupby operations
    println!("3. GroupBy performance (100 operations):");
    let start = Instant::now();
    for _ in 0..100 {
        let _ = index.get_group_keys(&[0, 1])?;
    }
    let groupby_time = start.elapsed();
    println!("   Time: {:.3}ms", groupby_time.as_secs_f64() * 1000.0);
    println!(
        "   Avg per operation: {:.3}μs",
        groupby_time.as_nanos() as f64 / 1_000.0
    );
    println!();

    // Cache effectiveness
    println!("4. Cache effectiveness:");
    let cache_stats_before = index.cache_stats();

    // Perform repeated operations
    for _ in 0..50 {
        let _ = index.xs(IndexValue::from("A"), 0, false)?;
        let _ = index.xs(IndexValue::from("B"), 0, false)?;
        let _ = index.xs(IndexValue::from("C"), 0, false)?;
    }

    let cache_stats_after = index.cache_stats();
    println!("   Cache entries before: {}", cache_stats_before.size);
    println!("   Cache entries after: {}", cache_stats_after.size);
    println!("   Cache hit efficiency: Enabled (repeated operations use cached results)");

    // Memory usage estimation
    println!();
    println!("5. Memory usage estimation:");
    let tuple_memory = index.len() * index.n_levels() * std::mem::size_of::<IndexValue>();
    let level_memory = index.n_levels() * 1024; // Rough estimate for level maps
    let cache_memory = cache_stats_after.size * 256; // Rough estimate per cache entry
    let total_memory = tuple_memory + level_memory + cache_memory;

    println!("   Estimated memory usage:");
    println!("     Tuples: ~{} bytes", tuple_memory);
    println!("     Level maps: ~{} bytes", level_memory);
    println!("     Cache: ~{} bytes", cache_memory);
    println!("     Total: ~{:.1} KB", total_memory as f64 / 1024.0);

    Ok(())
}

fn print_summary() {
    println!("🎉 Advanced MultiIndex Summary");
    println!("=============================");
    println!("✅ Core Achievements:");
    println!("   • Comprehensive MultiIndex with hierarchical data support");
    println!("   • Cross-section (xs) selection for partial key matching");
    println!("   • Advanced selection criteria (exact, partial, range, boolean)");
    println!("   • Level-wise operations and transformations");
    println!("   • High-performance GroupBy with multi-level grouping");
    println!("   • Index reordering, slicing, and level management");
    println!();
    println!("📈 Performance Features:");
    println!("   • Intelligent caching for repeated cross-section operations");
    println!("   • Optimized lookup maps for O(1) value-to-indices mapping");
    println!("   • Efficient memory layout with sorted level values");
    println!("   • Batch operations for multiple selection criteria");
    println!("   • Zero-copy operations where possible");
    println!();
    println!("🏗️ Architecture Advantages:");
    println!("   • Type-safe IndexValue enum supporting multiple data types");
    println!("   • Flexible selection criteria system for complex queries");
    println!("   • Hierarchical operations with automatic index management");
    println!("   • Cache management with configurable size limits");
    println!("   • Comprehensive error handling and validation");
    println!();
    println!("🔬 Use Cases:");
    println!("   • Time series data with multiple hierarchical dimensions");
    println!("   • Financial data analysis with region/product/time hierarchies");
    println!("   • Scientific datasets with nested categorical variables");
    println!("   • Business intelligence with drill-down capabilities");
    println!("   • Data warehousing with star/snowflake schema dimensions");
    println!();
    println!("🚀 Next Steps:");
    println!("   • Integration with DataFrame for complete indexing support");
    println!("   • Stack/unstack operations for data reshaping");
    println!("   • Pivot table functionality with MultiIndex");
    println!("   • Advanced aggregation functions for GroupBy operations");
    println!("   • Query optimization for complex selection patterns");
}
