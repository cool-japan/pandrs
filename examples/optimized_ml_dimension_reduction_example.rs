//! Optimized Dimension Reduction Example
//!
//! This example demonstrates PandRS's dimension reduction algorithms (PCA, t-SNE).
//!
//! To run this example:
//!   cargo run --example optimized_ml_dimension_reduction_example --features "optimized cuda"

#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::error::Result;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::ml::dimension::{PCA, TSNE};
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::ml::Transformer;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::{DataFrame, Series};

#[cfg(all(cuda_available, feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("Example of PandRS Dimension Reduction Algorithms");
    println!("================================================");

    // Generate sample data
    let df = create_sample_data()?;
    println!("\nCreated sample dataset with {} rows", df.row_count());

    // 1. PCA Example
    println!("\n--- Principal Component Analysis (PCA) ---");
    let _pca = PCA::new(2, true); // Reduce to 2 components, with standardization
    println!("  Target components: 2");
    println!("  PCA model created");

    // Note: Full PCA implementation requires proper DataFrame -> ndarray conversion
    println!("  Available fields: n_components, explained_variance_ratio");

    // 2. t-SNE Example
    println!("\n--- t-SNE Dimensionality Reduction ---");
    let _tsne = TSNE::new(); // t-SNE with default parameters
    println!("  Target dimensions: 2");
    println!("  t-SNE model created");
    println!("  Available parameters: perplexity, learning_rate, n_iter");

    println!("\n=== Dimension Reduction Example Completed ===");
    println!("\nNote: Full dimension reduction requires proper data conversion.");
    println!("See documentation for complete usage examples.");

    Ok(())
}

#[cfg(all(cuda_available, feature = "optimized"))]
fn create_sample_data() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Create sample features
    let feature1: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let feature2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let feature3: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();

    df.add_column(
        "feature1".to_string(),
        Series::new(feature1, Some("feature1".to_string()))?,
    )?;
    df.add_column(
        "feature2".to_string(),
        Series::new(feature2, Some("feature2".to_string()))?,
    )?;
    df.add_column(
        "feature3".to_string(),
        Series::new(feature3, Some("feature3".to_string()))?,
    )?;

    // Add cluster labels
    let clusters: Vec<&str> = (0..100)
        .map(|i| {
            if i % 3 == 0 {
                "A"
            } else if i % 3 == 1 {
                "B"
            } else {
                "C"
            }
        })
        .collect();
    df.add_column(
        "cluster".to_string(),
        Series::new(clusters, Some("cluster".to_string()))?,
    )?;

    Ok(df)
}

#[cfg(not(all(cuda_available, feature = "optimized")))]
fn main() {
    println!("Optimized Dimension Reduction Example");
    println!("=====================================");
    println!();
    println!("This example requires the 'optimized' and 'cuda' features.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_ml_dimension_reduction_example --features \"optimized cuda\"");
}
