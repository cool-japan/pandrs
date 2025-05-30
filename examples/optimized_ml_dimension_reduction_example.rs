//! Optimized Dimension Reduction Example
//!
//! This example demonstrates how to use PandRS's dimension reduction algorithms (PCA, t-SNE)
//! with the optimized DataFrame implementation and GPU acceleration. It shows how to create
//! a synthetic dataset, perform dimension reduction, and analyze the results.
//!
//! To run this example:
//!   cargo run --example optimized_ml_dimension_reduction_example --features "optimized cuda"
//!

#[cfg(all(feature = "optimized", feature = "cuda"))]
use pandrs::column::{Column, ColumnTrait, Float64Column, StringColumn};
#[cfg(all(feature = "optimized", feature = "cuda"))]
use pandrs::ml::dimension::{TSNEInit, PCA, TSNE};
#[cfg(all(feature = "optimized", feature = "cuda"))]
use pandrs::ml::{Transformer, UnsupervisedModel};
#[cfg(all(feature = "optimized", feature = "cuda"))]
use pandrs::{error::Error, OptimizedDataFrame};
#[cfg(all(feature = "optimized", feature = "cuda"))]
use rand::Rng;

#[cfg(all(feature = "optimized", feature = "cuda"))]
fn main() -> Result<(), Error> {
    println!("Example of PandRS Dimension Reduction Algorithms");
    println!("============================");

    // Generate sample data
    let df = create_sample_data()?;
    println!(
        "Original DataFrame: {:?} rows x {:?} columns",
        df.row_count(),
        df.column_names().len()
    );

    // Example of PCA
    pca_example(&df)?;

    // Example of t-SNE
    tsne_example(&df)?;

    Ok(())
}

#[cfg(all(feature = "optimized", feature = "cuda"))]
// Example of PCA
fn pca_example(df: &OptimizedDataFrame) -> Result<(), Error> {
    println!("\n==== PCA (Principal Component Analysis) ====");

    // Create PCA instance (reduce to 2 dimensions)
    let mut pca = PCA::new(2, true);

    // Execute PCA
    let pca_result = pca.fit_transform(df)?;

    println!(
        "\nDataFrame after PCA: {:?} rows x {:?} columns",
        pca_result.row_count(),
        pca_result.column_names().len()
    );

    // Display values of Principal Component 1 and Principal Component 2
    if let Ok(pc1_view) = pca_result.column("PC1") {
        if let Some(pc1) = pc1_view.as_float64() {
            // Manually calculate mean and standard deviation
            let values: Vec<f64> = (0..pc1.len())
                .filter_map(|i| pc1.get(i).ok().flatten())
                .collect();

            let mean = if !values.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };

            let std_dev = if values.len() > 1 {
                let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };

            println!("\nMean of Principal Component 1: {:.4}", mean);
            println!(
                "Standard Deviation of Principal Component 1: {:.4}",
                std_dev
            );
        }
    }

    if let Ok(pc2_view) = pca_result.column("PC2") {
        if let Some(pc2) = pc2_view.as_float64() {
            // Manually calculate mean and standard deviation
            let values: Vec<f64> = (0..pc2.len())
                .filter_map(|i| pc2.get(i).ok().flatten())
                .collect();

            let mean = if !values.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };

            let std_dev = if values.len() > 1 {
                let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };

            println!("Mean of Principal Component 2: {:.4}", mean);
            println!(
                "Standard Deviation of Principal Component 2: {:.4}",
                std_dev
            );
        }
    }

    // Display explained variance ratio
    println!("\nExplained Variance Ratio:");
    let explained_variance = pca.explained_variance;
    let total_variance = explained_variance.iter().sum::<f64>();

    // Calculate ratios ourselves
    let mut cumulative = 0.0;
    let mut cumulative_ratios = Vec::new();

    for (i, &variance) in explained_variance.iter().enumerate() {
        let ratio = variance / total_variance;
        cumulative += ratio;
        cumulative_ratios.push(cumulative);
        println!("PC{}: {:.4} ({:.2}%)", i + 1, ratio, ratio * 100.0);
    }

    println!("\nCumulative Explained Variance:");
    for (i, ratio) in cumulative_ratios.iter().enumerate() {
        println!("PC1-PC{}: {:.4} ({:.2}%)", i + 1, ratio, ratio * 100.0);
    }

    // Calculate mean of principal component values for each cluster
    calculate_cluster_means(&pca_result, "PC1", "PC2")?;

    // PCA to 3 dimensions (higher dimensions)
    println!("\n==== 3D PCA ====");
    let mut pca3d = PCA::new(3, true);
    let pca3d_result = pca3d.fit_transform(df)?;

    println!(
        "\nDataFrame after 3D PCA: {:?} rows x {:?} columns",
        pca3d_result.row_count(),
        pca3d_result.column_names().len()
    );

    // Display explained variance ratio
    println!("\nExplained Variance Ratio of 3D PCA:");
    let explained_variance = pca3d.explained_variance;
    let total_variance = explained_variance.iter().sum::<f64>();

    // Calculate ratios ourselves
    let mut cumulative = 0.0;
    let mut cumulative_ratios = Vec::new();

    for (i, &variance) in explained_variance.iter().enumerate() {
        let ratio = variance / total_variance;
        cumulative += ratio;
        cumulative_ratios.push(cumulative);
        println!("PC{}: {:.4} ({:.2}%)", i + 1, ratio, ratio * 100.0);
    }

    // Display cumulative variance
    let last_cumulative = cumulative_ratios.last().unwrap_or(&0.0);
    println!(
        "Cumulative Explained Variance: {:.4} ({:.2}%)",
        last_cumulative,
        last_cumulative * 100.0
    );

    Ok(())
}

#[cfg(all(feature = "optimized", feature = "cuda"))]
// Example of t-SNE
fn tsne_example(df: &OptimizedDataFrame) -> Result<(), Error> {
    println!("\n==== t-SNE (t-distributed Stochastic Neighbor Embedding) ====");

    // Create t-SNE instance (reduce to 2 dimensions)
    let mut tsne = TSNE::new();
    // Configure the TSNE instance
    tsne.n_components = 2;
    tsne.perplexity = 30.0;
    tsne.learning_rate = 200.0;
    tsne.max_iter = 100;
    tsne.init = TSNEInit::PCA;

    println!("\nRunning t-SNE (this may take some time)...");

    // Execute t-SNE
    let tsne_result = tsne.fit_transform(df)?;

    println!(
        "DataFrame after t-SNE: {:?} rows x {:?} columns",
        tsne_result.row_count(),
        tsne_result.column_names().len()
    );

    // Calculate mean of embedding coordinates for each cluster
    calculate_cluster_means(&tsne_result, "TSNE1", "TSNE2")?;

    Ok(())
}

#[cfg(all(feature = "optimized", feature = "cuda"))]
// Calculate and display mean coordinates for each cluster
fn calculate_cluster_means(df: &OptimizedDataFrame, x_col: &str, y_col: &str) -> Result<(), Error> {
    // Do nothing if there is no cluster column
    if !df.column_names().contains(&"cluster".to_string()) {
        return Ok(());
    }

    println!("\nMean Coordinates for Each Cluster:");

    let cluster_view = df.column("cluster")?;
    let x_view = df.column(x_col)?;
    let y_view = df.column(y_col)?;

    if let Some(cluster_col) = cluster_view.as_string() {
        if let (Some(x_col), Some(y_col)) = (x_view.as_float64(), y_view.as_float64()) {
            // Get cluster types
            let mut clusters = Vec::new();
            for i in 0..df.row_count() {
                if let Ok(Some(cluster)) = cluster_col.get(i) {
                    let cluster_str = cluster.to_string();
                    if !clusters.contains(&cluster_str) {
                        clusters.push(cluster_str);
                    }
                }
            }

            // Calculate sum and count for each cluster
            let mut cluster_sums = std::collections::HashMap::new();

            for i in 0..df.row_count() {
                if let (Ok(Some(cluster)), Ok(Some(x)), Ok(Some(y))) =
                    (cluster_col.get(i), x_col.get(i), y_col.get(i))
                {
                    let entry = cluster_sums
                        .entry(cluster.to_string())
                        .or_insert((0.0, 0.0, 0));
                    entry.0 += x;
                    entry.1 += y;
                    entry.2 += 1;
                }
            }

            // Calculate and display mean for each cluster
            for cluster in clusters {
                if let Some(&(x_sum, y_sum, count)) = cluster_sums.get(&cluster) {
                    let x_mean = x_sum / count as f64;
                    let y_mean = y_sum / count as f64;
                    println!(
                        "Cluster {}: ({:.4}, {:.4}), {} samples",
                        cluster, x_mean, y_mean, count
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(all(feature = "optimized", feature = "cuda"))]
// Generate sample data (high-dimensional data with 3 clusters)
fn create_sample_data() -> Result<OptimizedDataFrame, Error> {
    let mut rng = rand::thread_rng();
    let mut df = OptimizedDataFrame::new();

    // Generate data for 300 samples
    let n_samples = 300;
    let n_features = 10; // 10-dimensional data

    // Define clusters
    let clusters = ["A", "B", "C"];
    let centers = [
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Center of Cluster A
        vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], // Center of Cluster B
        vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0], // Center of Cluster C
    ];

    // Generate values for each feature
    for j in 0..n_features {
        let mut feature_values = Vec::with_capacity(n_samples);

        // Data for Cluster A (center [0, 0, ...], small variance)
        for _ in 0..n_samples / 3 {
            feature_values.push(centers[0][j] + rng.random_range(-1.0..1.0));
        }

        // Data for Cluster B (center [5, 5, ...], large variance)
        for _ in 0..n_samples / 3 {
            feature_values.push(centers[1][j] + rng.random_range(-2.0..2.0));
        }

        // Data for Cluster C (center [-5, -5, ...], medium variance)
        for _ in 0..n_samples / 3 {
            feature_values.push(centers[2][j] + rng.random_range(-1.5..1.5));
        }

        // Add feature column to DataFrame
        let column = Float64Column::with_name(feature_values, format!("feature{}", j + 1));
        df.add_column(format!("feature{}", j + 1), Column::Float64(column))?;
    }

    // Add cluster labels
    let mut cluster_labels = Vec::with_capacity(n_samples);

    for cluster_idx in 0..3 {
        for _ in 0..n_samples / 3 {
            cluster_labels.push(clusters[cluster_idx].to_string());
        }
    }

    let cluster_column = StringColumn::with_name(cluster_labels, "cluster");
    df.add_column("cluster", Column::String(cluster_column))?;

    Ok(df)
}

#[cfg(not(all(feature = "optimized", feature = "cuda")))]
fn main() {
    println!("This example requires both \"optimized\" and \"cuda\" feature flags to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_ml_dimension_reduction_example --features \"optimized cuda\"");
}
