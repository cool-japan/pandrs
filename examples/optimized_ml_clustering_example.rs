extern crate pandrs;

use pandrs::optimized::OptimizedDataFrame;
use pandrs::ml::clustering::{KMeans, DBSCAN, AgglomerativeClustering, DistanceMetric, Linkage};
use pandrs::column::{Float64Column, Column};
use pandrs::ml::pipeline::Transformer;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate pseudo clustering data
    println!("✅ Example of Clustering Algorithms");
    println!("=================================");
    println!("1. Generating pseudo data");
    
    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 1000;
    let n_clusters = 3;
    
    // Generate data from different centers for each cluster
    let cluster_centers = vec![
        (0.0, 0.0),       // Center of cluster 1
        (5.0, 5.0),       // Center of cluster 2
        (-5.0, 5.0),     // Center of cluster 3
    ];
    
    let mut x_values = Vec::with_capacity(n_samples);
    let mut y_values = Vec::with_capacity(n_samples);
    let mut true_labels = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        // Assign each sample to a cluster evenly
        let cluster_idx = i % n_clusters;
        let (center_x, center_y) = cluster_centers[cluster_idx];
        
        // Generate variation from the cluster center (normal distribution)
        let x = center_x + rng.random_range(-1.5..1.5);
        let y = center_y + rng.random_range(-1.5..1.5);
        
        x_values.push(x);
        y_values.push(y);
        true_labels.push(cluster_idx as i64);
    }
    
    // Create DataFrame
    let mut df = OptimizedDataFrame::new();
    
    let x_col = Column::Float64(Float64Column::with_name(x_values, "x"));
    let y_col = Column::Float64(Float64Column::with_name(y_values, "y"));
    let true_labels_col = Column::Int64(pandrs::column::Int64Column::with_name(true_labels, "true_cluster"));
    
    df.add_column("x".to_string(), x_col)?;
    df.add_column("y".to_string(), y_col)?;
    df.add_column("true_cluster".to_string(), true_labels_col)?;
    
    println!("Data generation complete: {} samples, {} clusters", n_samples, n_clusters);
    println!("First few rows of the DataFrame:");
    // Display the first 5 rows instead of df.head()
    println!("DataFrame (first 5 rows):");
    for i in 0..std::cmp::min(5, df.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            df.column("x").unwrap().as_float64().unwrap().get(i),
            df.column("y").unwrap().as_float64().unwrap().get(i),
            df.column("true_cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("Row {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // K-means clustering
    println!("\n2. k-means clustering");
    let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
    let kmeans_result = kmeans.fit_transform(&df)?;
    
    println!("k-means clustering complete");
    println!("Cluster centers: {:?}", kmeans.centroids());
    println!("Inertia: {:.4}", kmeans.inertia());
    println!("Number of iterations: {}", kmeans.n_iter());
    println!("First few rows of the result:");
    // Display the first 5 rows instead of kmeans_result.head()
    println!("k-means result (first 5 rows):");
    for i in 0..std::cmp::min(5, kmeans_result.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            kmeans_result.column("x").unwrap().as_float64().unwrap().get(i),
            kmeans_result.column("y").unwrap().as_float64().unwrap().get(i),
            kmeans_result.column("cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("Row {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // Display the distribution of clusters
    let cluster_counts = count_clusters(&kmeans_result, "cluster")?;
    println!("Distribution of clusters:");
    for (cluster, count) in cluster_counts {
        println!("Number of samples in cluster {}: {}", cluster, count);
    }
    
    // DBSCAN clustering
    println!("\n3. DBSCAN clustering");
    let mut dbscan = DBSCAN::new(1.0, 5, DistanceMetric::Euclidean);
    let dbscan_result = dbscan.fit_transform(&df)?;
    
    println!("DBSCAN clustering complete");
    println!("First few rows of the result:");
    // Display the first 5 rows instead of dbscan_result.head()
    println!("DBSCAN result (first 5 rows):");
    for i in 0..std::cmp::min(5, dbscan_result.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            dbscan_result.column("x").unwrap().as_float64().unwrap().get(i),
            dbscan_result.column("y").unwrap().as_float64().unwrap().get(i),
            dbscan_result.column("cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("Row {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // Display the distribution of clusters
    let cluster_counts = count_clusters(&dbscan_result, "cluster")?;
    println!("Distribution of clusters:");
    for (cluster, count) in cluster_counts {
        println!("Number of samples in cluster {}: {}", cluster, count);
    }
    
    // Hierarchical clustering
    println!("\n4. Hierarchical clustering");
    let mut agg_clustering = AgglomerativeClustering::new(3, Linkage::Ward, DistanceMetric::Euclidean);
    let agg_result = agg_clustering.fit_transform(&df)?;
    
    println!("Hierarchical clustering complete");
    println!("First few rows of the result:");
    // Display the first 5 rows instead of agg_result.head()
    println!("Hierarchical clustering result (first 5 rows):");
    for i in 0..std::cmp::min(5, agg_result.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            agg_result.column("x").unwrap().as_float64().unwrap().get(i),
            agg_result.column("y").unwrap().as_float64().unwrap().get(i),
            agg_result.column("cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("Row {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // Display the distribution of clusters
    let cluster_counts = count_clusters(&agg_result, "cluster")?;
    println!("Distribution of clusters:");
    for (cluster, count) in cluster_counts {
        println!("Number of samples in cluster {}: {}", cluster, count);
    }
    
    println!("\n=================================");
    println!("✅ Clustering example completed successfully");
    
    Ok(())
}

// Function to count the number of samples in each cluster
fn count_clusters(df: &OptimizedDataFrame, column: &str) -> Result<Vec<(i64, usize)>, Box<dyn std::error::Error>> {
    let mut counts = std::collections::HashMap::new();
    
    // Count the number of samples in each cluster
    let cluster_col = df.column(column)?;
    
    if let Some(int_col) = cluster_col.as_int64() {
        for i in 0..cluster_col.len() {
            if let Ok(Some(cluster)) = int_col.get(i) {
                *counts.entry(cluster).or_insert(0) += 1;
            }
        }
    }
    
    // Return the sorted result
    let mut result: Vec<(i64, usize)> = counts.into_iter().collect();
    result.sort_by_key(|(cluster, _)| *cluster);
    
    Ok(result)
}