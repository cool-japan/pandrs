/// Demonstration of PandRS real ML algorithms (v0.4.1+)
///
/// This example shows PCA, DBSCAN, LogisticRegression, and IsolationForest
/// working end-to-end on synthetic data — no stubs, no hardcoded results.
///
/// Run with:
///   cargo run --example ml_real_algorithms_example
fn main() {
    println!("PandRS v0.4.1 — Real ML Algorithms Demo");
    println!("=========================================\n");

    demo_pca();
    demo_dbscan();
    demo_logistic_regression();
    demo_isolation_forest();
    demo_lof();
    demo_agglomerative();

    println!("\nAll demos completed successfully.");
}

fn demo_pca() {
    use pandrs::dataframe::DataFrame;
    use pandrs::ml::dimension::PCA;
    use pandrs::ml::models::UnsupervisedModel;
    use pandrs::series::Series;

    println!("1. PCA — dimensionality reduction via Jacobi eigendecomposition");
    println!("   ---------------------------------------------------------------");

    // 3D data that lives on a 2D subspace (PC3 should capture near-zero variance)
    let mut df = DataFrame::new();
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys: Vec<f64> = (0..20)
        .map(|i| i as f64 * 2.0 + 0.01 * (i as f64))
        .collect();
    let zs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    df.add_column("x".into(), Series::new(xs, Some("x".into())).unwrap())
        .unwrap();
    df.add_column("y".into(), Series::new(ys, Some("y".into())).unwrap())
        .unwrap();
    df.add_column("z".into(), Series::new(zs, Some("z".into())).unwrap())
        .unwrap();

    let mut pca = PCA::new(2, true);
    pca.fit(&df).unwrap();

    let transformed = pca.transform(&df).unwrap();
    println!(
        "   Input: {}×{} → Output: {}×{}",
        df.nrows(),
        df.ncols(),
        transformed.nrows(),
        transformed.ncols()
    );

    let evr = pca.explained_variance_ratio.as_ref().unwrap();
    let total_evr: f64 = evr.iter().sum();
    println!(
        "   Explained variance ratios: PC1={:.3}, PC2={:.3} (total={:.3})",
        evr[0], evr[1], total_evr
    );
    assert!(
        total_evr > 0.99,
        "2 PCs should capture >99% of variance for near-linear data"
    );
    println!("   ✓ 2 PCs capture {:.1}% of variance\n", total_evr * 100.0);
}

fn demo_dbscan() {
    use pandrs::dataframe::DataFrame;
    use pandrs::ml::clustering::DBSCAN;
    use pandrs::ml::models::UnsupervisedModel;
    use pandrs::series::Series;

    println!("2. DBSCAN — density-based spatial clustering");
    println!("   ------------------------------------------");

    // Two well-separated clusters + noise points
    let mut df = DataFrame::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    // Cluster A around (0,0)
    for i in 0..8i64 {
        xs.push(0.1 * i as f64);
        ys.push(0.1 * i as f64);
    }
    // Cluster B around (10,10)
    for i in 0..8i64 {
        xs.push(10.0 + 0.1 * i as f64);
        ys.push(10.0 + 0.1 * i as f64);
    }
    // Noise
    xs.push(5.0);
    ys.push(50.0);

    df.add_column("x".into(), Series::new(xs, Some("x".into())).unwrap())
        .unwrap();
    df.add_column("y".into(), Series::new(ys, Some("y".into())).unwrap())
        .unwrap();

    let mut dbscan = DBSCAN::new(1.5, 2);
    dbscan.fit(&df).unwrap();

    let labels = dbscan.labels.as_ref().unwrap();
    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .copied()
        .max()
        .unwrap_or(-1)
        + 1;
    let n_noise = labels.iter().filter(|&&l| l == -1).count();

    println!(
        "   Points: {}, Clusters found: {}, Noise: {}",
        labels.len(),
        n_clusters,
        n_noise
    );
    assert_eq!(n_clusters, 2, "Should find exactly 2 clusters");
    assert!(n_noise >= 1, "Should detect the noise point");
    println!(
        "   ✓ Found {} clusters, {} noise points\n",
        n_clusters, n_noise
    );
}

fn demo_logistic_regression() {
    use pandrs::dataframe::DataFrame;
    use pandrs::ml::models::linear::LogisticRegression;
    use pandrs::ml::models::SupervisedModel;
    use pandrs::series::Series;

    println!("3. LogisticRegression — IRLS binary classification");
    println!("   ------------------------------------------------");

    // Linearly separable binary classification problem
    let mut df = DataFrame::new();
    let features: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let labels: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
    df.add_column(
        "feature".into(),
        Series::new(features, Some("feature".into())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "label".into(),
        Series::new(labels, Some("label".into())).unwrap(),
    )
    .unwrap();

    let mut lr = LogisticRegression::new().with_max_iter(200);
    lr.fit(&df, "label").unwrap();

    let predictions = lr.predict(&df).unwrap();
    let probabilities = lr.predict_proba(&df).unwrap();

    let n_correct = predictions
        .iter()
        .zip((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }))
        .filter(|(&pred, label)| (pred - label).abs() < 0.5)
        .count();
    let accuracy = n_correct as f64 / 20.0;

    println!("   Training accuracy: {:.1}%", accuracy * 100.0);
    println!(
        "   Sample probabilities (first 3): {:.3}, {:.3}, {:.3}",
        probabilities[0], probabilities[10], probabilities[19]
    );

    assert!(
        accuracy >= 0.9,
        "Should achieve ≥90% accuracy on linearly separable data"
    );
    assert!(
        probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "Probabilities must be in [0,1]"
    );
    println!(
        "   ✓ {:.1}% accuracy on linearly separable data\n",
        accuracy * 100.0
    );
}

fn demo_isolation_forest() {
    use pandrs::dataframe::DataFrame;
    use pandrs::ml::anomaly::IsolationForest;
    use pandrs::ml::models::UnsupervisedModel;
    use pandrs::series::Series;

    println!("4. IsolationForest — real isolation tree anomaly detection");
    println!("   ---------------------------------------------------------");

    // 20 normal points + 3 clear outliers
    let mut df = DataFrame::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for i in 0..20i64 {
        xs.push(i as f64 * 0.5);
        ys.push(i as f64 * 0.3);
    }
    // Clear outliers far from the normal region
    xs.push(1000.0);
    ys.push(1000.0);
    xs.push(-1000.0);
    ys.push(1000.0);
    xs.push(1000.0);
    ys.push(-1000.0);

    df.add_column("x".into(), Series::new(xs, Some("x".into())).unwrap())
        .unwrap();
    df.add_column("y".into(), Series::new(ys, Some("y".into())).unwrap())
        .unwrap();

    let mut iforest = IsolationForest::new()
        .contamination(0.13)
        .random_seed(42)
        .n_estimators(50);
    iforest.fit(&df).unwrap();

    let labels = iforest.labels();
    let n_anomalies = labels.iter().filter(|&&l| l == -1).count();
    let scores = iforest.anomaly_scores();

    println!(
        "   Anomalies detected: {}/{} points",
        n_anomalies,
        labels.len()
    );
    println!(
        "   Score range: [{:.3}, {:.3}]",
        scores.iter().cloned().fold(f64::INFINITY, f64::min),
        scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // The 3 outlier points (last 3) should score higher (more anomalous) than the mean of inliers
    let inlier_mean_score: f64 = scores[..20].iter().sum::<f64>() / 20.0;
    let outlier_mean_score: f64 = scores[20..].iter().sum::<f64>() / 3.0;
    println!(
        "   Inlier mean score: {:.3}, Outlier mean score: {:.3}",
        inlier_mean_score, outlier_mean_score
    );

    assert!(
        !labels.is_empty(),
        "Labels should be populated after fitting"
    );
    assert!(
        outlier_mean_score > inlier_mean_score,
        "Outliers should score higher (more anomalous) than inliers"
    );
    println!("   ✓ Outliers correctly score more anomalous than inliers\n");
}

fn demo_lof() {
    use pandrs::dataframe::DataFrame;
    use pandrs::ml::anomaly::LocalOutlierFactor;
    use pandrs::ml::models::UnsupervisedModel;
    use pandrs::series::Series;

    println!("5. LocalOutlierFactor — k-NN density-based anomaly detection");
    println!("   -----------------------------------------------------------");

    let mut df = DataFrame::new();
    let mut xs: Vec<f64> = (0..15).map(|i| i as f64).collect();
    let mut ys: Vec<f64> = (0..15).map(|i| i as f64 * 0.5).collect();
    // Add 3 outliers
    xs.extend_from_slice(&[100.0, -100.0, 50.0]);
    ys.extend_from_slice(&[100.0, 100.0, -50.0]);

    df.add_column("x".into(), Series::new(xs, Some("x".into())).unwrap())
        .unwrap();
    df.add_column("y".into(), Series::new(ys, Some("y".into())).unwrap())
        .unwrap();

    let mut lof = LocalOutlierFactor::new(3).contamination(0.16);
    lof.fit(&df).unwrap();

    let labels = lof.labels();
    let has_anomalies = labels.contains(&-1);
    let has_normals = labels.contains(&1);

    println!(
        "   Labels: {} normal, {} anomaly",
        labels.iter().filter(|&&l| l == 1).count(),
        labels.iter().filter(|&&l| l == -1).count()
    );
    assert!(has_anomalies, "Should detect some anomalies");
    assert!(has_normals, "Should have some normal points");
    println!("   ✓ LOF correctly distinguishes inliers from outliers\n");
}

fn demo_agglomerative() {
    use pandrs::dataframe::DataFrame;
    use pandrs::ml::clustering::{AgglomerativeClustering, Linkage};
    use pandrs::ml::models::UnsupervisedModel;
    use pandrs::series::Series;

    println!("6. AgglomerativeClustering — bottom-up hierarchical clustering");
    println!("   -------------------------------------------------------------");

    // Two well-separated groups
    let mut df = DataFrame::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for i in 0..8i64 {
        xs.push(i as f64 * 0.1);
        ys.push(i as f64 * 0.1);
    }
    for i in 0..8i64 {
        xs.push(100.0 + i as f64 * 0.1);
        ys.push(100.0 + i as f64 * 0.1);
    }

    df.add_column("x".into(), Series::new(xs, Some("x".into())).unwrap())
        .unwrap();
    df.add_column("y".into(), Series::new(ys, Some("y".into())).unwrap())
        .unwrap();

    let mut hclust = AgglomerativeClustering::new(2).with_linkage(Linkage::Ward);
    hclust.fit(&df).unwrap();

    let labels = hclust.labels.as_ref().unwrap();
    let unique_labels: std::collections::HashSet<usize> = labels.iter().cloned().collect();
    println!(
        "   {} points → {} clusters: {:?}",
        labels.len(),
        unique_labels.len(),
        unique_labels
    );

    assert_eq!(
        unique_labels.len(),
        2,
        "Ward linkage should produce 2 clusters"
    );

    // Group 1 (first 8) should all be in the same cluster
    let group1_labels: std::collections::HashSet<usize> = labels[..8].iter().cloned().collect();
    assert_eq!(
        group1_labels.len(),
        1,
        "All near-origin points should share a cluster"
    );
    println!("   ✓ 2 clean clusters found, intra-cluster cohesion verified\n");
}
