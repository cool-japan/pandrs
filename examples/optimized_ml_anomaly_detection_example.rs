#[cfg(feature = "optimized")]
use pandrs::column::{Column, Float64Column, Int64Column};
#[cfg(feature = "optimized")]
use pandrs::ml::anomaly::{IsolationForest, LocalOutlierFactor, OneClassSVM};
#[cfg(feature = "optimized")]
use pandrs::ml::UnsupervisedModel;
#[cfg(feature = "optimized")]
use pandrs::optimized::convert;
#[cfg(feature = "optimized")]
use pandrs::optimized::OptimizedDataFrame;
#[cfg(feature = "optimized")]
use rand::rngs::StdRng;
#[cfg(feature = "optimized")]
use rand::Rng;
#[cfg(feature = "optimized")]
use rand::SeedableRng;

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!(
        "  cargo run --example optimized_ml_anomaly_detection_example --features \"optimized\""
    );
}

#[cfg(feature = "optimized")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic anomaly detection data
    println!("✅ Example of Anomaly Detection Algorithms");
    println!("==========================");
    println!("1. Generating synthetic data");

    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 1000;

    // Generate normal data (2D data, normal distribution centered at [0, 0])
    let n_normal = 950;
    let mut x_values = Vec::with_capacity(n_samples);
    let mut y_values = Vec::with_capacity(n_samples);
    let mut true_labels = Vec::with_capacity(n_samples);

    for _ in 0..n_normal {
        x_values.push(rng.random_range(-3.0..3.0));
        y_values.push(rng.random_range(-3.0..3.0));
        true_labels.push(0); // Normal data is labeled as 0
    }

    // Generate anomalies (data far from normal data)
    let n_anomalies = n_samples - n_normal;

    for _ in 0..n_anomalies {
        // Randomly generate the position of outliers
        match rng.random_range(0..4) {
            0 => {
                // Top left
                x_values.push(rng.random_range(-10.0..-5.0));
                y_values.push(rng.random_range(5.0..10.0));
            }
            1 => {
                // Top right
                x_values.push(rng.random_range(5.0..10.0));
                y_values.push(rng.random_range(5.0..10.0));
            }
            2 => {
                // Bottom left
                x_values.push(rng.random_range(-10.0..-5.0));
                y_values.push(rng.random_range(-10.0..-5.0));
            }
            _ => {
                // Bottom right
                x_values.push(rng.random_range(5.0..10.0));
                y_values.push(rng.random_range(-10.0..-5.0));
            }
        }
        true_labels.push(1); // Anomalies are labeled as 1
    }

    // Create DataFrame
    let mut df = OptimizedDataFrame::new();

    // Use clone() to create copies of values
    let x_col = Column::Float64(Float64Column::with_name(x_values.clone(), "x"));
    let y_col = Column::Float64(Float64Column::with_name(y_values.clone(), "y"));
    let true_labels_col =
        Column::Int64(Int64Column::with_name(true_labels.clone(), "true_anomaly"));

    df.add_column("x".to_string(), x_col)?;
    df.add_column("y".to_string(), y_col)?;
    df.add_column("true_anomaly".to_string(), true_labels_col)?;

    println!(
        "Data generation complete: {} normal samples, {} anomaly samples",
        n_normal, n_anomalies
    );
    println!("First few rows of the DataFrame:");
    // Display the first 5 rows instead of using df.head()
    println!("DataFrame (first 5 rows):");
    for i in 0..std::cmp::min(5, df.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(anomaly))) = (
            df.column("x").unwrap().as_float64().unwrap().get(i),
            df.column("y").unwrap().as_float64().unwrap().get(i),
            df.column("true_anomaly")
                .unwrap()
                .as_int64()
                .unwrap()
                .get(i),
        ) {
            println!("Row {}: x={:.4}, y={:.4}, anomaly={}", i, x, y, anomaly);
        }
    }

    // Anomaly detection using IsolationForest
    println!("\n2. Anomaly detection using Isolation Forest");
    let mut isolation_forest = IsolationForest::new()
        .n_estimators(100)       // Number of trees
        .contamination(0.05)     // Contamination rate 5%
        .random_seed(42); // Random seed

    // Convert to standard DataFrame for the algorithm
    let reg_df = convert::standard_dataframe(&df)?;
    let if_result = isolation_forest.fit_transform(&reg_df)?;
    // Convert back to OptimizedDataFrame for consistent handling
    let if_result_opt = convert::optimize_dataframe(&if_result)?;

    println!("Isolation Forest detection complete");
    println!(
        "Number of detected anomalies: {}",
        isolation_forest
            .labels()
            .iter()
            .filter(|&&x| x == 1)
            .count()
    );
    println!("First few rows of the result:");
    // Display the first 5 rows instead of using if_result.head()
    println!("Isolation Forest result (first 5 rows):");
    for i in 0..std::cmp::min(5, if_result_opt.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(anomaly_score)), Ok(Some(anomaly))) = (
            if_result_opt
                .column("x")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            if_result_opt
                .column("y")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            if_result_opt
                .column("anomaly_score")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            if_result_opt
                .column("anomaly")
                .unwrap()
                .as_int64()
                .unwrap()
                .get(i),
        ) {
            println!(
                "Row {}: x={:.4}, y={:.4}, score={:.4}, anomaly={}",
                i, x, y, anomaly_score, anomaly
            );
        }
    }

    // Anomaly detection using LOF
    println!("\n3. Anomaly detection using Local Outlier Factor");
    let mut lof = LocalOutlierFactor::new(20) // Number of neighbors
        .contamination(0.05); // Contamination rate 5%

    // LocalOutlierFactor doesn't currently support transform in this implementation,
    // so we need to use fit() and then duplicate reg_df to create our result
    lof.fit(&reg_df)?;
    let lof_result = reg_df.clone();
    // Convert back to OptimizedDataFrame for consistent handling
    let lof_result_opt = convert::optimize_dataframe(&lof_result)?;

    println!("Local Outlier Factor detection complete");
    println!(
        "Number of detected anomalies: {}",
        lof.labels().iter().filter(|&&x| x == 1).count()
    );
    println!("First few rows of the result:");
    // Display the first 5 rows instead of using lof_result.head()
    println!("Local Outlier Factor result (first 5 rows):");
    for i in 0..std::cmp::min(5, lof_result_opt.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(lof_score)), Ok(Some(anomaly))) = (
            lof_result_opt
                .column("x")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            lof_result_opt
                .column("y")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            lof_result_opt
                .column("lof_score")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            lof_result_opt
                .column("anomaly")
                .unwrap()
                .as_int64()
                .unwrap()
                .get(i),
        ) {
            println!(
                "Row {}: x={:.4}, y={:.4}, score={:.4}, anomaly={}",
                i, x, y, lof_score, anomaly
            );
        }
    }

    // Anomaly detection using One-Class SVM
    println!("\n4. Anomaly detection using One-Class SVM");
    let mut one_class_svm = OneClassSVM::new()
        .nu(0.05)                // nu parameter
        .gamma(0.1); // gamma parameter

    let svm_result = one_class_svm.fit_transform(&reg_df)?;
    // Convert back to OptimizedDataFrame for consistent handling
    let svm_result_opt = convert::optimize_dataframe(&svm_result)?;

    println!("One-Class SVM detection complete");
    println!(
        "Number of detected anomalies: {}",
        one_class_svm.labels().iter().filter(|&&x| x == 1).count()
    );
    println!("First few rows of the result:");
    // Display the first 5 rows instead of using svm_result.head()
    println!("One-Class SVM result (first 5 rows):");
    for i in 0..std::cmp::min(5, svm_result_opt.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(decision_value)), Ok(Some(anomaly))) = (
            svm_result_opt
                .column("x")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            svm_result_opt
                .column("y")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            svm_result_opt
                .column("decision_value")
                .unwrap()
                .as_float64()
                .unwrap()
                .get(i),
            svm_result_opt
                .column("anomaly")
                .unwrap()
                .as_int64()
                .unwrap()
                .get(i),
        ) {
            println!(
                "Row {}: x={:.4}, y={:.4}, decision={:.4}, anomaly={}",
                i, x, y, decision_value, anomaly
            );
        }
    }

    // Compare anomaly flags of each algorithm
    println!("\n5. Comparing detection results");

    // Number of samples flagged as anomalies
    let if_anomalies = isolation_forest
        .labels()
        .iter()
        .filter(|&&x| x == 1)
        .count();
    let lof_anomalies = lof.labels().iter().filter(|&&x| x == 1).count();
    let svm_anomalies = one_class_svm.labels().iter().filter(|&&x| x == 1).count();

    println!("Isolation Forest: detected {} anomalies", if_anomalies);
    println!("Local Outlier Factor: detected {} anomalies", lof_anomalies);
    println!("One-Class SVM: detected {} anomalies", svm_anomalies);

    // Check agreement between algorithms
    let mut all_agree = 0;
    let mut if_lof_agree = 0;
    let mut if_svm_agree = 0;
    let mut lof_svm_agree = 0;

    for i in 0..n_samples {
        let if_label = isolation_forest.labels()[i];
        let lof_label = lof.labels()[i];
        let svm_label = one_class_svm.labels()[i];

        if if_label == lof_label && lof_label == svm_label {
            all_agree += 1;

            // Display the first few "all algorithms agree" anomalies
            if if_label == 1 && all_agree <= 5 {
                // Display only the first 5
                // Instead of accessing values, get values from the dataframe
                if let (Ok(Some(x)), Ok(Some(y))) = (
                    df.column("x").unwrap().as_float64().unwrap().get(i),
                    df.column("y").unwrap().as_float64().unwrap().get(i),
                ) {
                    println!(
                        "Sample {} is detected as an anomaly by all algorithms: x={:.2}, y={:.2}",
                        i, x, y
                    );
                }
            }
        }

        if if_label == lof_label {
            if_lof_agree += 1;
        }
        if if_label == svm_label {
            if_svm_agree += 1;
        }
        if lof_label == svm_label {
            lof_svm_agree += 1;
        }
    }

    println!(
        "Agreement rate of all algorithms: {:.1}%",
        100.0 * all_agree as f64 / n_samples as f64
    );
    println!(
        "Agreement rate of Isolation Forest and LOF: {:.1}%",
        100.0 * if_lof_agree as f64 / n_samples as f64
    );
    println!(
        "Agreement rate of Isolation Forest and SVM: {:.1}%",
        100.0 * if_svm_agree as f64 / n_samples as f64
    );
    println!(
        "Agreement rate of LOF and SVM: {:.1}%",
        100.0 * lof_svm_agree as f64 / n_samples as f64
    );

    // Compare with true anomalies
    println!("\n6. Comparison with true anomalies");

    // Evaluation metric function
    let calc_metrics = |algorithm_name: &str, labels: &[i64], true_labels: &[i64]| {
        let mut tp = 0; // True positive
        let mut fp = 0; // False positive
        let mut tn = 0; // True negative
        let mut fn_count = 0; // False negative

        for i in 0..labels.len() {
            let pred = labels[i];
            let true_val = true_labels[i];

            match (pred, true_val) {
                (1, 1) => tp += 1,        // True positive
                (1, 0) => fp += 1,        // False positive
                (-1, 0) => tn += 1,       // True negative
                (-1, 1) => fn_count += 1, // False negative
                _ => {}
            }
        }

        // Calculate precision, recall, and F1 score
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let accuracy = (tp + tn) as f64 / labels.len() as f64;

        println!(
            "{}: Precision={:.1}%, Recall={:.1}%, F1={:.3}, Accuracy={:.1}%",
            algorithm_name,
            precision * 100.0,
            recall * 100.0,
            f1,
            accuracy * 100.0
        );
    };

    // Get true labels from the dataframe
    let extracted_true_labels: Vec<i64> = (0..df.row_count())
        .filter_map(|i| {
            df.column("true_anomaly")
                .unwrap()
                .as_int64()
                .unwrap()
                .get(i)
                .ok()
                .flatten()
        })
        .collect();

    // Calculate evaluation metrics for each algorithm
    calc_metrics(
        "Isolation Forest",
        isolation_forest.labels(),
        &extracted_true_labels,
    );
    calc_metrics("Local Outlier Factor", lof.labels(), &extracted_true_labels);
    calc_metrics(
        "One-Class SVM",
        one_class_svm.labels(),
        &extracted_true_labels,
    );

    println!("\n==========================");
    println!("✅ Anomaly detection example completed successfully");

    Ok(())
}
