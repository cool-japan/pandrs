use pandrs::*;
use pandrs::ml::anomaly_detection::{IsolationForest, LocalOutlierFactor, OneClassSVM};
use pandrs::ml::clustering::DistanceMetric;
use rand::prelude::*;

fn main() -> Result<(), PandRSError> {
    println!("PandRS 異常検出アルゴリズムの例");
    println!("============================");
    
    // サンプルデータの生成
    let df = create_sample_data_with_outliers()?;
    println!("元のデータフレーム（最初の5行）:");
    println!("{}", df.head(5)?);
    
    // Isolation Forest異常検出の例
    isolation_forest_example(&df)?;
    
    // LOF (Local Outlier Factor) 異常検出の例
    lof_example(&df)?;
    
    // One-Class SVM異常検出の例
    one_class_svm_example(&df)?;
    
    // アルゴリズム比較の例
    compare_algorithms(&df)?;
    
    Ok(())
}

// Isolation Forest異常検出の例
fn isolation_forest_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== Isolation Forest 異常検出 ====");
    
    // Isolation Forestインスタンスの作成
    let mut iforest = IsolationForest::new(
        100, // 木の数
        Some(256), // サブサンプリングサイズ
        Some(1.0), // 特徴量のサブサンプリング率
        0.05, // 汚染率（異常値の割合）
        Some(42), // 乱数シード
    );
    
    // 異常検出の実行
    let iforest_result = iforest.fit_transform(df)?;
    
    // 検出された異常の数を表示
    let anomaly_count = iforest_result.column("anomaly").unwrap()
        .iter()
        .filter(|&x| match x {
            DataValue::Int64(v) => *v == 1,
            _ => false,
        })
        .count();
    
    println!("\nIsolation Forest 検出結果:");
    println!("検出された異常: {} / {} サンプル", anomaly_count, df.nrows());
    
    // 異常スコアの分布を表示
    let mut anomaly_scores = iforest.anomaly_scores().to_vec();
    anomaly_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // 降順
    
    println!("\n異常スコア分布 (上位10件):");
    for (i, score) in anomaly_scores.iter().take(10).enumerate() {
        println!("#{}: {:.4}", i + 1, score);
    }
    
    // 実際の異常値（もし存在すれば）との比較
    if df.column("is_outlier").is_some() {
        let accuracy = compare_detection(&iforest_result, "anomaly", "is_outlier")?;
        println!("\n真の異常値との一致率: {:.2}%", accuracy * 100.0);
    }
    
    Ok(())
}

// Local Outlier Factor異常検出の例
fn lof_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== Local Outlier Factor (LOF) 異常検出 ====");
    
    // LOFインスタンスの作成
    let mut lof = LocalOutlierFactor::new(
        20, // 近傍数
        0.05, // 汚染率（異常値の割合）
        DistanceMetric::Euclidean, // 距離メトリック
    );
    
    // 異常検出の実行
    let lof_result = lof.fit_transform(df)?;
    
    // 検出された異常の数を表示
    let anomaly_count = lof_result.column("anomaly").unwrap()
        .iter()
        .filter(|&x| match x {
            DataValue::Int64(v) => *v == 1,
            _ => false,
        })
        .count();
    
    println!("\nLOF 検出結果:");
    println!("検出された異常: {} / {} サンプル", anomaly_count, df.nrows());
    
    // LOFスコアの分布を表示
    let mut lof_scores = lof.lof_scores().to_vec();
    lof_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // 降順
    
    println!("\nLOFスコア分布 (上位10件):");
    for (i, score) in lof_scores.iter().take(10).enumerate() {
        println!("#{}: {:.4}", i + 1, score);
    }
    
    // 実際の異常値（もし存在すれば）との比較
    if df.column("is_outlier").is_some() {
        let accuracy = compare_detection(&lof_result, "anomaly", "is_outlier")?;
        println!("\n真の異常値との一致率: {:.2}%", accuracy * 100.0);
    }
    
    // 異なる近傍数での比較
    println!("\n異なる近傍数での比較:");
    
    let neighbors = vec![5, 10, 20, 30, 50];
    
    for n_neighbors in neighbors {
        let mut model = LocalOutlierFactor::new(n_neighbors, 0.05, DistanceMetric::Euclidean);
        let result = model.fit_transform(df)?;
        let anomaly_count = result.column("anomaly").unwrap()
            .iter()
            .filter(|&x| match x {
                DataValue::Int64(v) => *v == 1,
                _ => false,
            })
            .count();
        
        println!("近傍数 = {} の結果: {} 異常", n_neighbors, anomaly_count);
        
        if df.column("is_outlier").is_some() {
            let accuracy = compare_detection(&result, "anomaly", "is_outlier")?;
            println!("真の異常値との一致率: {:.2}%", accuracy * 100.0);
        }
    }
    
    Ok(())
}

// One-Class SVM異常検出の例
fn one_class_svm_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== One-Class SVM 異常検出 ====");
    
    // One-Class SVMインスタンスの作成
    let mut ocsvm = OneClassSVM::new(
        0.05, // ニュー（異常値の期待割合に相当）
        0.1, // ガンマ（RBFカーネルのパラメータ）
        100, // 最大イテレーション数
        1e-3, // 収束閾値
    );
    
    // 異常検出の実行
    let ocsvm_result = ocsvm.fit_transform(df)?;
    
    // 検出された異常の数を表示
    let anomaly_count = ocsvm_result.column("anomaly").unwrap()
        .iter()
        .filter(|&x| match x {
            DataValue::Int64(v) => *v == 1,
            _ => false,
        })
        .count();
    
    println!("\nOne-Class SVM 検出結果:");
    println!("検出された異常: {} / {} サンプル", anomaly_count, df.nrows());
    
    // 決定スコアの分布を表示
    let mut decision_scores = ocsvm.decision_scores().to_vec();
    decision_scores.sort_by(|a, b| a.partial_cmp(b).unwrap()); // 昇順（負の値が異常）
    
    println!("\n決定スコア分布 (下位10件):");
    for (i, score) in decision_scores.iter().take(10).enumerate() {
        println!("#{}: {:.4}", i + 1, score);
    }
    
    // 実際の異常値（もし存在すれば）との比較
    if df.column("is_outlier").is_some() {
        let accuracy = compare_detection(&ocsvm_result, "anomaly", "is_outlier")?;
        println!("\n真の異常値との一致率: {:.2}%", accuracy * 100.0);
    }
    
    // 異なるガンマ値での比較
    println!("\n異なるガンマ値での比較:");
    
    let gamma_values = vec![0.01, 0.05, 0.1, 0.5, 1.0];
    
    for gamma in gamma_values {
        let mut model = OneClassSVM::new(0.05, gamma, 100, 1e-3);
        let result = model.fit_transform(df)?;
        let anomaly_count = result.column("anomaly").unwrap()
            .iter()
            .filter(|&x| match x {
                DataValue::Int64(v) => *v == 1,
                _ => false,
            })
            .count();
        
        println!("ガンマ = {} の結果: {} 異常", gamma, anomaly_count);
        
        if df.column("is_outlier").is_some() {
            let accuracy = compare_detection(&result, "anomaly", "is_outlier")?;
            println!("真の異常値との一致率: {:.2}%", accuracy * 100.0);
        }
    }
    
    Ok(())
}

// 異常検出アルゴリズムの比較
fn compare_algorithms(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== 異常検出アルゴリズムの比較 ====");
    
    // 1. Isolation Forest
    let mut iforest = IsolationForest::new(100, Some(256), Some(1.0), 0.05, Some(42));
    let iforest_result = iforest.fit_transform(df)?;
    
    // 2. LOF (Local Outlier Factor)
    let mut lof = LocalOutlierFactor::new(20, 0.05, DistanceMetric::Euclidean);
    let lof_result = lof.fit_transform(df)?;
    
    // 3. One-Class SVM
    let mut ocsvm = OneClassSVM::new(0.05, 0.1, 100, 1e-3);
    let ocsvm_result = ocsvm.fit_transform(df)?;
    
    // 検出された異常の数を表示
    let iforest_count = count_anomalies(&iforest_result, "anomaly")?;
    let lof_count = count_anomalies(&lof_result, "anomaly")?;
    let ocsvm_count = count_anomalies(&ocsvm_result, "anomaly")?;
    
    println!("\n検出された異常の数:");
    println!("- Isolation Forest: {} / {} サンプル", iforest_count, df.nrows());
    println!("- Local Outlier Factor: {} / {} サンプル", lof_count, df.nrows());
    println!("- One-Class SVM: {} / {} サンプル", ocsvm_count, df.nrows());
    
    // 実際の異常値（もし存在すれば）との比較
    if df.column("is_outlier").is_some() {
        let iforest_accuracy = compare_detection(&iforest_result, "anomaly", "is_outlier")?;
        let lof_accuracy = compare_detection(&lof_result, "anomaly", "is_outlier")?;
        let ocsvm_accuracy = compare_detection(&ocsvm_result, "anomaly", "is_outlier")?;
        
        println!("\n真の異常値との一致率:");
        println!("- Isolation Forest: {:.2}%", iforest_accuracy * 100.0);
        println!("- Local Outlier Factor: {:.2}%", lof_accuracy * 100.0);
        println!("- One-Class SVM: {:.2}%", ocsvm_accuracy * 100.0);
        
        // 精度、再現率、F1スコアの計算
        let (if_precision, if_recall, if_f1) = calc_precision_recall(&iforest_result, "anomaly", "is_outlier")?;
        let (lof_precision, lof_recall, lof_f1) = calc_precision_recall(&lof_result, "anomaly", "is_outlier")?;
        let (ocsvm_precision, ocsvm_recall, ocsvm_f1) = calc_precision_recall(&ocsvm_result, "anomaly", "is_outlier")?;
        
        println!("\n評価指標:");
        println!("- Isolation Forest: 精度={:.2}%, 再現率={:.2}%, F1={:.2}", 
                 if_precision * 100.0, if_recall * 100.0, if_f1);
        println!("- Local Outlier Factor: 精度={:.2}%, 再現率={:.2}%, F1={:.2}", 
                 lof_precision * 100.0, lof_recall * 100.0, lof_f1);
        println!("- One-Class SVM: 精度={:.2}%, 再現率={:.2}%, F1={:.2}", 
                 ocsvm_precision * 100.0, ocsvm_recall * 100.0, ocsvm_f1);
    }
    
    // アルゴリズム間の一致度分析
    let if_lof_agreement = calc_agreement(&iforest_result, &lof_result, "anomaly", "anomaly")?;
    let if_ocsvm_agreement = calc_agreement(&iforest_result, &ocsvm_result, "anomaly", "anomaly")?;
    let lof_ocsvm_agreement = calc_agreement(&lof_result, &ocsvm_result, "anomaly", "anomaly")?;
    
    println!("\nアルゴリズム間の一致度:");
    println!("- Isolation Forest vs LOF: {:.2}%", if_lof_agreement * 100.0);
    println!("- Isolation Forest vs One-Class SVM: {:.2}%", if_ocsvm_agreement * 100.0);
    println!("- LOF vs One-Class SVM: {:.2}%", lof_ocsvm_agreement * 100.0);
    
    Ok(())
}

// 異常値の数をカウント
fn count_anomalies(df: &DataFrame, anomaly_col: &str) -> Result<usize, PandRSError> {
    let anomaly_series = df.column(anomaly_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", anomaly_col))
    })?;
    
    let count = anomaly_series
        .iter()
        .filter(|&x| match x {
            DataValue::Int64(v) => *v == 1,
            _ => false,
        })
        .count();
    
    Ok(count)
}

// 検出結果を真の異常値と比較
fn compare_detection(df: &DataFrame, pred_col: &str, true_col: &str) -> Result<f64, PandRSError> {
    let pred_series = df.column(pred_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", pred_col))
    })?;
    
    let true_series = df.column(true_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", true_col))
    })?;
    
    // 一致率を計算
    let mut correct = 0;
    let total = df.nrows();
    
    for i in 0..total {
        let pred_value = match pred_series.get(i) {
            DataValue::Int64(v) => *v == 1,
            _ => false,
        };
        
        let true_value = match true_series.get(i) {
            DataValue::Int64(v) => *v == 1,
            DataValue::Boolean(b) => *b,
            _ => false,
        };
        
        if pred_value == true_value {
            correct += 1;
        }
    }
    
    Ok(correct as f64 / total as f64)
}

// 精度、再現率、F1スコアを計算
fn calc_precision_recall(df: &DataFrame, pred_col: &str, true_col: &str) -> Result<(f64, f64, f64), PandRSError> {
    let pred_series = df.column(pred_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", pred_col))
    })?;
    
    let true_series = df.column(true_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", true_col))
    })?;
    
    let mut true_positive = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;
    
    for i in 0..df.nrows() {
        let pred_value = match pred_series.get(i) {
            DataValue::Int64(v) => *v == 1,
            _ => false,
        };
        
        let true_value = match true_series.get(i) {
            DataValue::Int64(v) => *v == 1,
            DataValue::Boolean(b) => *b,
            _ => false,
        };
        
        if pred_value && true_value {
            true_positive += 1;
        } else if pred_value && !true_value {
            false_positive += 1;
        } else if !pred_value && true_value {
            false_negative += 1;
        }
    }
    
    let precision = if true_positive + false_positive > 0 {
        true_positive as f64 / (true_positive + false_positive) as f64
    } else {
        0.0
    };
    
    let recall = if true_positive + false_negative > 0 {
        true_positive as f64 / (true_positive + false_negative) as f64
    } else {
        0.0
    };
    
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    Ok((precision, recall, f1))
}

// アルゴリズム間の一致度を計算
fn calc_agreement(df1: &DataFrame, df2: &DataFrame, col1: &str, col2: &str) -> Result<f64, PandRSError> {
    let series1 = df1.column(col1).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", col1))
    })?;
    
    let series2 = df2.column(col2).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", col2))
    })?;
    
    let mut agreement = 0;
    let total = df1.nrows();
    
    for i in 0..total {
        let val1 = match series1.get(i) {
            DataValue::Int64(v) => *v,
            _ => 0,
        };
        
        let val2 = match series2.get(i) {
            DataValue::Int64(v) => *v,
            _ => 0,
        };
        
        if val1 == val2 {
            agreement += 1;
        }
    }
    
    Ok(agreement as f64 / total as f64)
}

// 異常値を含むサンプルデータの生成
fn create_sample_data_with_outliers() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::thread_rng();
    
    // 300サンプルのデータを生成
    let n_samples = 300;
    let n_outliers = 15; // 異常値の数
    let n_features = 5; // 5次元データ
    
    // 正常データのパラメータ
    let mean = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let std_dev = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    
    // データ構造を準備
    let mut features = Vec::new();
    for _ in 0..n_features {
        features.push(Vec::with_capacity(n_samples));
    }
    
    let mut is_outlier = Vec::with_capacity(n_samples);
    
    // 正常データを生成
    for _ in 0..(n_samples - n_outliers) {
        for j in 0..n_features {
            let value = rng.gen_range(-3.0..3.0) * std_dev[j] + mean[j];
            features[j].push(value);
        }
        is_outlier.push(DataValue::Int64(0)); // 正常 = 0
    }
    
    // 異常値を生成
    for _ in 0..n_outliers {
        for j in 0..n_features {
            // 異常値は正常範囲から離れた値
            let outlier_type = rng.gen_range(0..3);
            let value = match outlier_type {
                0 => rng.gen_range(5.0..10.0) * std_dev[j] + mean[j], // 正の大きな値
                1 => rng.gen_range(-10.0..-5.0) * std_dev[j] + mean[j], // 負の大きな値
                _ => {
                    if rng.gen_bool(0.5) {
                        rng.gen_range(5.0..10.0) * std_dev[j] + mean[j]
                    } else {
                        rng.gen_range(-10.0..-5.0) * std_dev[j] + mean[j]
                    }
                }
            };
            features[j].push(value);
        }
        is_outlier.push(DataValue::Int64(1)); // 異常 = 1
    }
    
    // データをシャッフル
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    
    // シャッフルされたデータを作成
    let mut shuffled_features = Vec::new();
    for _ in 0..n_features {
        shuffled_features.push(Vec::with_capacity(n_samples));
    }
    
    let mut shuffled_outliers = Vec::with_capacity(n_samples);
    
    for &idx in &indices {
        for j in 0..n_features {
            shuffled_features[j].push(features[j][idx]);
        }
        shuffled_outliers.push(is_outlier[idx].clone());
    }
    
    // DataFrame作成
    let mut df = DataFrame::new();
    
    for (j, feature) in shuffled_features.iter().enumerate() {
        df.add_column(format!("feature{}", j + 1), Series::from_vec(feature.clone())?)?;
    }
    
    // 異常値フラグを追加
    df.add_column("is_outlier".to_string(), Series::from_vec(shuffled_outliers)?)?;
    
    Ok(df)
}