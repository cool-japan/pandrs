extern crate pandrs;

use pandrs::optimized::OptimizedDataFrame;
use pandrs::ml::anomaly_detection::{IsolationForest, LocalOutlierFactor, OneClassSVM, DistanceMetric};
use pandrs::column::{Float64Column, Column, Int64Column};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 擬似的な異常検出データを生成
    println!("✅ 異常検出アルゴリズムの例");
    println!("==========================");
    println!("1. 擬似データの生成");
    
    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 1000;
    
    // 正常データの生成（2次元データ、中心 [0, 0] の正規分布）
    let n_normal = 950;
    let mut x_values = Vec::with_capacity(n_samples);
    let mut y_values = Vec::with_capacity(n_samples);
    let mut true_labels = Vec::with_capacity(n_samples);
    
    for _ in 0..n_normal {
        x_values.push(rng.gen_range(-3.0..3.0));
        y_values.push(rng.gen_range(-3.0..3.0));
        true_labels.push(0); // 正常なデータは0
    }
    
    // 異常値の生成（正常データから離れた場所）
    let n_anomalies = n_samples - n_normal;
    
    for _ in 0..n_anomalies {
        // 外れ値の位置をランダムに生成
        match rng.gen_range(0..4) {
            0 => { // 左上
                x_values.push(rng.gen_range(-10.0..-5.0));
                y_values.push(rng.gen_range(5.0..10.0));
            },
            1 => { // 右上
                x_values.push(rng.gen_range(5.0..10.0));
                y_values.push(rng.gen_range(5.0..10.0));
            },
            2 => { // 左下
                x_values.push(rng.gen_range(-10.0..-5.0));
                y_values.push(rng.gen_range(-10.0..-5.0));
            },
            _ => { // 右下
                x_values.push(rng.gen_range(5.0..10.0));
                y_values.push(rng.gen_range(-10.0..-5.0));
            }
        }
        true_labels.push(1); // 異常値は1
    }
    
    // データフレームの作成
    let mut df = OptimizedDataFrame::new();
    
    let x_col = Column::Float64(Float64Column::new(x_values, false, "x".to_string())?);
    let y_col = Column::Float64(Float64Column::new(y_values, false, "y".to_string())?);
    let true_labels_col = Column::Int64(Int64Column::new(true_labels, false, "true_anomaly".to_string())?);
    
    df.add_column("x".to_string(), x_col)?;
    df.add_column("y".to_string(), y_col)?;
    df.add_column("true_anomaly".to_string(), true_labels_col)?;
    
    println!("データ生成完了: {} 正常サンプル, {} 異常サンプル", n_normal, n_anomalies);
    println!("データフレームの最初の数行:");
    println!("{}", df.head(5)?);
    
    // IsolationForest による異常検出
    println!("\n2. Isolation Forest による異常検出");
    let mut isolation_forest = IsolationForest::new(
        100,                    // 決定木の数
        None,                   // サブサンプリングサイズ（デフォルト）
        None,                   // 特徴量サブサンプリング（デフォルト）
        0.05,                   // 汚染率 5%
        Some(42),               // 乱数シード
    );
    
    let if_result = isolation_forest.fit_transform(&df)?;
    
    println!("Isolation Forest 検出完了");
    println!("検出された異常値の数: {}", isolation_forest.labels().iter().filter(|&&x| x == 1).count());
    println!("結果の最初の数行:");
    println!("{}", if_result.head(5)?);
    
    // LOF による異常検出
    println!("\n3. Local Outlier Factor による異常検出");
    let mut lof = LocalOutlierFactor::new(
        20,                      // 近傍数
        0.05,                    // 汚染率 5%
        DistanceMetric::Euclidean, // 距離メトリック
    );
    
    let lof_result = lof.fit_transform(&df)?;
    
    println!("Local Outlier Factor 検出完了");
    println!("検出された異常値の数: {}", lof.labels().iter().filter(|&&x| x == 1).count());
    println!("結果の最初の数行:");
    println!("{}", lof_result.head(5)?);
    
    // 一クラスSVM による異常検出
    println!("\n4. One-Class SVM による異常検出");
    let mut one_class_svm = OneClassSVM::new(
        0.05,                    // nu パラメータ
        0.1,                     // gamma パラメータ
        100,                     // 最大反復回数
        1e-3,                    // 収束閾値
    );
    
    let svm_result = one_class_svm.fit_transform(&df)?;
    
    println!("One-Class SVM 検出完了");
    println!("検出された異常値の数: {}", one_class_svm.labels().iter().filter(|&&x| x == 1).count());
    println!("結果の最初の数行:");
    println!("{}", svm_result.head(5)?);
    
    // 各アルゴリズムの異常フラグの比較
    println!("\n5. 検出結果の比較");
    
    // 異常とフラグ付けされたサンプルの数
    let if_anomalies = isolation_forest.labels().iter().filter(|&&x| x == 1).count();
    let lof_anomalies = lof.labels().iter().filter(|&&x| x == 1).count();
    let svm_anomalies = one_class_svm.labels().iter().filter(|&&x| x == 1).count();
    
    println!("Isolation Forest: {} 個の異常値を検出", if_anomalies);
    println!("Local Outlier Factor: {} 個の異常値を検出", lof_anomalies);
    println!("One-Class SVM: {} 個の異常値を検出", svm_anomalies);
    
    // アルゴリズム間の一致度を確認
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
            
            // 最初の数個の「全アルゴリズムが一致した」異常値を表示
            if if_label == 1 && all_agree <= 5 { // 最初の5個のみ表示
                println!("サンプル {} は全アルゴリズムが異常値と判定: x={:.2}, y={:.2}", 
                    i, x_values[i], y_values[i]);
            }
        }
        
        if if_label == lof_label { if_lof_agree += 1; }
        if if_label == svm_label { if_svm_agree += 1; }
        if lof_label == svm_label { lof_svm_agree += 1; }
    }
    
    println!("全アルゴリズムの判定一致率: {:.1}%", 100.0 * all_agree as f64 / n_samples as f64);
    println!("Isolation ForestとLOFの判定一致率: {:.1}%", 100.0 * if_lof_agree as f64 / n_samples as f64);
    println!("Isolation ForestとSVMの判定一致率: {:.1}%", 100.0 * if_svm_agree as f64 / n_samples as f64);
    println!("LOFとSVMの判定一致率: {:.1}%", 100.0 * lof_svm_agree as f64 / n_samples as f64);
    
    // 真の異常値との比較
    println!("\n6. 真の異常値との比較");
    
    // 評価指標関数
    let calc_metrics = |algorithm_name: &str, labels: &[i64], true_labels: &[i64]| {
        let mut tp = 0; // 真陽性
        let mut fp = 0; // 偽陽性
        let mut tn = 0; // 真陰性
        let mut fn = 0; // 偽陰性
        
        for i in 0..labels.len() {
            let pred = labels[i];
            let true_val = true_labels[i];
            
            match (pred, true_val) {
                (1, 1) => tp += 1, // 真陽性
                (1, 0) => fp += 1, // 偽陽性
                (-1, 0) => tn += 1, // 真陰性
                (-1, 1) => fn += 1, // 偽陰性
                _ => {}
            }
        }
        
        // 精度、再現率、F1スコアを計算
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn > 0 { tp as f64 / (tp + fn) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        let accuracy = (tp + tn) as f64 / labels.len() as f64;
        
        println!("{}: 精度={:.1}%, 再現率={:.1}%, F1={:.3}, 正解率={:.1}%", 
                algorithm_name, precision * 100.0, recall * 100.0, f1, accuracy * 100.0);
    };
    
    // 各アルゴリズムの評価指標を計算
    calc_metrics("Isolation Forest", isolation_forest.labels(), &true_labels);
    calc_metrics("Local Outlier Factor", lof.labels(), &true_labels);
    calc_metrics("One-Class SVM", one_class_svm.labels(), &true_labels);
    
    println!("\n==========================");
    println!("✅ 異常検出の例が正常に完了しました");
    
    Ok(())
}