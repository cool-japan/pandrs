//! SciRS2 Integration Example
//!
//! Demonstrates the `scirs2` feature: Spearman/Pearson correlation, covariance,
//! hypothesis tests, PCA, QR decomposition, and matrix rank — all via `SciRS2Ext`.
//!
//! Run with:
//! ```bash
//! cargo run --example scirs2_integration_example --features scirs2
//! ```

#[cfg(not(feature = "scirs2"))]
fn main() {
    eprintln!("This example requires the `scirs2` feature.");
    eprintln!("Run: cargo run --example scirs2_integration_example --features scirs2");
}

#[cfg(feature = "scirs2")]
fn main() {
    use pandrs::scirs2_integration::dataframe_ext::SciRS2Ext;
    use pandrs::scirs2_integration::stats::SciRS2Stats;
    use pandrs::{DataFrame, Series};

    // ── Build a small DataFrame ──────────────────────────────────────────────
    let n = 20usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let z: Vec<f64> = x.iter().map(|v| v * v - 5.0 * v + 10.0).collect();

    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(x.clone(), Some("x".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "y".to_string(),
        Series::new(y.clone(), Some("y".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "z".to_string(),
        Series::new(z.clone(), Some("z".to_string())).unwrap(),
    )
    .unwrap();

    println!("=== SciRS2 Integration Demo ===\n");

    // ── Pearson correlation ──────────────────────────────────────────────────
    println!("--- Pearson Correlation ---");
    let pearson = df.scirs2_corr().expect("Pearson corr");
    let cols = pearson.column_names();
    print!("{:<12}", "");
    for c in &cols {
        print!("{:<12}", c);
    }
    println!();
    for row in 0..pearson.nrows() {
        for c in &cols {
            let vals = pearson.get_column_numeric_values(c).unwrap_or_else(|_| {
                pearson
                    .get_column_string_values(c)
                    .map(|v| {
                        v.iter()
                            .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
                            .collect()
                    })
                    .unwrap_or_default()
            });
            if c == "column" {
                if let Ok(sv) = pearson.get_column_string_values(c) {
                    print!("{:<12}", sv.get(row).map(|s| s.as_str()).unwrap_or(""));
                } else {
                    print!("{:<12}", "");
                }
            } else {
                let v = vals.get(row).copied().unwrap_or(f64::NAN);
                print!("{:<12.4}", v);
            }
        }
        println!();
    }

    // ── Spearman correlation ─────────────────────────────────────────────────
    println!("\n--- Spearman Rank Correlation ---");
    let spearman = df.scirs2_spearman_corr().expect("Spearman corr");
    println!(
        "Columns: {:?}",
        spearman
            .column_names()
            .iter()
            .filter(|c| c.as_str() != "column")
            .collect::<Vec<_>>()
    );
    println!("(see Pearson table format above; spearman ρ for monotone data ≈ 1.0)");

    // ── Covariance matrix ────────────────────────────────────────────────────
    println!("\n--- Covariance Matrix ---");
    let cov = df.scirs2_cov().expect("Cov matrix");
    let cov_cols: Vec<_> = cov
        .column_names()
        .into_iter()
        .filter(|c| c != "column")
        .collect();
    for c in &cov_cols {
        let vals = cov.get_column_numeric_values(c).unwrap();
        println!(
            "  cov[{}]: {:?}",
            c,
            vals.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>()
        );
    }

    // ── Paired t-test: x vs y ────────────────────────────────────────────────
    println!("\n--- Paired t-test (x vs y) ---");
    let tt = SciRS2Stats::ttest_paired(&x, &y).expect("ttest_paired");
    println!(
        "  t = {:.4}, p = {:.4e}, df = {:.1}",
        tt.statistic, tt.p_value, tt.df
    );

    // ── One-way ANOVA across three groups ────────────────────────────────────
    println!("\n--- One-way ANOVA (x, y, z) ---");
    let g1: Vec<f64> = x[..7].to_vec();
    let g2: Vec<f64> = y[..7].to_vec();
    let g3: Vec<f64> = z[..7].to_vec();
    let anova = SciRS2Stats::f_oneway(&[&g1, &g2, &g3]).expect("f_oneway");
    println!("  F = {:.4}, p = {:.4e}", anova.f_statistic, anova.p_value);

    // ── Shapiro-Wilk normality test ──────────────────────────────────────────
    println!("\n--- Shapiro-Wilk Normality Test (x) ---");
    // Use a smaller sample (Shapiro-Wilk requires n ≤ ~5000)
    let norm_result = SciRS2Stats::shapiro_wilk_test(&x[..15]).expect("shapiro_wilk");
    println!(
        "  W = {:.4}, p = {:.4e}, is_normal = {}",
        norm_result.statistic, norm_result.p_value, norm_result.is_normal
    );

    // ── PCA ──────────────────────────────────────────────────────────────────
    println!("\n--- PCA (2 components) ---");
    let pca = df.scirs2_pca(2).expect("PCA");
    println!(
        "  Explained variance ratio: {:?}",
        pca.explained_variance_ratio
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );
    println!(
        "  Components shape: {} rows × {} cols",
        pca.components.nrows(),
        pca.components.ncols()
    );

    // ── QR decomposition ─────────────────────────────────────────────────────
    println!("\n--- QR Decomposition ---");
    let mut sq = DataFrame::new();
    sq.add_column(
        "a".to_string(),
        Series::new(vec![1.0f64, 2.0, 3.0], Some("a".to_string())).unwrap(),
    )
    .unwrap();
    sq.add_column(
        "b".to_string(),
        Series::new(vec![4.0f64, 5.0, 6.0], Some("b".to_string())).unwrap(),
    )
    .unwrap();
    sq.add_column(
        "c".to_string(),
        Series::new(vec![7.0f64, 8.0, 9.1], Some("c".to_string())).unwrap(),
    )
    .unwrap();
    let qr = sq.scirs2_qr().expect("QR");
    println!("  Q shape: {} × {}", qr.q.nrows(), qr.q.ncols());
    println!("  R shape: {} × {}", qr.r.nrows(), qr.r.ncols());

    // ── Matrix rank ───────────────────────────────────────────────────────────
    println!("\n--- Matrix Rank ---");
    let rank = sq.scirs2_matrix_rank().expect("rank");
    println!("  Rank of 3×3 near-singular matrix: {}", rank);

    println!("\n=== Done ===");
}
