//! Regression tests for `src/ml/{clustering,anomaly,dimension}/mod.rs` (wave 2).
//!
//! Pins down behaviour that was previously wrong:
//!
//! * `KMeans::fit` inertia is the sum of SQUARED distances to the assigned
//!   centroid (it used to sum raw, unsquared distances, which also made the
//!   `tol` convergence check compare the wrong quantity).
//! * `KMeans` now defaults to k-means++ initialization with `n_init = 10`
//!   independent restarts (previously a single uniform-random restart),
//!   which reliably finds a better (lower-inertia) optimum on fixtures where
//!   random initialization can miss a whole cluster.
//! * `IsolationForest::predict` reuses the threshold computed from the
//!   TRAINING data at `fit` time, instead of recomputing a threshold from
//!   whatever batch is passed to `predict` (which made a single-row batch
//!   always "the most anomalous row in its own batch", and therefore always
//!   flagged as an anomaly regardless of how normal it actually was).
//! * `TSNE::evaluate`'s `kl_divergence` metric is the real KL divergence
//!   between the high-dimensional P and the final embedding's Q, recomputed
//!   from the cached (non-early-exaggerated) P matrix — it used to be a
//!   hardcoded `0.0` placeholder that could never reflect optimization
//!   progress.
//!
//! (A fifth fix from the same audit — the PCA Jacobi eigendecomposition's
//! rotation budget scaling with matrix size instead of a fixed 1000-rotation
//! cap — is regression-tested directly against the private
//! `jacobi_eigen_symmetric` helper inside
//! `src/ml/dimension/mod.rs`'s own `#[cfg(test)]` module
//! (`test_jacobi_eigen_p50_dense_matches_reference`), since that helper is
//! not part of the public API and cannot be reached from an external
//! integration test.)

use pandrs::ml::clustering::KMeansInit;
use pandrs::{
    DataFrame, IsolationForest, KMeans, ModelEvaluator, Series, TSNEInit, UnsupervisedModel, TSNE,
};

fn make_df(cols: &[(&str, Vec<f64>)]) -> DataFrame {
    let mut df = DataFrame::new();
    for (name, data) in cols {
        df.add_column(
            name.to_string(),
            Series::new(data.clone(), Some(name.to_string())).expect("Series::new"),
        )
        .expect("add_column");
    }
    df
}

// ---------------------------------------------------------------------
// KMeans inertia = sum of SQUARED distances
// ---------------------------------------------------------------------

/// Two exactly-symmetric 1-D clusters — {-1, 0, 1} around mean 0 and
/// {9, 10, 11} around mean 10 — have an obvious, unique global optimum: the
/// two cluster means as centroids. The resulting inertia is hand-computable:
/// sum of squared distances = (1^2 + 0^2 + 1^2) + (1^2 + 0^2 + 1^2) = 4.0.
/// Before the fix, `inertia` summed raw (unsquared) distances, which would
/// have given (1 + 0 + 1) + (1 + 0 + 1) = 4.0 as well for THIS particular
/// symmetric fixture by coincidence of all distances being 0 or 1 — so the
/// assertion below pins the value AND the assertion further down
/// (`test_kmeans_inertia_matches_hand_computed_sum_of_squares_asymmetric`)
/// uses an asymmetric fixture where squared-vs-unsquared sums differ, which
/// is what actually discriminates the fix from the bug.
#[test]
fn test_kmeans_inertia_matches_hand_computed_sum_of_squares() {
    let xs = vec![-1.0_f64, 0.0, 1.0, 9.0, 10.0, 11.0];
    let df = make_df(&[("x", xs)]);

    let mut km = KMeans::new(2)
        .random_seed(42)
        .n_init(10)
        .with_columns(vec!["x".to_string()]);
    km.fit(&df).unwrap();

    let inertia = km.inertia.expect("inertia must be set after fit");
    assert!(
        (inertia - 4.0).abs() < 1e-9,
        "expected inertia == 4.0 (sum of squared distances to the two cluster means), got {}",
        inertia
    );
}

/// An asymmetric fixture where "sum of squared distances" and "sum of raw
/// distances" give DIFFERENT numbers, so this test actually discriminates
/// the fix (inertia must be the SQUARED-distance sum) from the bug (it used
/// to sum raw distances). Cluster A = {0, 2} (mean 1, squared-dist sum
/// 1^2+1^2=2, raw-dist sum 1+1=2 — still coincides) is not useful; instead
/// use asymmetric offsets 0.5 and 2.0 around a cluster whose mean is exactly
/// known by construction: points {-2, 0, 2} (mean 0) and {6, 10, 14} (mean
/// 10). Squared-distance inertia = (4+0+4) + (16+0+16) = 40. Raw-distance
/// "inertia" (the old, buggy quantity) would instead have been
/// (2+0+2) + (4+0+4) = 12 — a different number, so this test would fail
/// under the old behaviour and passes under the fixed one.
#[test]
fn test_kmeans_inertia_matches_hand_computed_sum_of_squares_asymmetric() {
    let xs = vec![-2.0_f64, 0.0, 2.0, 6.0, 10.0, 14.0];
    let df = make_df(&[("x", xs)]);

    let mut km = KMeans::new(2)
        .random_seed(1)
        .n_init(10)
        .max_iter(200)
        .with_columns(vec!["x".to_string()]);
    km.fit(&df).unwrap();

    let inertia = km.inertia.expect("inertia must be set after fit");
    assert!(
        (inertia - 40.0).abs() < 1e-6,
        "expected inertia == 40.0 (sum of SQUARED distances to the two cluster means: \
         (4+0+4)+(16+0+16)), got {} -- if this is 12.0 the code is summing raw \
         (unsquared) distances again",
        inertia
    );
}

// ---------------------------------------------------------------------
// k-means++ beats uniform-random initialization
// ---------------------------------------------------------------------

/// Four tight, well-separated clusters at the corners of a square. With a
/// single uniform-random restart (`n_init(1)`), there is a real chance
/// (combinatorially, about 89% for 4 draws out of 4 groups of 8 -- see the
/// derivation below) of drawing two initial centroids from the SAME corner,
/// leaving another corner with no dedicated centroid. k-means++ explicitly
/// spreads its picks out (probability proportional to squared distance from
/// already-chosen centroids), so it reliably picks one centroid per corner.
///
/// `max_iter` is deliberately small (2). With a generous iteration budget,
/// plain Lloyd reassignment -- helped further by this crate's OWN
/// distinct-point empty-cluster reseeding -- eventually self-corrects a bad
/// random start on data this well-separated, converging to the same optimum
/// as k-means++ almost every time and masking the very difference this test
/// exists to catch. Capping `max_iter` at 2 isolates the quality of the
/// STARTING point (which is exactly what k-means++ improves) rather than
/// the algorithm's long-run ability to recover from a bad one.
///
/// The true global optimum for this exact fixture can be computed by hand:
/// each corner's 8 points share the same multiset of small offsets in x and
/// in y (just permuted between the two axes), so every corner has the same
/// within-cluster sum of squared deviations from its own mean, `0.505`, in
/// both x and y => `1.01` per corner => `4 * 1.01 = 4.04` total. k-means++
/// reaches (very close to) this exact value because it reliably seeds one
/// centroid per corner; a stranded random start cannot, within only 2
/// iterations.
///
/// `SEED` was chosen empirically (by sweeping seeds 0..3000 against this
/// exact fixture) to be one of the ~89% of seeds where uniform-random
/// initialization actually strands a cluster within 2 iterations -- an
/// earlier choice of seed happened to fall in the ~11% where even the random
/// restart reaches the same global optimum by luck, which made the "strictly
/// better" assertion below flaky with respect to seed choice even though the
/// underlying fix is correct. `SEED = 0` reproducibly gives random-init
/// inertia ~535 against k-means++'s ~4.04 -- not a borderline margin.
#[test]
fn test_kmeans_plusplus_beats_random_init_on_four_corners_fixture() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let corners = [(0.0_f64, 0.0_f64), (0.0, 30.0), (30.0, 0.0), (30.0, 30.0)];
    let jitter = [-0.4_f64, -0.2, -0.1, 0.05, 0.15, 0.35, -0.3, 0.25];
    for &(cx, cy) in &corners {
        for (i, &j) in jitter.iter().enumerate() {
            xs.push(cx + j);
            ys.push(cy + jitter[(i + 3) % jitter.len()]);
        }
    }
    let df = make_df(&[("x", xs), ("y", ys)]);

    const SEED: u64 = 0;
    const MAX_ITER: usize = 2;
    const KNOWN_OPTIMUM: f64 = 4.04;

    let mut km_random = KMeans::new(4)
        .init(KMeansInit::Random)
        .n_init(1)
        .max_iter(MAX_ITER)
        .random_seed(SEED)
        .with_columns(vec!["x".to_string(), "y".to_string()]);
    km_random.fit(&df).unwrap();
    let random_inertia = km_random.inertia.unwrap();

    let mut km_plusplus = KMeans::new(4)
        .init(KMeansInit::KMeansPlusPlus)
        .n_init(1)
        .max_iter(MAX_ITER)
        .random_seed(SEED)
        .with_columns(vec!["x".to_string(), "y".to_string()]);
    km_plusplus.fit(&df).unwrap();
    let plusplus_inertia = km_plusplus.inertia.unwrap();

    // Strongest, seed-independent claim: k-means++ reaches (very close to)
    // the analytically-known global optimum within just 2 Lloyd iterations,
    // because it reliably places one initial centroid per corner.
    assert!(
        (plusplus_inertia - KNOWN_OPTIMUM).abs() < 0.5,
        "k-means++ should reach ~{} (the known global optimum) within {} iterations \
         by reliably seeding one centroid per corner, got {}",
        KNOWN_OPTIMUM,
        MAX_ITER,
        plusplus_inertia
    );

    // Direct comparison: k-means++ must be at least as good as random init.
    assert!(
        plusplus_inertia <= random_inertia + 1e-9,
        "k-means++ inertia ({}) should be <= random-init inertia ({}) on a fixture \
         designed so random init can strand a whole cluster",
        plusplus_inertia,
        random_inertia
    );
    assert!(
        plusplus_inertia < random_inertia - 50.0,
        "expected k-means++ to be STRICTLY better than random init by a wide margin for \
         seed {} on this fixture (plusplus={}, random={}); if the gap collapses the \
         fixture/seed no longer demonstrates the improvement and should be revisited",
        SEED,
        plusplus_inertia,
        random_inertia
    );
}

// ---------------------------------------------------------------------
// IsolationForest::predict uses the FITTED threshold, not a per-batch one
// ---------------------------------------------------------------------

/// Fit on a single tight cluster of "normal" points, then call `predict` on
/// a lone point sitting right at the middle of that cluster. Before the fix,
/// `predict` recomputed its threshold from the prediction batch itself, so a
/// single-row batch was always "the most anomalous row among itself" and
/// therefore ALWAYS labeled -1 (anomaly) no matter how normal the point
/// actually was. After the fix, the threshold comes from the training
/// distribution, so a clearly-central point must be labeled 1 (normal).
#[test]
fn test_isolation_forest_single_row_predict_not_always_anomaly() {
    // A tight, roughly symmetric cluster of "normal" points centered at the
    // origin.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..10i32 {
        for j in 0..4i32 {
            let fx = (i - 5) as f64 * 0.3 + (j as f64) * 0.02;
            let fy = (j - 2) as f64 * 0.3 + (i as f64) * 0.01;
            xs.push(fx);
            ys.push(fy);
        }
    }
    let train_df = make_df(&[("x", xs), ("y", ys)]);

    let mut ifo = IsolationForest::new()
        .n_estimators(200)
        .contamination(0.1)
        .random_seed(42);
    ifo.fit(&train_df).unwrap();

    // A single row sitting at the dead center of the training distribution.
    let single_row_df = make_df(&[("x", vec![0.0_f64]), ("y", vec![0.0_f64])]);
    let labels = ifo.predict(&single_row_df).unwrap();

    assert_eq!(labels.len(), 1);
    assert_eq!(
        labels[0], 1.0,
        "A single row at the center of the training distribution must be predicted \
         NORMAL (1.0), not anomalous (-1.0) purely because it is alone in its batch"
    );

    // Sanity check in the other direction: an extreme outlier, still
    // predicted alone in its own batch, must still come out anomalous.
    let outlier_df = make_df(&[("x", vec![50.0_f64]), ("y", vec![50.0_f64])]);
    let outlier_labels = ifo.predict(&outlier_df).unwrap();
    assert_eq!(
        outlier_labels[0], -1.0,
        "An extreme outlier predicted alone in its own batch must still be flagged \
         anomalous"
    );
}

// ---------------------------------------------------------------------
// t-SNE KL divergence decreases with more optimization
// ---------------------------------------------------------------------

/// Fit two t-SNE embeddings on the SAME data and seed, differing only in
/// `n_iter` (a handful of iterations vs. a fully-optimized run), and compare
/// the real KL divergence `evaluate()` now reports. More optimization should
/// bring the low-dimensional Q closer to the high-dimensional P, decreasing
/// KL(P || Q). Before the fix, `evaluate()` always reported a hardcoded
/// `0.0`, which trivially could not decrease (or do anything else).
#[test]
fn test_tsne_kl_divergence_decreases_with_more_iterations() {
    let n = 24usize;
    let mut fa = Vec::new();
    let mut fb = Vec::new();
    let mut fc = Vec::new();
    for i in 0..n {
        let cluster = i % 3;
        let base = (cluster as f64) * 15.0;
        let jitter = (i as f64) * 0.05;
        fa.push(base + jitter);
        fb.push(base * 0.6 + jitter * 0.3);
        fc.push(base * 1.3 - jitter * 0.2);
    }
    let df = make_df(&[("a", fa), ("b", fb), ("c", fc)]);

    let mut tsne_short = TSNE::with_params(2, 5.0, 2, 100.0, TSNEInit::Random);
    tsne_short.random_seed = Some(11);
    tsne_short.fit(&df).unwrap();
    let short_kl = tsne_short
        .evaluate(&df, "")
        .unwrap()
        .get_metric("kl_divergence")
        .copied()
        .unwrap();

    let mut tsne_long = TSNE::with_params(2, 5.0, 500, 100.0, TSNEInit::Random);
    tsne_long.random_seed = Some(11);
    tsne_long.fit(&df).unwrap();
    let long_kl = tsne_long
        .evaluate(&df, "")
        .unwrap()
        .get_metric("kl_divergence")
        .copied()
        .unwrap();

    assert!(short_kl.is_finite() && long_kl.is_finite());
    assert!(
        long_kl < short_kl,
        "KL divergence after 500 iterations ({}) should be lower than after only 2 \
         iterations ({}) -- evaluate() must recompute a real, improving KL divergence, \
         not a constant placeholder",
        long_kl,
        short_kl
    );
}
