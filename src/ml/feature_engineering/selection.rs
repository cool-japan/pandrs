//! Feature selection methods for [`AutoFeatureEngineer`]: `select_features` and each
//! `FeatureSelectionMethod` variant's implementation (KBest dispatch, recursive
//! elimination, L1-based Lasso, tree-based importances, mutual information, and
//! variance threshold), plus the private numeric helpers those need (Lasso coordinate
//! descent, decision-tree feature importances).

use super::{AutoFeatureEngineer, FeatureSelectionMethod};
use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::model_selection::SelectKBest;
use crate::ml::models::linear::LinearRegression;
use crate::ml::models::{DecisionTreeConfig, DecisionTreeRegressor, SupervisedModel};
use crate::series::Series;
use std::collections::HashMap;

/// Soft-thresholding operator used by Lasso coordinate descent:
/// `S(x, lambda) = sign(x) * max(|x| - lambda, 0)`.
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// One coordinate-descent solve to (near-)convergence for a fixed `alpha`,
/// warm-started from `beta_init`. `x_std` is row-major (`n` rows x `p`
/// columns) and pre-standardized to zero mean / unit population variance
/// per column; `y_c` is `y` pre-centered to zero mean. Solves:
///
/// `argmin_beta (1/(2n)) * ||y_c - x_std @ beta||^2 + alpha * ||beta||_1`
///
/// via cyclic coordinate descent (Friedman, Hastie & Tibshirani's
/// `glmnet` algorithm): each coordinate is updated in turn to its
/// soft-thresholded partial-residual solution, holding all others fixed,
/// until the largest per-coordinate change drops below `tol` or
/// `max_iter` passes are used.
fn lasso_coordinate_descent(
    x_std: &[Vec<f64>],
    y_c: &[f64],
    alpha: f64,
    beta_init: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = y_c.len();
    let p = beta_init.len();
    let mut beta = beta_init.to_vec();

    // residual = y_c - X*beta, maintained incrementally as beta updates.
    let mut residual = y_c.to_vec();
    for (i, row) in x_std.iter().enumerate() {
        for (j, &xij) in row.iter().enumerate() {
            residual[i] -= xij * beta[j];
        }
    }

    let n_f = n as f64;
    for _ in 0..max_iter {
        let mut max_change: f64 = 0.0;
        for j in 0..p {
            // rho_j = (1/n) * sum_i x_ij * (r_i + x_ij * beta_j): the
            // partial residual w.r.t. feature j, i.e. the residual with
            // feature j's own current contribution added back in.
            let mut rho = 0.0;
            for i in 0..n {
                rho += x_std[i][j] * (residual[i] + x_std[i][j] * beta[j]);
            }
            rho /= n_f;

            let new_beta_j = soft_threshold(rho, alpha);
            let delta = new_beta_j - beta[j];
            if delta.abs() > 1e-15 {
                for i in 0..n {
                    residual[i] -= x_std[i][j] * delta;
                }
            }
            max_change = max_change.max(delta.abs());
            beta[j] = new_beta_j;
        }
        if max_change < tol {
            break;
        }
    }

    beta
}

/// Compute normalized feature importances for a fitted
/// [`DecisionTreeRegressor`], from its public `nodes()`.
///
/// The regressor does not (yet) populate its own `feature_importances()`
/// (unlike `DecisionTreeClassifier`, which computes this internally), so
/// this reads the tree structure directly using the same
/// impurity-decrease-weighted formula rather than depending on that.
fn tree_feature_importances(tree: &DecisionTreeRegressor, n_features: usize) -> Vec<f64> {
    let nodes = tree.nodes();
    let mut importances = vec![0.0_f64; n_features];
    let total_samples = nodes.first().map(|n| n.n_samples).unwrap_or(1) as f64;

    for node in nodes {
        if node.is_leaf {
            continue;
        }
        if let (Some(feature_idx), Some(left_idx), Some(right_idx)) =
            (node.feature_index, node.left_child, node.right_child)
        {
            let left = &nodes[left_idx];
            let right = &nodes[right_idx];
            let weighted_decrease = (node.n_samples as f64 / total_samples)
                * (node.impurity
                    - (left.n_samples as f64 / node.n_samples as f64) * left.impurity
                    - (right.n_samples as f64 / node.n_samples as f64) * right.impurity);
            if feature_idx < importances.len() {
                importances[feature_idx] += weighted_decrease;
            }
        }
    }

    let sum: f64 = importances.iter().sum();
    if sum > 0.0 {
        for imp in &mut importances {
            *imp /= sum;
        }
    }
    importances
}

impl AutoFeatureEngineer {
    /// Select features using the configured selection method.
    ///
    /// Also populates `feature_scores_` with the per-feature score each
    /// method produced (variance, |Lasso coefficient|, tree importance, MI,
    /// RFE survival count, or the `SelectKBest` statistic), so callers can
    /// inspect *why* features were ranked the way they were via
    /// `get_feature_scores()`. `Custom` selectors are opaque functions with
    /// no natural per-feature score, so they leave `feature_scores_` unset
    /// rather than fabricating one.
    pub(super) fn select_features(&mut self, x: &DataFrame, y: &DataFrame) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();

        let (selected_indices, scores): (Vec<usize>, Option<Vec<f64>>) =
            match &self.selection_method {
                FeatureSelectionMethod::KBest(score_func) => {
                    // Clamp k to the available feature count: a caller-supplied
                    // `n_features_to_select` larger than `n_features` (e.g. an
                    // AutoML step requesting 50 features from a 10-feature
                    // dataset) previously made `SelectKBest::fit` hard-error.
                    let k = self
                        .n_features_to_select
                        .unwrap_or(n_features.min(20))
                        .min(n_features)
                        .max(1);
                    let mut selector = SelectKBest::new(score_func.clone(), k);
                    selector.fit(x, y)?;
                    let selected = selector.get_selected_features().unwrap_or(&[]).to_vec();
                    let scores = selector.get_scores().map(|s| s.to_vec());
                    (selected, scores)
                }
                FeatureSelectionMethod::VarianceThreshold(threshold) => {
                    let (selected, scores) = self.select_by_variance_threshold(x, *threshold)?;
                    (selected, Some(scores))
                }
                FeatureSelectionMethod::RecursiveElimination => {
                    let k = self.n_features_to_select.unwrap_or(n_features.min(10));
                    let (selected, scores) = self.select_recursive_elimination(x, y, k)?;
                    (selected, Some(scores))
                }
                FeatureSelectionMethod::L1Based => {
                    let k = self.n_features_to_select.unwrap_or(n_features.min(15));
                    let (selected, scores) = self.select_l1_based(x, y, k)?;
                    (selected, Some(scores))
                }
                FeatureSelectionMethod::TreeBased => {
                    let k = self.n_features_to_select.unwrap_or(n_features.min(12));
                    let (selected, scores) = self.select_tree_based(x, y, k)?;
                    (selected, Some(scores))
                }
                FeatureSelectionMethod::MutualInformation => {
                    let k = self.n_features_to_select.unwrap_or(n_features.min(18));
                    let (selected, scores) = self.select_mutual_information(x, y, k)?;
                    (selected, Some(scores))
                }
                FeatureSelectionMethod::Custom(func) => (func(x, y)?, None),
            };

        if let Some(scores) = scores {
            let score_map: HashMap<String, f64> = feature_names
                .iter()
                .cloned()
                .zip(scores.into_iter())
                .collect();
            self.feature_scores_ = Some(score_map);
        }

        Ok(selected_indices)
    }

    /// Recursive Feature Elimination (RFE): iteratively removes the feature
    /// with the smallest absolute LinearRegression coefficient until `k` remain.
    fn select_recursive_elimination(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        // Extract target column from y (use first column)
        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?.to_vec();

        // Track which original feature indices are still active (by position in feature_names)
        let mut active_indices: Vec<usize> = (0..n_features).collect();
        // Number of elimination rounds each feature was still active for.
        // Features removed earlier stop accumulating sooner, so this is a
        // monotonic "importance by survival" score: the final survivors
        // (incremented once more after the loop) always rank above
        // anything eliminated along the way. This avoids mixing OLS
        // coefficient magnitudes across the different reduced models fit
        // at each round, which aren't on a comparable scale.
        let mut survived = vec![0usize; n_features];

        while active_indices.len() > k {
            for &idx in &active_indices {
                survived[idx] += 1;
            }

            // Build a temporary DataFrame with active features + target
            let mut train_df = DataFrame::new();
            for &idx in &active_indices {
                let fname = &feature_names[idx];
                let col = x.get_column::<f64>(fname)?;
                train_df.add_column(fname.clone(), col.clone())?;
            }
            train_df.add_column(
                target_col_name.clone(),
                Series::new(target_vals.clone(), Some(target_col_name.clone()))?,
            )?;

            // Fit LinearRegression on the current active feature subset
            let mut lr = LinearRegression::new();
            match lr.fit(&train_df, target_col_name) {
                Ok(()) => {}
                Err(_) => {
                    // If fit fails (e.g., singular matrix), just drop the last active feature
                    active_indices.pop();
                    continue;
                }
            }

            let coefs = lr.coefficients.as_ref().ok_or_else(|| {
                Error::InvalidOperation("LinearRegression fit produced no coefficients".into())
            })?;

            // Find the active feature with the smallest |coefficient|
            let mut min_importance = f64::INFINITY;
            let mut min_pos = 0usize; // position in active_indices to remove

            for (pos, &orig_idx) in active_indices.iter().enumerate() {
                let fname = &feature_names[orig_idx];
                let importance = coefs.get(fname).map(|c| c.abs()).unwrap_or(0.0);
                if importance < min_importance {
                    min_importance = importance;
                    min_pos = pos;
                }
            }

            active_indices.remove(min_pos);
        }

        // Final survivors get one more increment so they strictly outrank
        // anything eliminated in the very last round.
        for &idx in &active_indices {
            survived[idx] += 1;
        }

        active_indices.sort_unstable();
        let scores: Vec<f64> = survived.into_iter().map(|c| c as f64).collect();
        Ok((active_indices, scores))
    }

    /// L1-Based selection: genuine Lasso regression via coordinate descent
    /// on internally standardized features.
    ///
    /// Follows a decreasing regularization path from `alpha_max` (the
    /// smallest alpha at which the closed-form coordinate descent
    /// stationarity condition makes every coefficient exactly zero) down
    /// until at least `k` coefficients are nonzero, warm-starting each
    /// solve from the previous one. The `k` features with the largest
    /// resulting `|coefficient|` are returned, so the selection reflects
    /// genuine L1-induced sparsity rather than an OLS magnitude ranking.
    fn select_l1_based(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?;
        let n = target_vals.len();
        if n == 0 {
            return Err(Error::InvalidValue(
                "L1Based feature selection: target has no rows".into(),
            ));
        }

        // Standardize every feature to zero mean / unit population
        // variance so the L1 penalty applies uniformly regardless of each
        // feature's native scale. A constant column standardizes to all
        // zeros, which the penalty then correctly forces to beta=0.
        let mut x_std = vec![vec![0.0_f64; n_features]; n];
        for (j, name) in feature_names.iter().enumerate() {
            let col = x.get_column::<f64>(name)?;
            let vals = col.as_f64()?;
            if vals.len() != n {
                return Err(Error::DimensionMismatch(format!(
                    "L1Based feature selection: column '{}' has {} rows, expected {}",
                    name,
                    vals.len(),
                    n
                )));
            }
            let mean = vals.iter().sum::<f64>() / n as f64;
            let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.sqrt();
            let scale = if std > 1e-10 { std } else { 1.0 };
            for i in 0..n {
                x_std[i][j] = (vals[i] - mean) / scale;
            }
        }
        let y_mean = target_vals.iter().sum::<f64>() / n as f64;
        let y_c: Vec<f64> = target_vals.iter().map(|&v| v - y_mean).collect();

        // alpha_max: the smallest alpha for which the all-zero solution is
        // optimal (the standard closed-form Lasso stationarity bound at
        // beta=0: max_j |X_j' y| / n).
        let mut alpha_max = 0.0_f64;
        for j in 0..n_features {
            let mut rho = 0.0;
            for i in 0..n {
                rho += x_std[i][j] * y_c[i];
            }
            alpha_max = alpha_max.max((rho / n as f64).abs());
        }

        if alpha_max < 1e-12 {
            // No feature has any linear association with the target;
            // every alpha yields beta=0. There is no genuine signal to
            // rank by, so fall back to the first k columns in input order.
            let selected: Vec<usize> = (0..k).collect();
            return Ok((selected, vec![0.0; n_features]));
        }

        // Walk a log-spaced decreasing path of alphas from alpha_max down
        // to a small fraction of it, warm-starting each solve from the
        // previous solution, until at least k coefficients are nonzero.
        const N_ALPHAS: usize = 40;
        const MAX_ITER: usize = 300;
        const TOL: f64 = 1e-7;
        let log_max = alpha_max.ln();
        let log_min = (alpha_max * 1e-4).ln();

        let mut beta = vec![0.0_f64; n_features];
        for t in 0..N_ALPHAS {
            let frac = t as f64 / (N_ALPHAS - 1) as f64;
            let alpha = (log_max + frac * (log_min - log_max)).exp();
            beta = lasso_coordinate_descent(&x_std, &y_c, alpha, &beta, MAX_ITER, TOL);
            let nonzero = beta.iter().filter(|&&b| b.abs() > 1e-10).count();
            if nonzero >= k {
                break;
            }
        }

        let mut indexed: Vec<(usize, f64)> = beta.iter().map(|&b| b.abs()).enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        selected.sort_unstable();

        let scores: Vec<f64> = beta.iter().map(|b| b.abs()).collect();
        Ok((selected, scores))
    }

    /// Tree-Based selection: fit a real [`DecisionTreeRegressor`] on all
    /// features and rank by its feature importances.
    ///
    /// `DecisionTreeRegressor` does not (yet) populate its own
    /// `feature_importances()`, so importances are computed here directly
    /// from the fitted tree's public `nodes()` using the standard impurity
    /// -decrease-weighted formula (the same one
    /// `DecisionTreeClassifier::calculate_feature_importances` uses):
    /// each internal node contributes `(n_samples/total) * (impurity -
    /// weighted child impurity)` to its split feature, normalized to sum
    /// to 1 across all features.
    fn select_tree_based(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?.to_vec();

        let mut train_df = x.clone();
        train_df.add_column(
            target_col_name.clone(),
            Series::new(target_vals, Some(target_col_name.clone()))?,
        )?;

        let mut tree = DecisionTreeRegressor::new(DecisionTreeConfig::default());
        tree.fit(&train_df, target_col_name).map_err(|e| {
            Error::Computation(format!("TreeBased feature selection: fit failed: {}", e))
        })?;

        let importances = tree_feature_importances(&tree, n_features);

        let mut indexed: Vec<(usize, f64)> = importances.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        selected.sort_unstable();

        Ok((selected, importances))
    }

    /// Mutual Information selection: estimate MI between each feature and the target
    /// using histogram-based density estimation, then select top-k features.
    ///
    /// Bins are computed per-feature using the Sturges rule: k = ceil(log2(n) + 1).
    fn select_mutual_information(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?;

        let n = target_vals.len();
        if n == 0 {
            return Err(Error::InvalidValue(
                "MutualInformation feature selection: target has no rows".into(),
            ));
        }

        // Sturges rule for number of histogram bins
        let n_bins = ((n as f64).log2().ceil() as usize + 1).max(2).min(50);

        let mut mi_scores: Vec<f64> = Vec::with_capacity(n_features);

        // Precompute target histogram range
        let t_min = target_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = target_vals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let t_range = (t_max - t_min).max(1e-10);

        for feat_name in feature_names {
            let col = x.get_column::<f64>(feat_name)?;
            let feat_vals = col.as_f64()?;

            if feat_vals.len() != n {
                mi_scores.push(0.0);
                continue;
            }

            let f_min = feat_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let f_max = feat_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let f_range = (f_max - f_min).max(1e-10);

            // Build 2D joint histogram: n_bins × n_bins
            let mut joint_hist = vec![vec![0usize; n_bins]; n_bins];
            let mut feat_hist = vec![0usize; n_bins];
            let mut target_hist = vec![0usize; n_bins];

            for (&fv, &tv) in feat_vals.iter().zip(target_vals.iter()) {
                // bin_width = range / n_bins, matching the histogram
                // convention `model_selection`'s MI/chi2 estimators use.
                // Scaling by `n_bins - 1` (as before) compresses every
                // value into the first `n_bins - 1` bins and leaves the
                // top bin populated only by the exact maximum.
                let fi = (((fv - f_min) / f_range) * n_bins as f64).floor() as usize;
                let ti = (((tv - t_min) / t_range) * n_bins as f64).floor() as usize;
                let fi = fi.min(n_bins - 1);
                let ti = ti.min(n_bins - 1);
                joint_hist[fi][ti] += 1;
                feat_hist[fi] += 1;
                target_hist[ti] += 1;
            }

            // Compute MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
            let n_f = n as f64;
            let mut mi = 0.0_f64;
            for fi in 0..n_bins {
                for ti in 0..n_bins {
                    let pxy = joint_hist[fi][ti] as f64 / n_f;
                    let px = feat_hist[fi] as f64 / n_f;
                    let py = target_hist[ti] as f64 / n_f;
                    if pxy > 1e-12 && px > 1e-12 && py > 1e-12 {
                        mi += pxy * (pxy / (px * py)).ln();
                    }
                }
            }
            mi_scores.push(mi.max(0.0));
        }

        // Select top-k by MI score descending
        let mut indexed: Vec<(usize, f64)> = mi_scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        selected.sort_unstable();
        Ok((selected, mi_scores))
    }

    /// Select features by variance threshold. Returns the selected indices
    /// together with every feature's variance (its natural score).
    fn select_by_variance_threshold(
        &self,
        x: &DataFrame,
        threshold: f64,
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let feature_names = x.column_names();
        let mut selected_indices = Vec::new();
        let mut variances = Vec::with_capacity(feature_names.len());

        for (i, feature_name) in feature_names.iter().enumerate() {
            let col = x.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            if values.is_empty() {
                variances.push(0.0);
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variances.push(variance);

            if variance > threshold {
                selected_indices.push(i);
            }
        }

        Ok((selected_indices, variances))
    }
}
