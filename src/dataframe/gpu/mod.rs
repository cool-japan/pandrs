use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;

/// GPU acceleration extensions for DataFrame
pub trait DataFrameGpuExt {
    /// Apply GPU acceleration to a DataFrame
    fn gpu_accelerate(&self) -> Result<Self>
    where
        Self: Sized;

    /// GPU-accelerated correlation matrix
    fn gpu_corr(&self, columns: &[&str]) -> Result<Self>
    where
        Self: Sized;

    /// GPU-accelerated linear regression
    fn gpu_linear_regression(&self, target: &str, features: &[&str]) -> Result<Self>
    where
        Self: Sized;

    /// GPU-accelerated Principal Component Analysis (PCA)
    fn gpu_pca(&self, columns: &[&str], n_components: usize) -> Result<Self>
    where
        Self: Sized;

    /// GPU-accelerated k-means clustering
    fn gpu_kmeans(&self, columns: &[&str], k: usize, max_iterations: usize) -> Result<Self>
    where
        Self: Sized;
}

// HONESTY NOTE: there is no real CUDA kernel behind any of these entry
// points (see the module-level notes throughout `crate::gpu`). `gpu_corr`,
// `gpu_linear_regression`, `gpu_pca`, and `gpu_kmeans` used to all honestly
// return `NotImplemented` — but the Python bindings
// (`py_bindings::py_gpu::PyOptimizedDataFrame`) implement these exact same
// operations with real CPU math (`stats::descriptive::pearson_correlation`,
// `OptimizedDataFrame::linear_regression`, `ml::PCA`, `ml::KMeans`), so
// Python callers got real results while Rust callers of this trait always
// got an error for the identical operation. These now route through that
// same real math (`crate::gpu::cpu_math`, `crate::stats::linear_regression`,
// `crate::ml::PCA`/`crate::ml::KMeans` — the latter two are exactly what
// `py_gpu.rs` calls) instead of duplicating it. `gpu_accelerate` alone stays
// `NotImplemented`: unlike the operations above, there is no CPU computation
// it could honestly perform instead — GPU dispatch in this crate happens
// per-operation (in `gpu_corr` et al., and in `GpuMatrix`), not as a
// one-shot transform of the whole frame ahead of time.
impl DataFrameGpuExt for DataFrame {
    fn gpu_accelerate(&self) -> Result<Self> {
        // There is no real GPU kernel behind this entry point. Returning the
        // input unchanged while claiming "GPU acceleration" would be dishonest,
        // so report that the operation is not implemented.
        Err(Error::NotImplemented(
            "GPU acceleration for DataFrame not implemented (no real CUDA kernel)".into(),
        ))
    }

    fn gpu_corr(&self, columns: &[&str]) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::InvalidValue(
                "gpu_corr requires at least one column".to_string(),
            ));
        }

        // `get_column_numeric_values` uses `f64::NAN` as this crate's
        // missing-value sentinel for a plain (non-`NA`-wrapped) numeric
        // `Series` (see `Series::<f64>::sum`'s doc comment for the same
        // convention); bridge that to `Option<f64>` for the shared listwise
        // correlation helper.
        let mut data: Vec<Vec<Option<f64>>> = Vec::with_capacity(columns.len());
        for &col in columns {
            let raw = self.get_column_numeric_values(col)?;
            data.push(
                raw.into_iter()
                    .map(|v| if v.is_nan() { None } else { Some(v) })
                    .collect(),
            );
        }

        let matrix = crate::gpu::cpu_math::correlation_matrix_listwise(&data)?;

        // Same output shape as `DataFrame::corr_matrix`: one column per
        // input column, row-indexed by the same names.
        let mut result = DataFrame::new();
        for (j, &col_name) in columns.iter().enumerate() {
            let col_values: Vec<f64> = (0..columns.len()).map(|i| matrix[[i, j]]).collect();
            result.add_column(
                col_name.to_string(),
                crate::series::Series::new(col_values, Some(col_name.to_string()))?,
            )?;
        }
        let row_labels: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        result.set_index(crate::index::Index::new(row_labels)?)?;

        Ok(result)
    }

    fn gpu_linear_regression(&self, target: &str, features: &[&str]) -> Result<Self> {
        let fit = crate::stats::linear_regression(self, target, features)?;

        // One row per term (the intercept, then each feature in order),
        // with the fit's overall R^2/adjusted R^2 broadcast to every row so
        // the whole result is a single, self-contained table (the trait
        // signature returns one `DataFrame`, so there's no second return
        // slot for scalar summary statistics).
        let mut terms: Vec<String> = Vec::with_capacity(features.len() + 1);
        let mut coefficients: Vec<f64> = Vec::with_capacity(features.len() + 1);
        terms.push("intercept".to_string());
        coefficients.push(fit.intercept);
        for (&name, &coef) in features.iter().zip(fit.coefficients.iter()) {
            terms.push(name.to_string());
            coefficients.push(coef);
        }
        let r_squared_col = vec![fit.r_squared; terms.len()];
        let adj_r_squared_col = vec![fit.adj_r_squared; terms.len()];

        let mut result = DataFrame::new();
        result.add_column(
            "term".to_string(),
            crate::series::Series::new(terms, Some("term".to_string()))?,
        )?;
        result.add_column(
            "coefficient".to_string(),
            crate::series::Series::new(coefficients, Some("coefficient".to_string()))?,
        )?;
        result.add_column(
            "r_squared".to_string(),
            crate::series::Series::new(r_squared_col, Some("r_squared".to_string()))?,
        )?;
        result.add_column(
            "adj_r_squared".to_string(),
            crate::series::Series::new(adj_r_squared_col, Some("adj_r_squared".to_string()))?,
        )?;

        Ok(result)
    }

    fn gpu_pca(&self, columns: &[&str], n_components: usize) -> Result<Self> {
        use crate::ml::UnsupervisedModel;

        let mut subset = DataFrame::new();
        for &col_name in columns {
            let values = self.get_column_numeric_values(col_name)?;
            subset.add_column(
                col_name.to_string(),
                crate::series::Series::new(values, Some(col_name.to_string()))?,
            )?;
        }

        // Same model `py_gpu.rs::gpu_pca` uses.
        let mut pca = crate::ml::PCA::new(n_components, false);
        pca.fit(&subset)?;
        pca.transform(&subset)
    }

    fn gpu_kmeans(&self, columns: &[&str], k: usize, max_iterations: usize) -> Result<Self> {
        use crate::ml::UnsupervisedModel;

        let mut subset = DataFrame::new();
        for &col_name in columns {
            let values = self.get_column_numeric_values(col_name)?;
            subset.add_column(
                col_name.to_string(),
                crate::series::Series::new(values, Some(col_name.to_string()))?,
            )?;
        }

        // Same model `py_gpu.rs::gpu_kmeans` uses.
        let mut km = crate::ml::KMeans::new(k)
            .max_iter(max_iterations)
            .with_columns(columns.iter().map(|s| s.to_string()).collect());
        km.fit(&subset)?;

        let labels = km.labels.ok_or_else(|| {
            Error::Computation("KMeans fit did not produce cluster labels".to_string())
        })?;
        let labels_i64: Vec<i64> = labels.into_iter().map(|l| l as i64).collect();

        let mut result = subset;
        result.add_column(
            "cluster".to_string(),
            crate::series::Series::new(labels_i64, Some("cluster".to_string()))?,
        )?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("b".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");
        df
    }

    #[test]
    fn gpu_corr_matches_real_correlation() {
        let df = sample_df();
        let corr = df.gpu_corr(&["a", "b"]).expect("operation should succeed");
        let b_col = corr
            .get_column_numeric_values("b")
            .expect("operation should succeed");
        // a and b are perfectly linearly correlated.
        assert!((b_col[0] - 1.0).abs() < 1e-9, "corr(a,b) = {}", b_col[0]);
    }

    #[test]
    fn gpu_accelerate_is_honest_not_implemented() {
        let df = sample_df();
        assert!(df.gpu_accelerate().is_err());
    }
}
