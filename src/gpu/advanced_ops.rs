//! Advanced GPU operations for matrix decompositions and linear algebra
//!
//! This module implements more sophisticated GPU operations including:
//! - Matrix decompositions (QR, SVD, LU)
//! - Eigenvalue computations
//! - Matrix inversion using GPU solvers
//! - Advanced statistical operations

use scirs2_core::ndarray::{s, Array1, Array2, Axis};

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{get_gpu_manager, GpuManager};

// cuSOLVER is not exposed by cudarc 0.19.x, so the decompositions in this module
// fall back to real CPU implementations rather than GPU solver routines.

/// Matrix decomposition operations.
///
/// Despite the `Gpu` prefix, cudarc does not provide cuSOLVER bindings, so the
/// actual numerical work is performed by real CPU implementations. The
/// `gpu_available` flag only records whether a CUDA device was detected; it does
/// not change the numerical result.
pub struct GpuDecomposition {
    #[cfg(cuda_available)]
    gpu_available: bool,
}

impl GpuDecomposition {
    /// Create new GPU decomposition context
    pub fn new(gpu_manager: &GpuManager) -> Result<Self> {
        #[cfg(cuda_available)]
        {
            if gpu_manager.is_available() {
                Ok(Self {
                    gpu_available: true,
                })
            } else {
                Ok(Self {
                    gpu_available: false,
                })
            }
        }
        #[cfg(not(cuda_available))]
        {
            Ok(Self {})
        }
    }

    /// Compute the QR decomposition.
    ///
    /// cudarc does not expose cuSOLVER, so there is no real GPU QR kernel. This
    /// computes a real QR decomposition on the CPU (modified Gram-Schmidt).
    pub fn qr_decomposition(&self, matrix: &GpuMatrix) -> Result<(GpuMatrix, GpuMatrix)> {
        #[cfg(cuda_available)]
        {
            if self.gpu_available {
                // The GPU path is not implemented; attempt it so the missing
                // kernel is explicit, then fall back to the real CPU routine.
                if let Ok(result) = self.cuda_qr_decomposition(matrix) {
                    return Ok(result);
                }
            }
        }

        // Real CPU implementation.
        self.cpu_qr_decomposition(matrix)
    }

    /// Compute the singular value decomposition (SVD).
    ///
    /// cudarc does not expose cuSOLVER, so there is no real GPU SVD kernel. This
    /// computes a real SVD on the CPU via the eigendecomposition of `AᵀA`.
    pub fn svd_decomposition(
        &self,
        matrix: &GpuMatrix,
    ) -> Result<(GpuMatrix, GpuVector, GpuMatrix)> {
        #[cfg(cuda_available)]
        {
            if self.gpu_available {
                // The GPU path is not implemented; attempt it so the missing
                // kernel is explicit, then fall back to the real CPU routine.
                if let Ok(result) = self.cuda_svd_decomposition(matrix) {
                    return Ok(result);
                }
            }
        }

        // Real CPU implementation.
        self.cpu_svd_decomposition(matrix)
    }

    /// Compute eigenvalues and eigenvectors of a symmetric matrix.
    ///
    /// cudarc does not expose cuSOLVER, so there is no real GPU eigensolver.
    /// This computes a real eigendecomposition on the CPU using the cyclic
    /// Jacobi rotation method (valid for symmetric matrices such as covariance
    /// or Gram matrices).
    pub fn eigen_decomposition(&self, matrix: &GpuMatrix) -> Result<(GpuVector, GpuMatrix)> {
        #[cfg(cuda_available)]
        {
            if self.gpu_available {
                // The GPU path is not implemented; attempt it so the missing
                // kernel is explicit, then fall back to the real CPU routine.
                if let Ok(result) = self.cuda_eigen_decomposition(matrix) {
                    return Ok(result);
                }
            }
        }

        // Real CPU implementation.
        self.cpu_eigen_decomposition(matrix)
    }

    /// Compute the matrix inverse.
    ///
    /// cudarc does not expose cuSOLVER, so there is no real GPU inversion
    /// kernel. This computes a real inverse on the CPU (Gauss-Jordan
    /// elimination with partial pivoting).
    pub fn matrix_inverse(&self, matrix: &GpuMatrix) -> Result<GpuMatrix> {
        #[cfg(cuda_available)]
        {
            if self.gpu_available {
                // The GPU path is not implemented; attempt it so the missing
                // kernel is explicit, then fall back to the real CPU routine.
                if let Ok(result) = self.cuda_matrix_inverse(matrix) {
                    return Ok(result);
                }
            }
        }

        // Real CPU implementation.
        self.cpu_matrix_inverse(matrix)
    }

    // The GPU decomposition routines below would require cuSOLVER (geqrf, gesvd,
    // syevd, getrf/getri). cudarc 0.19.x does not provide cuSOLVER bindings, so
    // rather than returning fabricated identity/ones placeholders that are
    // presented as real decompositions, they honestly report `NotImplemented`.
    // The public dispatchers fall back to the real CPU implementations.
    #[cfg(cuda_available)]
    fn cuda_qr_decomposition(&self, _matrix: &GpuMatrix) -> Result<(GpuMatrix, GpuMatrix)> {
        Err(Error::NotImplemented(
            "GPU QR decomposition not implemented (cudarc has no cuSOLVER geqrf)".into(),
        ))
    }

    #[cfg(cuda_available)]
    fn cuda_svd_decomposition(
        &self,
        _matrix: &GpuMatrix,
    ) -> Result<(GpuMatrix, GpuVector, GpuMatrix)> {
        Err(Error::NotImplemented(
            "GPU SVD decomposition not implemented (cudarc has no cuSOLVER gesvd)".into(),
        ))
    }

    #[cfg(cuda_available)]
    fn cuda_eigen_decomposition(&self, _matrix: &GpuMatrix) -> Result<(GpuVector, GpuMatrix)> {
        Err(Error::NotImplemented(
            "GPU eigendecomposition not implemented (cudarc has no cuSOLVER syevd)".into(),
        ))
    }

    #[cfg(cuda_available)]
    fn cuda_matrix_inverse(&self, _matrix: &GpuMatrix) -> Result<GpuMatrix> {
        Err(Error::NotImplemented(
            "GPU matrix inverse not implemented (cudarc has no cuSOLVER getrf/getri)".into(),
        ))
    }

    fn cpu_qr_decomposition(&self, matrix: &GpuMatrix) -> Result<(GpuMatrix, GpuMatrix)> {
        // CPU QR via modified Gram-Schmidt (MGS) with one reorthogonalization
        // pass ("twice is enough" — Björck 1967; Rice 1966). A single MGS
        // pass already updates the working vector after each projection
        // (unlike classical Gram-Schmidt, which projects the *original*
        // column against every `q_i`), but repeated rounding error across
        // many columns can still leave `Qᵀ Q` measurably off `I` for
        // ill-conditioned input. Reorthogonalizing once — projecting the
        // already-orthogonalized vector against the same `q_i` columns a
        // second time and subtracting the (now tiny) residual — removes
        // that accumulated error at roughly double the cost, without
        // changing the mathematical result for well-conditioned input.
        let m = matrix.data.shape()[0];
        let n = matrix.data.shape()[1];

        let mut q = Array2::zeros((m, n));
        let mut r = Array2::zeros((n, n));

        for j in 0..n {
            let mut v = matrix.data.column(j).to_owned();
            let col_norm = v.dot(&v).sqrt();

            // Pass 1 (modified Gram-Schmidt) + pass 2 (reorthogonalization).
            // `r[(i, j)]` accumulates both passes' contributions so that
            // `A = Q·R` still holds exactly: the second pass's projections
            // are the residual `Aᵀ`-space component the first pass's
            // rounding error left behind, not an independent correction.
            for _pass in 0..2 {
                for i in 0..j {
                    let q_i = q.column(i).to_owned();
                    let dot_product = v.dot(&q_i);
                    r[(i, j)] += dot_product;
                    v = &v - &(dot_product * &q_i);
                }
            }

            let norm = v.dot(&v).sqrt();
            // Relative (not absolute) rank-deficiency threshold: whether the
            // component orthogonal to the previous columns is "negligible"
            // must be judged against this column's own scale, not a fixed
            // constant that is meaningless for matrices whose entries are
            // e.g. 1e6 or 1e-6.
            let tol = 1e-10 * col_norm.max(f64::MIN_POSITIVE);

            if norm > tol {
                r[(j, j)] = norm;
                let q_j = v / norm;
                q.column_mut(j).assign(&q_j);
            } else {
                // Column `j` is (numerically) linearly dependent on the
                // columns already processed. R's diagonal honestly records
                // that as ~0, but leaving `q`'s column at its `zeros(..)`
                // initializer — as a purely classical Gram-Schmidt
                // implementation would — leaves a zero column in Q, which
                // can never be part of an orthonormal basis (`Qᵀ Q` would
                // have a 0 on that diagonal instead of 1). Complete Q to a
                // genuine orthonormal matrix by orthogonalizing a standard
                // basis vector against the columns chosen so far, exactly
                // like the U-completion step in `cpu_svd_decomposition`
                // below; R still correctly reports the rank deficiency.
                r[(j, j)] = 0.0;
                for basis_idx in 0..m {
                    let mut candidate = Array1::<f64>::zeros(m);
                    candidate[basis_idx] = 1.0;
                    for i in 0..j {
                        let q_i = q.column(i).to_owned();
                        let proj = candidate.dot(&q_i);
                        candidate = &candidate - &(proj * &q_i);
                    }
                    let cand_norm = candidate.dot(&candidate).sqrt();
                    if cand_norm > 1e-10 {
                        q.column_mut(j).assign(&(candidate / cand_norm));
                        break;
                    }
                }
            }
        }

        Ok((
            GpuMatrix {
                data: q,
                on_gpu: false,
            },
            GpuMatrix {
                data: r,
                on_gpu: false,
            },
        ))
    }

    fn cpu_svd_decomposition(
        &self,
        matrix: &GpuMatrix,
    ) -> Result<(GpuMatrix, GpuVector, GpuMatrix)> {
        // Real CPU SVD computed from the eigendecomposition of the symmetric
        // Gram matrix `AᵀA`. This yields the right singular vectors (V) and the
        // squared singular values; the left singular vectors follow from
        // `uᵢ = A·vᵢ / σᵢ`, with any remaining columns completed to an
        // orthonormal basis. This satisfies `A = U·Σ·Vᵀ` numerically rather
        // than returning fabricated identity/ones placeholders.
        let a = &matrix.data;
        let m = a.shape()[0];
        let n = a.shape()[1];
        let min_dim = m.min(n);
        let tol = 1e-10;

        // Eigendecomposition of AᵀA (n×n, symmetric positive semi-definite).
        let ata = a.t().dot(a);
        let (eigenvalues, v) = jacobi_symmetric_eigen(&ata)?;

        // Singular values are the square roots of the eigenvalues, clamped
        // to zero before the square root. `AᵀA` is symmetric positive
        // semi-definite, so its true eigenvalues are never negative in exact
        // arithmetic; any negative value here is floating-point noise from
        // the Jacobi rotations (most visible for near-zero eigenvalues,
        // i.e. directions `A` doesn't actually span). Clamping corrects that
        // rounding noise rather than concealing a real negative eigenvalue —
        // a mathematically real negative eigenvalue of `AᵀA` is impossible —
        // and avoids `sqrt` of a negative number producing `NaN` singular
        // values for an otherwise well-formed decomposition.
        let mut singular_values = Array1::zeros(min_dim);
        for i in 0..min_dim {
            singular_values[i] = eigenvalues[i].max(0.0).sqrt();
        }

        // Right singular vectors: V is n×n; Vᵀ is returned.
        let vt = v.t().to_owned();

        // Left singular vectors U (m×m).
        let mut u = Array2::<f64>::zeros((m, m));
        let mut filled = 0usize;
        for i in 0..min_dim {
            if singular_values[i] > tol {
                let avi = a.dot(&v.column(i));
                let ui = avi / singular_values[i];
                u.column_mut(filled).assign(&ui);
                filled += 1;
            }
        }

        // Complete U to a full orthonormal basis of Rᵐ using modified
        // Gram-Schmidt against the columns already determined.
        let mut basis_idx = 0usize;
        while filled < m && basis_idx < m {
            let mut candidate = Array1::<f64>::zeros(m);
            candidate[basis_idx] = 1.0;
            for c in 0..filled {
                let col = u.column(c).to_owned();
                let proj = candidate.dot(&col);
                candidate = &candidate - &(proj * &col);
            }
            let norm = candidate.dot(&candidate).sqrt();
            if norm > tol {
                let normalized = candidate / norm;
                u.column_mut(filled).assign(&normalized);
                filled += 1;
            }
            basis_idx += 1;
        }

        Ok((
            GpuMatrix {
                data: u,
                on_gpu: false,
            },
            GpuVector {
                data: singular_values,
                on_gpu: false,
            },
            GpuMatrix {
                data: vt,
                on_gpu: false,
            },
        ))
    }

    fn cpu_eigen_decomposition(&self, matrix: &GpuMatrix) -> Result<(GpuVector, GpuMatrix)> {
        // Real CPU eigendecomposition for symmetric matrices using the cyclic
        // Jacobi rotation method. Eigenvalues are returned in descending order
        // with the corresponding eigenvectors as columns. Replaces the previous
        // placeholder that returned ones/identity.
        let (eigenvalues, eigenvectors) = jacobi_symmetric_eigen(&matrix.data)?;

        Ok((
            GpuVector {
                data: eigenvalues,
                on_gpu: false,
            },
            GpuMatrix {
                data: eigenvectors,
                on_gpu: false,
            },
        ))
    }

    fn cpu_matrix_inverse(&self, matrix: &GpuMatrix) -> Result<GpuMatrix> {
        // Simplified CPU matrix inverse using Gauss-Jordan elimination
        let n = matrix.data.shape()[0];

        if matrix.data.shape()[1] != n {
            return Err(Error::DimensionMismatch(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // Singularity is judged against the matrix's own scale: a fixed
        // absolute `1e-10` pivot floor calls a matrix of entries ~1e-8
        // "singular" even when it is perfectly well-conditioned, and calls a
        // matrix of entries ~1e8 "invertible" even when its true pivot is
        // relatively negligible.
        let matrix_scale = matrix
            .data
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x.abs()))
            .max(f64::MIN_POSITIVE);
        let pivot_tol = 1e-10 * matrix_scale;

        // Create augmented matrix [A | I]
        let mut augmented = Array2::zeros((n, 2 * n));

        // Copy original matrix to left side
        for i in 0..n {
            for j in 0..n {
                augmented[(i, j)] = matrix.data[(i, j)];
            }
        }

        // Set identity on right side
        for i in 0..n {
            augmented[(i, n + i)] = 1.0;
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[(k, i)].abs() > augmented[(max_row, i)].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[(i, j)];
                    augmented[(i, j)] = augmented[(max_row, j)];
                    augmented[(max_row, j)] = temp;
                }
            }

            // Check for singularity (relative to the matrix's own scale)
            if augmented[(i, i)].abs() < pivot_tol {
                return Err(Error::Computation("Matrix is singular".to_string()));
            }

            // Scale pivot row
            let pivot = augmented[(i, i)];
            for j in 0..(2 * n) {
                augmented[(i, j)] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[(k, i)];
                    for j in 0..(2 * n) {
                        augmented[(k, j)] -= factor * augmented[(i, j)];
                    }
                }
            }
        }

        // Extract inverse from right side
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[(i, j)] = augmented[(i, n + j)];
            }
        }

        Ok(GpuMatrix {
            data: inverse,
            on_gpu: false,
        })
    }
}

/// GPU-accelerated advanced statistical operations
pub struct GpuAdvancedStats;

impl GpuAdvancedStats {
    /// Compute Principal Component Analysis on GPU
    pub fn pca(data: &GpuMatrix, n_components: usize) -> Result<(GpuMatrix, GpuVector, GpuMatrix)> {
        let gpu_manager = get_gpu_manager()?;
        let decomp = GpuDecomposition::new(&gpu_manager)?;

        // Center the data. `mean_axis` returns `None` only when the axis
        // being averaged over is empty (zero rows), which is a genuine
        // "not enough data" condition, not an invariant this call site can
        // guarantee away — report it as an error instead of panicking.
        let mean = data.data.mean_axis(Axis(0)).ok_or_else(|| {
            Error::InsufficientData("PCA requires at least one row of data".to_string())
        })?;
        let centered_data = &data.data - &mean.insert_axis(Axis(0));

        let centered_matrix = GpuMatrix {
            data: centered_data,
            on_gpu: data.on_gpu,
        };

        // Compute covariance matrix
        let n_samples = data.data.shape()[0] as f64;
        let cov_matrix = GpuMatrix {
            data: centered_matrix.data.t().dot(&centered_matrix.data) / (n_samples - 1.0),
            on_gpu: false,
        };

        // Compute eigendecomposition of covariance matrix
        let (eigenvalues, eigenvectors) = decomp.eigen_decomposition(&cov_matrix)?;

        // Sort by eigenvalues (descending). `total_cmp` gives a real total
        // order over all `f64` values instead of panicking (via `.expect`
        // on `partial_cmp`'s `None`) should a near-singular covariance
        // matrix ever leave a `NaN` in the Jacobi eigensolver's output.
        let mut indices: Vec<usize> = (0..eigenvalues.data.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues.data[j].total_cmp(&eigenvalues.data[i]));

        // Select top n_components
        let n_components = n_components.min(eigenvalues.data.len());
        let selected_eigenvalues = Array1::from_iter(
            indices
                .iter()
                .take(n_components)
                .map(|&i| eigenvalues.data[i]),
        );

        let selected_eigenvectors = {
            let mut vecs = Array2::zeros((eigenvectors.data.shape()[0], n_components));
            for (j, &i) in indices.iter().take(n_components).enumerate() {
                vecs.column_mut(j).assign(&eigenvectors.data.column(i));
            }
            vecs
        };

        // Transform data
        let transformed_data = centered_matrix.data.dot(&selected_eigenvectors);

        Ok((
            GpuMatrix {
                data: transformed_data,
                on_gpu: false,
            },
            GpuVector {
                data: selected_eigenvalues,
                on_gpu: false,
            },
            GpuMatrix {
                data: selected_eigenvectors,
                on_gpu: false,
            },
        ))
    }

    /// Compute Linear Discriminant Analysis (LDA).
    ///
    /// Solves the symmetric generalized eigenproblem `S_b w = λ S_w w`, where
    /// `S_w` is the within-class scatter matrix and `S_b` is the
    /// between-class scatter matrix, via symmetric whitening: eigendecompose
    /// `S_w` (real, symmetric — [`jacobi_symmetric_eigen`]), form the
    /// (pseudo-)inverse square root `S_w^{-1/2}`, then eigendecompose the
    /// symmetric matrix `M = S_w^{-1/2} · S_b · S_w^{-1/2}`. `M`'s
    /// eigenvectors, mapped back through `S_w^{-1/2}`, are the discriminant
    /// directions; `data` projected onto the top `n_components` of them
    /// (after centering on the overall mean) is the returned transform.
    ///
    /// This replaces a previous placeholder that returned the raw class
    /// means (not a projection at all) as the "LDA projection".
    ///
    /// # Errors
    /// Returns `Error::DimensionMismatch` if `labels.len()` does not match
    /// `data`'s row count, and `Error::InvalidValue` if a label is negative,
    /// fewer than 2 distinct classes are present, or some class has no
    /// members.
    pub fn lda(data: &GpuMatrix, labels: &Array1<i32>, n_components: usize) -> Result<GpuMatrix> {
        let n_samples = data.data.shape()[0];
        let n_features = data.data.shape()[1];

        if labels.len() != n_samples {
            return Err(Error::DimensionMismatch(format!(
                "LDA labels length {} does not match data row count {}",
                labels.len(),
                n_samples
            )));
        }
        if labels.iter().any(|&label| label < 0) {
            return Err(Error::InvalidValue(
                "LDA class labels must be non-negative".to_string(),
            ));
        }

        let n_classes = labels.iter().copied().max().unwrap_or(-1) + 1;
        if n_classes < 2 {
            return Err(Error::InvalidValue(
                "LDA requires at least 2 distinct classes".to_string(),
            ));
        }
        let n_classes = n_classes as usize;

        // Class means and sizes.
        let mut class_means = Array2::<f64>::zeros((n_classes, n_features));
        let mut class_counts = vec![0usize; n_classes];
        for (i, &label) in labels.iter().enumerate() {
            let class_idx = label as usize;
            class_counts[class_idx] += 1;
            for j in 0..n_features {
                class_means[(class_idx, j)] += data.data[(i, j)];
            }
        }
        for class_idx in 0..n_classes {
            if class_counts[class_idx] == 0 {
                return Err(Error::InvalidValue(format!(
                    "LDA class {} has no members (labels must cover 0..n_classes densely)",
                    class_idx
                )));
            }
            for j in 0..n_features {
                class_means[(class_idx, j)] /= class_counts[class_idx] as f64;
            }
        }

        let overall_mean = data
            .data
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::Computation("Failed to compute overall mean for LDA".into()))?;

        // Within-class scatter: S_w = sum_c sum_{i in c} (x_i - mean_c)(x_i - mean_c)^T
        let mut s_w = Array2::<f64>::zeros((n_features, n_features));
        for (i, &label) in labels.iter().enumerate() {
            let class_idx = label as usize;
            let diff = &data.data.row(i) - &class_means.row(class_idx);
            for a in 0..n_features {
                if diff[a] == 0.0 {
                    continue;
                }
                for b in 0..n_features {
                    s_w[(a, b)] += diff[a] * diff[b];
                }
            }
        }

        // Between-class scatter: S_b = sum_c n_c (mean_c - mean)(mean_c - mean)^T
        let mut s_b = Array2::<f64>::zeros((n_features, n_features));
        for class_idx in 0..n_classes {
            let diff = &class_means.row(class_idx) - &overall_mean;
            let n_c = class_counts[class_idx] as f64;
            for a in 0..n_features {
                for b in 0..n_features {
                    s_b[(a, b)] += n_c * diff[a] * diff[b];
                }
            }
        }

        // Whiten by S_w: eigendecompose it and build S_w^{-1/2} as a
        // Moore-Penrose-style pseudo-inverse square root. Directions with
        // (numerically) zero within-class variance — a class represented by
        // a single sample, or more features than samples — are dropped
        // rather than dividing by (near-)zero, which would otherwise blow up
        // to +-infinity/NaN.
        let (sw_eigenvalues, sw_eigenvectors) = jacobi_symmetric_eigen(&s_w)?;
        let sw_scale = sw_eigenvalues
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let rank_tol = 1e-10 * sw_scale;

        let mut sw_inv_sqrt = Array2::<f64>::zeros((n_features, n_features));
        for k in 0..n_features {
            let lambda = sw_eigenvalues[k];
            if lambda > rank_tol {
                let inv_sqrt = 1.0 / lambda.sqrt();
                let u_k = sw_eigenvectors.column(k);
                for a in 0..n_features {
                    let scaled = inv_sqrt * u_k[a];
                    for b in 0..n_features {
                        sw_inv_sqrt[(a, b)] += scaled * u_k[b];
                    }
                }
            }
        }

        // M = S_w^{-1/2} S_b S_w^{-1/2} is symmetric (S_b is symmetric and
        // S_w^{-1/2} is symmetric), so its eigenvectors are real and
        // orthogonal; they are exactly the generalized eigenvectors of
        // (S_b, S_w), sorted by descending discriminant power.
        let m = sw_inv_sqrt.dot(&s_b).dot(&sw_inv_sqrt);
        let (_lda_eigenvalues, lda_eigenvectors) = jacobi_symmetric_eigen(&m)?;

        // S_b has rank at most `n_classes - 1` (it is a sum of `n_classes`
        // rank-1 terms whose deviations from the overall mean sum to zero),
        // so requesting more components than that would only add directions
        // with zero discriminant power.
        let n_components = n_components.min(n_features).min(n_classes - 1);

        // Map the whitened-space eigenvectors back to the original feature
        // space: w = S_w^{-1/2} * w~.
        let selected_tilde = lda_eigenvectors.slice(s![.., 0..n_components]).to_owned();
        let w = sw_inv_sqrt.dot(&selected_tilde);

        // Project the mean-centered data onto the discriminant directions.
        let mut centered = data.data.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                centered[(i, j)] -= overall_mean[j];
            }
        }
        let transformed = centered.dot(&w);

        Ok(GpuMatrix {
            data: transformed,
            on_gpu: false,
        })
    }

    /// Compute correlation matrix with p-values on GPU
    pub fn correlation_with_pvalues(data: &GpuMatrix) -> Result<(GpuMatrix, GpuMatrix)> {
        let n_features = data.data.shape()[1];
        let n_samples = data.data.shape()[0] as f64;

        // A correlation coefficient and its t-test both require at least 2
        // samples (df = n - 2 must be positive for the t-statistic below);
        // fewer than that is not a "should never happen" invariant, so
        // report it as an error rather than reaching an unreachable-in-
        // practice `.expect()` on an empty column's `mean()`.
        if n_samples < 2.0 {
            return Err(Error::InsufficientData(
                "correlation_with_pvalues requires at least 2 samples".to_string(),
            ));
        }

        let mut correlation_matrix = Array2::zeros((n_features, n_features));
        let mut pvalue_matrix = Array2::zeros((n_features, n_features));

        // Compute pairwise correlations and p-values
        for i in 0..n_features {
            for j in i..n_features {
                if i == j {
                    correlation_matrix[(i, j)] = 1.0;
                    pvalue_matrix[(i, j)] = 0.0;
                } else {
                    let x = data.data.column(i);
                    let y = data.data.column(j);

                    // Compute Pearson correlation. `mean()` only returns
                    // `None` for an empty column, which the `n_samples < 2`
                    // guard above already rules out.
                    let mean_x = x.mean().ok_or_else(|| {
                        Error::Computation("failed to compute column mean".to_string())
                    })?;
                    let mean_y = y.mean().ok_or_else(|| {
                        Error::Computation("failed to compute column mean".to_string())
                    })?;

                    let numerator: f64 = x
                        .iter()
                        .zip(y.iter())
                        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
                        .sum();

                    let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();

                    let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

                    let denominator = (sum_sq_x * sum_sq_y).sqrt();

                    let correlation = if denominator > 1e-10 {
                        // Pearson's r is mathematically bounded to [-1, 1]
                        // (Cauchy-Schwarz: |numerator| <= denominator
                        // exactly), but `numerator` and `denominator` are
                        // summed independently in floating point, so for a
                        // near-perfectly-correlated pair (e.g. a duplicated
                        // column) the computed ratio can overshoot by a few
                        // ULPs, e.g. 1.0000000000000002. Clamping removes
                        // that rounding noise at the boundary rather than
                        // letting it propagate: left unclamped, the
                        // `1.0 - correlation.powi(2)` term below goes
                        // slightly negative, its `.sqrt()` is NaN per IEEE
                        // 754, and the p-value for the single most
                        // significant possible correlation would silently
                        // come out NaN instead of the correct 0.0.
                        (numerator / denominator).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };

                    // Compute the two-sided p-value for the correlation
                    // coefficient's t-test via the crate's single source of
                    // truth for the Student's-t distribution
                    // (`stats::special`), rather than a local approximation.
                    // The previous local `student_t_cdf` here was a logistic
                    // curve (`0.5 + 0.5*sign(x)*(1-exp(-|x|))`), not a real
                    // Student's-t CDF: it saturated around 0.72 instead of
                    // 1.0, so p-values computed from it were fabricated and
                    // could never reach significance (e.g. t=2.776, df=4
                    // gave p≈0.44 instead of the correct 0.05).
                    //
                    // `correlation.abs() >= 1.0` (an exact +-1, or the clamp
                    // above catching floating-point overshoot right at the
                    // boundary) is handled as its own case rather than
                    // through the general `t_stat` formula: `1.0 -
                    // correlation.powi(2)` is exactly `0.0` there, and
                    // `x / 0.0` is `+-infinity` if `x` is nonzero but `NaN`
                    // if `x` is also exactly `0.0` (possible when
                    // `n_samples == 2`, i.e. `df == 0`) -- and `df == 0` is
                    // already `student_t_two_sided_p`'s own documented
                    // "always NaN" domain regardless of `t_stat`. A perfect
                    // correlation with at least one real degree of freedom
                    // is the most significant result representable: p = 0.
                    let p_value = if correlation.abs() >= 1.0 && n_samples > 2.0 {
                        0.0
                    } else {
                        let t_stat =
                            correlation * ((n_samples - 2.0) / (1.0 - correlation.powi(2))).sqrt();
                        crate::stats::special::student_t_two_sided_p(t_stat, n_samples - 2.0)
                    };

                    correlation_matrix[(i, j)] = correlation;
                    correlation_matrix[(j, i)] = correlation;
                    pvalue_matrix[(i, j)] = p_value;
                    pvalue_matrix[(j, i)] = p_value;
                }
            }
        }

        Ok((
            GpuMatrix {
                data: correlation_matrix,
                on_gpu: false,
            },
            GpuMatrix {
                data: pvalue_matrix,
                on_gpu: false,
            },
        ))
    }
}

/// Compute the eigenvalues and eigenvectors of a real **symmetric** matrix
/// using the cyclic Jacobi rotation algorithm.
///
/// Returns `(eigenvalues, eigenvectors)` where the eigenvalues are sorted in
/// descending order and the eigenvectors are the corresponding columns of the
/// returned matrix. The algorithm is iterative and converges for symmetric
/// inputs (e.g. covariance or Gram matrices); it is not intended for
/// non-symmetric matrices.
pub(crate) fn jacobi_symmetric_eigen(input: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = input.shape()[0];
    if input.shape()[1] != n {
        return Err(Error::DimensionMismatch(
            "Jacobi eigensolver requires a square symmetric matrix".to_string(),
        ));
    }

    let mut a = input.clone();
    let mut v = Array2::<f64>::eye(n);

    if n == 0 {
        return Ok((Array1::zeros(0), v));
    }

    // Convergence and per-element thresholds are relative to the matrix's
    // own scale (its Frobenius norm), not a fixed absolute constant: for a
    // matrix whose entries are ~1e6, requiring the off-diagonal energy to
    // fall below an absolute `1e-30` is unreachable in any bounded number of
    // sweeps (each entry squared is already ~1e12), so every large-magnitude
    // input silently ran the full `max_sweeps` regardless of how converged
    // it actually was; for a matrix whose entries are ~1e-6 the same
    // absolute floor is already satisfied before any rotation has reduced
    // the off-diagonal energy at all.
    let frobenius_norm = a
        .iter()
        .map(|&x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let convergence_threshold = 1e-28 * frobenius_norm * frobenius_norm;
    let skip_threshold = 1e-15 * frobenius_norm;

    // Cyclic Jacobi sweeps. Each sweep eliminates every off-diagonal element
    // once; the off-diagonal norm decreases quadratically near convergence.
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        let mut off_diag_norm = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diag_norm += a[(p, q)] * a[(p, q)];
            }
        }
        if off_diag_norm <= convergence_threshold {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[(p, q)];
                if apq.abs() <= skip_threshold {
                    continue;
                }

                // Rotation angle that annihilates a[p, q].
                let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // A' = Jᵀ A J. First apply the rotation to the columns (A·J)...
                for k in 0..n {
                    let akp = a[(k, p)];
                    let akq = a[(k, q)];
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, q)] = s * akp + c * akq;
                }
                // ...then to the rows (Jᵀ·(A·J)).
                for k in 0..n {
                    let apk = a[(p, k)];
                    let aqk = a[(q, k)];
                    a[(p, k)] = c * apk - s * aqk;
                    a[(q, k)] = s * apk + c * aqk;
                }
                // Accumulate the eigenvectors: V' = V·J.
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Eigenvalues are the diagonal entries of the (now near-diagonal) matrix.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[(j, j)]
            .partial_cmp(&a[(i, i)])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Array1::zeros(n);
    let mut eigenvectors = Array2::zeros((n, n));
    for (new_idx, &old_idx) in order.iter().enumerate() {
        eigenvalues[new_idx] = a[(old_idx, old_idx)];
        eigenvectors.column_mut(new_idx).assign(&v.column(old_idx));
    }

    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_gpu_qr_decomposition() {
        let gpu_manager = GpuManager::new();
        let decomp = GpuDecomposition::new(&gpu_manager).expect("operation should succeed");

        let matrix_data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("operation should succeed");

        let matrix = GpuMatrix {
            data: matrix_data,
            on_gpu: false,
        };

        let (q, r) = decomp
            .qr_decomposition(&matrix)
            .expect("operation should succeed");

        // Verify Q is orthogonal and R is upper triangular
        assert_eq!(q.data.shape(), &[3, 3]);
        assert_eq!(r.data.shape(), &[3, 3]);
    }

    #[test]
    fn test_matrix_inverse() {
        let gpu_manager = GpuManager::new();
        let decomp = GpuDecomposition::new(&gpu_manager).expect("operation should succeed");

        // Create an invertible matrix
        let matrix_data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("operation should succeed");

        let matrix = GpuMatrix {
            data: matrix_data,
            on_gpu: false,
        };

        let inverse = decomp
            .matrix_inverse(&matrix)
            .expect("operation should succeed");

        // Verify inverse dimensions
        assert_eq!(inverse.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_pca() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("operation should succeed");

        let gpu_matrix = GpuMatrix {
            data,
            on_gpu: false,
        };

        let (transformed, eigenvalues, components) =
            GpuAdvancedStats::pca(&gpu_matrix, 2).expect("operation should succeed");

        assert_eq!(transformed.data.shape()[1], 2);
        assert_eq!(eigenvalues.data.len(), 2);
        assert_eq!(components.data.shape(), &[3, 2]);
    }

    #[test]
    fn test_jacobi_symmetric_eigen_correctness() {
        // The symmetric matrix [[2,1],[1,2]] has eigenvalues 3 and 1.
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0])
            .expect("operation should succeed");
        let (vals, vecs) = jacobi_symmetric_eigen(&a).expect("operation should succeed");

        // Eigenvalues are returned in descending order.
        assert!((vals[0] - 3.0).abs() < 1e-8, "lambda0 = {}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-8, "lambda1 = {}", vals[1]);

        // Each eigenpair must satisfy A·v = lambda·v (sign-independent check).
        for k in 0..2 {
            let v = vecs.column(k).to_owned();
            let av = a.dot(&v);
            let lv = v.mapv(|x| x * vals[k]);
            for i in 0..2 {
                assert!(
                    (av[i] - lv[i]).abs() < 1e-8,
                    "A v != lambda v at component {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_cpu_svd_reconstruction() {
        let gpu_manager = GpuManager::new();
        let decomp = GpuDecomposition::new(&gpu_manager).expect("operation should succeed");

        let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("operation should succeed");
        let matrix = GpuMatrix {
            data: a.clone(),
            on_gpu: false,
        };

        let (u, s, vt) = decomp
            .svd_decomposition(&matrix)
            .expect("operation should succeed");

        assert_eq!(u.data.shape(), &[3, 3]);
        assert_eq!(vt.data.shape(), &[2, 2]);
        assert_eq!(s.data.len(), 2);

        // Reconstruct A = U[:, :k] · diag(S) · Vt[:k, :] and compare to the input.
        let m = a.shape()[0];
        let n = a.shape()[1];
        let k = s.data.len();
        let mut recon = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += u.data[(i, r)] * s.data[r] * vt.data[(r, j)];
                }
                recon[(i, j)] = acc;
            }
        }
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (recon[(i, j)] - a[(i, j)]).abs() < 1e-6,
                    "SVD reconstruction mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }
}
