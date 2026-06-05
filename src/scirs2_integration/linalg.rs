//! Linear algebra bridge using SciRS2's implementations.
//!
//! All types and functions in this module are gated behind the `scirs2` feature flag.

#[cfg(feature = "scirs2")]
use crate::core::error::{Error, Result};
#[cfg(feature = "scirs2")]
use crate::dataframe::DataFrame;
#[cfg(feature = "scirs2")]
use crate::scirs2_integration::conversion::{array2_to_dataframe, dataframe_to_array2};
#[cfg(feature = "scirs2")]
use crate::series::Series;

/// Result of an eigenvalue decomposition.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct EigResult {
    /// Eigenvalues (real parts for symmetric/Hermitian matrices)
    pub values: Vec<f64>,
    /// Eigenvectors as a DataFrame (each column is an eigenvector)
    pub vectors: DataFrame,
}

/// Result of a Singular Value Decomposition.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors as a DataFrame
    pub u: DataFrame,
    /// Singular values
    pub s: Vec<f64>,
    /// Right singular vectors (transposed) as a DataFrame
    pub vt: DataFrame,
}

/// Result of a QR decomposition.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct QrResult {
    /// Orthogonal matrix Q as a DataFrame
    pub q: DataFrame,
    /// Upper triangular matrix R as a DataFrame
    pub r: DataFrame,
}

/// Result of an LU decomposition.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct LuResult {
    /// Lower triangular matrix L (with unit diagonal) as a DataFrame
    pub l: DataFrame,
    /// Upper triangular matrix U as a DataFrame
    pub u: DataFrame,
}

/// Result of a least-squares solve.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct LstsqDataFrameResult {
    /// Solution matrix as a DataFrame (one column per right-hand side column in b)
    pub solution: DataFrame,
    /// Sum of squared residuals for each RHS column
    pub residuals: Vec<f64>,
    /// Effective rank of matrix A
    pub rank: usize,
}

/// Linear algebra operations on DataFrames using SciRS2's implementations.
///
/// All methods treat DataFrames as numeric matrices. Only f64 numeric columns
/// are supported. Column names in the result DataFrames follow the convention
/// described in each method's documentation.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "scirs2")]
/// # {
/// use pandrs::{DataFrame, Series};
/// use pandrs::scirs2_integration::linalg::SciRS2LinAlg;
///
/// let mut a = DataFrame::new();
/// a.add_column("c0".to_string(),
///     Series::new(vec![2.0f64, 1.0], Some("c0".to_string())).expect("ok")).expect("ok");
/// a.add_column("c1".to_string(),
///     Series::new(vec![1.0f64, 2.0], Some("c1".to_string())).expect("ok")).expect("ok");
///
/// let det = SciRS2LinAlg::det(&a).expect("det ok");
/// // det of [[2, 1], [1, 2]] = 3.0
/// assert!((det - 3.0).abs() < 1e-10);
/// # }
/// ```
#[cfg(feature = "scirs2")]
pub struct SciRS2LinAlg;

#[cfg(feature = "scirs2")]
impl SciRS2LinAlg {
    /// Perform matrix multiplication of two DataFrames.
    ///
    /// Both DataFrames are treated as numeric matrices. The result has columns
    /// named `c0`, `c1`, ..., `c{n-1}` where n is the number of columns in `b`.
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix DataFrame with shape (m, k)
    /// * `b` - Right matrix DataFrame with shape (k, n)
    ///
    /// # Errors
    ///
    /// Returns an error if the inner dimensions do not match.
    pub fn matmul(a: &DataFrame, b: &DataFrame) -> Result<DataFrame> {
        let a_col_names = a.column_names();
        let a_cols: Vec<&str> = a_col_names.iter().map(|s| s.as_str()).collect();
        let b_col_names = b.column_names();
        let b_cols: Vec<&str> = b_col_names.iter().map(|s| s.as_str()).collect();

        let arr_a = dataframe_to_array2(a, &a_cols)?;
        let arr_b = dataframe_to_array2(b, &b_cols)?;

        let (m, k_a) = arr_a.dim();
        let (k_b, n) = arr_b.dim();

        if k_a != k_b {
            return Err(Error::InvalidInput(format!(
                "Matrix dimensions incompatible for multiplication: ({}, {}) x ({}, {})",
                m, k_a, k_b, n
            )));
        }

        let result = arr_a.dot(&arr_b);
        let col_names: Vec<String> = (0..n).map(|i| format!("c{}", i)).collect();
        array2_to_dataframe(&result, col_names)
    }

    /// Compute eigenvalues and eigenvectors of a symmetric matrix DataFrame.
    ///
    /// Uses SciRS2's `eigh` function which is optimised for symmetric matrices.
    /// Eigenvectors are returned as columns named `ev0`, `ev1`, ..., `ev{n-1}`.
    ///
    /// # Arguments
    ///
    /// * `df` - A square symmetric numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is not square or the decomposition fails.
    pub fn eig(df: &DataFrame) -> Result<EigResult> {
        use scirs2_linalg::eigh;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (n_rows, n_cols) = arr.dim();
        if n_rows != n_cols {
            return Err(Error::InvalidInput(format!(
                "Eigendecomposition requires a square matrix, got ({}, {})",
                n_rows, n_cols
            )));
        }

        let (eigenvalues, eigenvectors) = eigh(&arr.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 eigh failed: {}", e)))?;

        let values: Vec<f64> = eigenvalues.iter().copied().collect();

        let ev_col_names: Vec<String> = (0..n_cols).map(|i| format!("ev{}", i)).collect();
        let vectors_df = array2_to_dataframe(&eigenvectors, ev_col_names)?;

        Ok(EigResult {
            values,
            vectors: vectors_df,
        })
    }

    /// Compute the Singular Value Decomposition (SVD) of a DataFrame.
    ///
    /// Returns a triple (U, S, Vt) where U and Vt are DataFrames and S is a vector
    /// of singular values.
    ///
    /// Column naming:
    /// - `u`: columns named `u0`, `u1`, ..., `u{m-1}`
    /// - `vt`: columns named `v0`, `v1`, ..., `v{n-1}`
    ///
    /// # Arguments
    ///
    /// * `df` - The numeric DataFrame with shape (m, n)
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is empty or the SVD fails.
    pub fn svd(df: &DataFrame) -> Result<SvdResult> {
        use scirs2_linalg::svd;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (m, n) = arr.dim();

        let (u, s, vt) = svd(&arr.view(), false, None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 svd failed: {}", e)))?;

        let k = s.len();
        let singular_values: Vec<f64> = s.iter().copied().collect();

        let u_col_names: Vec<String> = (0..u.ncols()).map(|i| format!("u{}", i)).collect();
        let vt_col_names: Vec<String> = (0..vt.ncols()).map(|i| format!("v{}", i)).collect();

        let u_df = array2_to_dataframe(&u, u_col_names)?;
        let vt_df = array2_to_dataframe(&vt, vt_col_names)?;

        Ok(SvdResult {
            u: u_df,
            s: singular_values,
            vt: vt_df,
        })
    }

    /// Solve the linear system Ax = b for x.
    ///
    /// Both `a` and `b` are DataFrames. `a` must be square. `b` can have
    /// multiple columns (each represents a right-hand side vector).
    ///
    /// Result columns are named `x0`, `x1`, ..., `x{k-1}`.
    ///
    /// # Arguments
    ///
    /// * `a` - The coefficient matrix (square, n×n)
    /// * `b` - The right-hand side (n×k)
    ///
    /// # Errors
    ///
    /// Returns an error if `a` is not square or the system is singular.
    pub fn solve(a: &DataFrame, b: &DataFrame) -> Result<DataFrame> {
        use scirs2_linalg::solve_multiple;

        let a_cols: Vec<String> = a.column_names();
        let b_cols: Vec<String> = b.column_names();
        let a_col_refs: Vec<&str> = a_cols.iter().map(|s| s.as_str()).collect();
        let b_col_refs: Vec<&str> = b_cols.iter().map(|s| s.as_str()).collect();

        let arr_a = dataframe_to_array2(a, &a_col_refs)?;
        let arr_b = dataframe_to_array2(b, &b_col_refs)?;

        let (n_rows, n_cols) = arr_a.dim();
        if n_rows != n_cols {
            return Err(Error::InvalidInput(format!(
                "Coefficient matrix must be square, got ({}, {})",
                n_rows, n_cols
            )));
        }

        let x = solve_multiple(&arr_a.view(), &arr_b.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 solve failed: {}", e)))?;

        let k = x.ncols();
        let x_col_names: Vec<String> = (0..k).map(|i| format!("x{}", i)).collect();
        array2_to_dataframe(&x, x_col_names)
    }

    /// Compute the matrix inverse of a square numeric DataFrame.
    ///
    /// Result columns retain the same names as the input DataFrame.
    ///
    /// # Arguments
    ///
    /// * `df` - A square numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is not square or the matrix is singular.
    pub fn inv(df: &DataFrame) -> Result<DataFrame> {
        use scirs2_linalg::inv;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (n_rows, n_cols) = arr.dim();
        if n_rows != n_cols {
            return Err(Error::InvalidInput(format!(
                "Matrix inverse requires a square matrix, got ({}, {})",
                n_rows, n_cols
            )));
        }

        let inv_arr = inv(&arr.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 inv failed: {}", e)))?;

        array2_to_dataframe(&inv_arr, cols)
    }

    /// Compute the determinant of a square numeric DataFrame.
    ///
    /// # Arguments
    ///
    /// * `df` - A square numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is not square or the computation fails.
    pub fn det(df: &DataFrame) -> Result<f64> {
        use scirs2_linalg::det;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (n_rows, n_cols) = arr.dim();
        if n_rows != n_cols {
            return Err(Error::InvalidInput(format!(
                "Determinant requires a square matrix, got ({}, {})",
                n_rows, n_cols
            )));
        }

        det(&arr.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 det failed: {}", e)))
    }

    /// Compute the QR decomposition of a DataFrame.
    ///
    /// Returns a `QrResult` containing orthogonal Q and upper-triangular R such that
    /// `A = Q * R`.  Column names follow the conventions `q0, q1, ...` and `r0, r1, ...`.
    ///
    /// # Arguments
    ///
    /// * `df` - The input numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is empty or the decomposition fails.
    pub fn qr(df: &DataFrame) -> Result<QrResult> {
        use scirs2_linalg::qr;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (q_arr, r_arr) = qr(&arr.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 qr failed: {}", e)))?;

        let q_col_names: Vec<String> = (0..q_arr.ncols()).map(|i| format!("q{}", i)).collect();
        let r_col_names: Vec<String> = (0..r_arr.ncols()).map(|i| format!("r{}", i)).collect();

        let q_df = array2_to_dataframe(&q_arr, q_col_names)?;
        let r_df = array2_to_dataframe(&r_arr, r_col_names)?;

        Ok(QrResult { q: q_df, r: r_df })
    }

    /// Compute the Cholesky decomposition of a symmetric positive-definite DataFrame.
    ///
    /// Returns the lower-triangular factor L such that `A = L * L.T`.
    /// Result columns retain the same names as the input DataFrame.
    ///
    /// # Arguments
    ///
    /// * `df` - A square symmetric positive-definite numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square, not positive-definite, or the
    /// decomposition fails.
    pub fn cholesky(df: &DataFrame) -> Result<DataFrame> {
        use scirs2_linalg::cholesky;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (n_rows, n_cols) = arr.dim();
        if n_rows != n_cols {
            return Err(Error::InvalidInput(format!(
                "Cholesky decomposition requires a square matrix, got ({}, {})",
                n_rows, n_cols
            )));
        }

        let l = cholesky(&arr.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 cholesky failed: {}", e)))?;

        array2_to_dataframe(&l, cols)
    }

    /// Compute the LU decomposition of a DataFrame.
    ///
    /// Returns an `LuResult` with lower-triangular L (unit diagonal) and
    /// upper-triangular U such that `P * A = L * U` (pivot matrix P is absorbed).
    /// Column names follow `l0, l1, ...` and `u0, u1, ...`.
    ///
    /// # Arguments
    ///
    /// * `df` - The input numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is empty, singular, or the decomposition fails.
    pub fn lu(df: &DataFrame) -> Result<LuResult> {
        use scirs2_linalg::lu;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (_p_arr, l_arr, u_arr) = lu(&arr.view(), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 lu failed: {}", e)))?;

        let l_col_names: Vec<String> = (0..l_arr.ncols()).map(|i| format!("l{}", i)).collect();
        let u_col_names: Vec<String> = (0..u_arr.ncols()).map(|i| format!("u{}", i)).collect();

        let l_df = array2_to_dataframe(&l_arr, l_col_names)?;
        let u_df = array2_to_dataframe(&u_arr, u_col_names)?;

        Ok(LuResult { l: l_df, u: u_df })
    }

    /// Solve the least-squares problem min ||Ax - b||² for each column of b.
    ///
    /// Uses SciRS2's `lstsq` function which supports over-determined and
    /// under-determined systems.  Solution columns are named `x0, x1, ...`.
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix (m × n)
    /// * `b` - Right-hand side matrix (m × k); each column is an independent RHS
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions are incompatible or the solve fails.
    pub fn lstsq(a: &DataFrame, b: &DataFrame) -> Result<LstsqDataFrameResult> {
        use scirs2_linalg::lstsq;

        let a_cols: Vec<String> = a.column_names();
        let b_cols: Vec<String> = b.column_names();
        let a_col_refs: Vec<&str> = a_cols.iter().map(|s| s.as_str()).collect();
        let b_col_refs: Vec<&str> = b_cols.iter().map(|s| s.as_str()).collect();

        let arr_a = dataframe_to_array2(a, &a_col_refs)?;
        let arr_b = dataframe_to_array2(b, &b_col_refs)?;

        let (m_a, _n) = arr_a.dim();
        let (m_b, k) = arr_b.dim();

        if m_a != m_b {
            return Err(Error::InvalidInput(format!(
                "lstsq: A has {} rows but b has {} rows",
                m_a, m_b
            )));
        }

        let mut solution_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut residuals: Vec<f64> = Vec::with_capacity(k);
        let mut final_rank = 0usize;

        for j in 0..k {
            let col_b: scirs2_core::ndarray::Array1<f64> = arr_b.column(j).to_owned();
            let res = lstsq(&arr_a.view(), &col_b.view(), None)
                .map_err(|e| Error::OperationFailed(format!("SciRS2 lstsq failed: {}", e)))?;

            solution_cols.push(res.x.iter().copied().collect());
            residuals.push(res.residuals);
            final_rank = res.rank;
        }

        // Build solution DataFrame
        let n_sol = if solution_cols.is_empty() {
            0
        } else {
            solution_cols[0].len()
        };
        let col_names: Vec<String> = (0..k).map(|i| format!("x{}", i)).collect();

        let mut sol_df = DataFrame::new();
        for (j, col_name) in col_names.iter().enumerate() {
            let vals = solution_cols[j].clone();
            let series = Series::new(vals, Some(col_name.clone()))?;
            sol_df.add_column(col_name.clone(), series)?;
        }
        let _ = n_sol; // silence unused warning

        Ok(LstsqDataFrameResult {
            solution: sol_df,
            residuals,
            rank: final_rank,
        })
    }

    /// Compute a matrix norm of a DataFrame.
    ///
    /// Supported `ord` values:
    /// - `"fro"` — Frobenius norm (default if unrecognised)
    /// - `"1"` — maximum column sum (1-norm)
    /// - `"inf"` — maximum row sum (infinity-norm)
    ///
    /// # Arguments
    ///
    /// * `df` - The numeric DataFrame
    /// * `ord` - Norm order string (`"fro"`, `"1"`, or `"inf"`)
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is empty or the computation fails.
    pub fn matrix_norm(df: &DataFrame, ord: &str) -> Result<f64> {
        use scirs2_linalg::matrix_norm;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        matrix_norm(&arr.view(), ord, None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 matrix_norm failed: {}", e)))
    }

    /// Compute the numerical rank of a DataFrame using SVD.
    ///
    /// Uses a tolerance of `max(m, n) * eps * sigma_max` to determine which
    /// singular values are significant.
    ///
    /// # Arguments
    ///
    /// * `df` - The numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is empty or the SVD fails.
    pub fn matrix_rank(df: &DataFrame) -> Result<usize> {
        use scirs2_linalg::matrix_rank;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        matrix_rank(&arr.view(), None, None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 matrix_rank failed: {}", e)))
    }

    /// Compute the 2-norm condition number of a square numeric DataFrame.
    ///
    /// Returns `sigma_max / sigma_min` where sigma values are the singular values.
    /// Returns `f64::INFINITY` for singular matrices.
    ///
    /// # Arguments
    ///
    /// * `df` - A square numeric DataFrame
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is not square or the computation fails.
    pub fn condition_number(df: &DataFrame) -> Result<f64> {
        use scirs2_linalg::cond;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr = dataframe_to_array2(df, &col_refs)?;

        let (n_rows, n_cols) = arr.dim();
        if n_rows != n_cols {
            return Err(Error::InvalidInput(format!(
                "Condition number requires a square matrix, got ({}, {})",
                n_rows, n_cols
            )));
        }

        cond(&arr.view(), Some("2"), None)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 cond failed: {}", e)))
    }

    /// Compute the pseudoinverse of a numeric DataFrame using least-squares.
    ///
    /// Computes `A_pinv = lstsq(A, I)` where `I` is the m×m identity matrix.
    /// Result columns are named `pi0, pi1, ...`.
    ///
    /// # Arguments
    ///
    /// * `df` - The numeric DataFrame with shape (m, n)
    ///
    /// # Errors
    ///
    /// Returns an error if the DataFrame is empty or the solve fails.
    pub fn pinv(df: &DataFrame) -> Result<DataFrame> {
        use scirs2_core::ndarray::Array2;
        use scirs2_linalg::lstsq;

        let cols: Vec<String> = df.column_names();
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        let arr_a = dataframe_to_array2(df, &col_refs)?;

        let (m, n) = arr_a.dim();
        if m == 0 || n == 0 {
            return Err(Error::EmptyData("pinv: input matrix is empty".to_string()));
        }

        // Identity matrix of size m
        let eye = Array2::<f64>::eye(m);

        // Solve A * X = I column by column and collect the n×m pseudoinverse
        let mut pinv_cols: Vec<Vec<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            let col_b: scirs2_core::ndarray::Array1<f64> = eye.column(j).to_owned();
            let res = lstsq(&arr_a.view(), &col_b.view(), None).map_err(|e| {
                Error::OperationFailed(format!("SciRS2 lstsq (pinv) failed: {}", e))
            })?;
            pinv_cols.push(res.x.iter().copied().collect());
        }

        // pinv_cols[j] is the j-th column of A_pinv (length n)
        // Build result DataFrame with m columns
        let col_names: Vec<String> = (0..m).map(|i| format!("pi{}", i)).collect();
        let mut result_df = DataFrame::new();
        for (j, col_name) in col_names.iter().enumerate() {
            let vals = pinv_cols[j].clone();
            let series = Series::new(vals, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }

        Ok(result_df)
    }
}
