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
}
