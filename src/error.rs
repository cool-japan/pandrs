// Re-export from core module with a deprecation notice
#[deprecated(since = "0.1.0-alpha.2", note = "Use crate::core::error instead")]
pub use crate::core::error::{io_error, Error, PandRSError, Result};

// For backward compatibility, these From impls are implemented in the core::error module

#[allow(deprecated)]
#[cfg(feature = "plotters")]
impl<E: std::error::Error + Send + Sync + 'static> From<plotters::drawing::DrawingAreaErrorKind<E>>
    for Error
{
    fn from(err: plotters::drawing::DrawingAreaErrorKind<E>) -> Self {
        Error::Visualization(format!("Plot drawing error: {}", err))
    }
}
