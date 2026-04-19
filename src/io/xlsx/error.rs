//! Error helpers for the xlsx module.
//!
//! The xlsx submodule maps its internal failure conditions onto the crate-wide
//! [`pandrs::error::Error`] enum. We intentionally do not introduce a new error
//! type because the public surface of `pandrs::io::excel` already returns
//! `pandrs::error::Result<T>`, and a secondary error type would force all
//! callers to bridge between two hierarchies.

use crate::error::Error;

/// Build an I/O-flavoured error with a formatted message.
#[inline]
pub(super) fn io_err(message: impl Into<String>) -> Error {
    Error::IoError(message.into())
}

/// Build an invalid-input error with a formatted message.
#[inline]
pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

/// Convert a `std::io::Error` into an `Error::Io` variant.
#[inline]
pub(super) fn std_io(err: std::io::Error) -> Error {
    Error::Io(err)
}

/// Convert a [`quick_xml::Error`] into a `pandrs::error::Error`.
#[inline]
pub(super) fn xml_err(err: quick_xml::Error) -> Error {
    Error::IoError(format!("xlsx xml error: {err}"))
}

/// Convert any error with a `Display` implementation from the oxiarc stack into
/// a `pandrs::error::Error`. We keep this generic so we do not need a direct
/// dependency on `oxiarc-core`.
#[inline]
pub(super) fn zip_err<E: std::fmt::Display>(err: E) -> Error {
    Error::IoError(format!("xlsx zip error: {err}"))
}
