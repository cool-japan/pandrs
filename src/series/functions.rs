use crate::core::error::Result;
use std::fmt::Debug;

/// Common Series functions
pub mod common_functions {
    use super::*;

    /// Calculate the sum of a numeric Series
    pub fn sum<T>(values: &[T]) -> Result<T>
    where
        T: Debug + Clone + std::iter::Sum + Default,
    {
        if values.is_empty() {
            return Ok(T::default());
        }

        Ok(values.iter().cloned().sum())
    }

    /// Calculate the mean of a numeric Series
    pub fn mean<T>(values: &[T]) -> Result<T>
    where
        T: Debug
            + Clone
            + std::iter::Sum
            + std::ops::Div<Output = T>
            + num_traits::NumCast
            + Default,
    {
        if values.is_empty() {
            return Err(crate::core::error::Error::EmptyData(
                "Cannot calculate mean of an empty Series".to_string(),
            ));
        }

        let sum = sum(values)?;
        let count = match num_traits::cast(values.len()) {
            Some(n) => n,
            None => {
                return Err(crate::core::error::Error::Cast(
                    "Cannot cast length to numeric type".to_string(),
                ))
            }
        };

        Ok(sum / count)
    }
}
