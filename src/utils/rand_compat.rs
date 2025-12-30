// Compatibility module for rand crate versions
// Providing backward compatibility from rand 0.9.0 to previous versions

use rand::distr::uniform::{SampleRange, SampleUniform};
use rand::Rng;

/// Compatibility wrapper for thread_rng
///
/// This function is provided for backward compatibility with code
/// that uses rand::thread_rng(). In newer versions of rand,
/// this has been replaced with rand::rng()
pub fn thread_rng() -> impl rand::Rng {
    rand::rng()
}

/// Compatibility extension trait for gen_range
///
/// This trait is provided for backward compatibility with code
/// that uses rng.gen_range(). In newer versions of rand,
/// this has been replaced with rng.random_range()
pub trait GenRangeCompat: Rng {
    fn gen_range<R, T>(&mut self, range: R) -> T
    where
        R: SampleRange<T>,
        T: SampleUniform,
    {
        self.random_range(range)
    }
}

// Implement GenRangeCompat for anything that implements Rng
impl<T: Rng> GenRangeCompat for T {}
