//! Time Series Analysis and Forecasting Module
//!
//! This module provides comprehensive time series analysis and forecasting capabilities
//! including seasonal decomposition, trend analysis, forecasting algorithms, and
//! statistical tests for temporal data.

pub mod core;
pub mod decomposition;
pub mod forecasting;
pub mod analysis;
pub mod features;
pub mod stats;
pub mod preprocessing;

pub use core::{TimeSeries, TimeSeriesBuilder, TimePoint, Frequency, DateTimeIndex};
pub use decomposition::{SeasonalDecomposition, DecompositionMethod, DecompositionResult};
pub use forecasting::{
    Forecaster, ForecastResult, ForecastMetrics, ArimaForecaster, ExponentialSmoothingForecaster,
    SimpleMovingAverageForecaster, LinearTrendForecaster,
};
pub use analysis::{
    TrendAnalysis, SeasonalityAnalysis, StationarityTest, AutocorrelationAnalysis,
    ChangePointDetection,
};
pub use features::{TimeSeriesFeatureExtractor, FeatureSet, WindowFeatures, StatisticalFeatures};
pub use stats::{
    TimeSeriesStats, AugmentedDickeyFullerTest, KwiatkowskiPhillipsSchmidtShinTest,
    SeasonalTest, WhiteNoiseTest,
};
pub use preprocessing::{
    TimeSeriesPreprocessor, MissingValueStrategy, OutlierDetection, Normalization,
    Differencing,
};

use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};