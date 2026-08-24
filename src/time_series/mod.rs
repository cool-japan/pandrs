//! Time Series Analysis and Forecasting Module
//!
//! This module provides comprehensive time series analysis and forecasting capabilities
//! including seasonal decomposition, trend analysis, forecasting algorithms, and
//! statistical tests for temporal data.
//!
//! # Forecasting Algorithms
//!
//! - Simple Moving Average
//! - Linear Trend
//! - Exponential Smoothing (Simple, Double/Holt, Triple/Holt-Winters)
//! - ARIMA
//! - SARIMA (Seasonal ARIMA)
//! - Auto ARIMA (automatic model selection)

pub mod advanced_forecasting;
pub mod analysis;
pub mod core;
pub mod decomposition;
pub mod features;
pub(crate) mod filters;
pub mod forecasting;
pub(crate) mod loess;
pub mod preprocessing;
pub(crate) mod smoothing;
pub(crate) mod spectral;
pub mod stats;
pub(crate) mod stl;

pub use analysis::{
    AutocorrelationAnalysis, ChangePointDetection, SeasonalityAnalysis, StationarityTest,
    TrendAnalysis,
};
pub use core::{DateTimeIndex, Frequency, TimePoint, TimeSeries, TimeSeriesBuilder};
pub use decomposition::{DecompositionMethod, DecompositionResult, SeasonalDecomposition};
pub use features::{FeatureSet, StatisticalFeatures, TimeSeriesFeatureExtractor, WindowFeatures};
pub use forecasting::{
    ArimaForecaster, ExponentialSmoothingForecaster, ForecastMetrics, ForecastResult, Forecaster,
    LinearTrendForecaster, SimpleMovingAverageForecaster,
};
pub use preprocessing::{
    Differencing, MissingValueStrategy, Normalization, OutlierDetection, SmoothingConfig,
    SmoothingMethod, TimeSeriesPreprocessor,
};
pub use stats::{
    AugmentedDickeyFullerTest, KwiatkowskiPhillipsSchmidtShinTest, SeasonalTest, TimeSeriesStats,
    WhiteNoiseTest,
};

// Advanced forecasting exports
pub use advanced_forecasting::{
    AutoArima, ModelSelectionCriterion, ModelSelectionResult, SarimaForecaster,
};
