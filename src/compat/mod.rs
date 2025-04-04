// 互換性機能を提供するモジュール
mod dataframe_adapter;

// エクスポート
pub use dataframe_adapter::{DataFrameCompat, ParallelCompat};