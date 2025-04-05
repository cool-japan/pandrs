//! 機械学習機能を提供するモジュール
//! 
//! このモジュールは、PandRSのデータ構造を機械学習アルゴリズムで使用するための
//! 変換パイプラインとユーティリティを提供します。
//!
//! 注: このモジュールは最適化されたOptimizedDataFrameを使用して実装されています。

pub mod pipeline;
pub mod preprocessing;
pub mod metrics;
pub mod dimension_reduction;
pub mod models;
pub mod clustering;
pub mod anomaly_detection;

// 外部クレートをreexportする予定
// pub use linfa;