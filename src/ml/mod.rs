//! 機械学習機能を提供するモジュール
//! 
//! このモジュールは、PandRSのデータ構造を機械学習アルゴリズムで使用するための
//! 変換パイプラインとユーティリティを提供します。
//!
//! 注: このモジュールは最適化されたOptimizedDataFrameを使用して実装されています。

pub mod pipeline;
pub mod preprocessing;

// 以下のモジュールはコメントアウトして、基本機能のみをサポート
// pub mod metrics;
// pub mod models;
// pub mod dimension_reduction;
// pub mod clustering;
// pub mod anomaly_detection;

// 外部クレートをreexportする予定
// pub use linfa;