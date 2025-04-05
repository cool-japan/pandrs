//! 機械学習機能を提供するモジュール
//! 
//! このモジュールは、PandRSのデータ構造を機械学習アルゴリズムで使用するための
//! 変換パイプラインとユーティリティを提供します。

pub mod pipeline;
pub mod preprocessing;
pub mod metrics;
pub mod models;
pub mod dimension_reduction;
pub mod clustering;

// 外部クレートをreexportする予定
// pub use linfa;