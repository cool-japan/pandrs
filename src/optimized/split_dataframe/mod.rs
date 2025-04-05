//! 分割されたOptimizedDataFrameの実装

// コアモジュール
pub mod core;

// 列操作
pub mod column_ops;

// データ操作
pub mod data_ops;

// 入出力
pub mod io;

// 結合操作
pub mod join;

// グループ化と集計
pub mod group;

// インデックス操作
pub mod index;

// ColumnView実装
pub mod column_view;

// 再エクスポート
pub use core::{OptimizedDataFrame, ColumnView};

// 入出力系の型を再エクスポート
pub use io::{JsonOrient, ParquetCompression};

// 結合系の型を再エクスポート
pub use join::JoinType;

// グループ化系の型を再エクスポート
pub use group::{GroupBy, AggregateOp};
