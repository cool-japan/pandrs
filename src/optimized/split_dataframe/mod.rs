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

// 行操作
pub mod row_ops;

// 関数適用
pub mod apply;

// 並列処理
pub mod parallel;

// 選択操作
pub mod select;

// 集計操作
pub mod aggregate;

// ソート操作
pub mod sort;

// シリアライズ操作
pub mod serialize;

// ColumnView実装
pub mod column_view;

// 統計関数モジュール
pub mod stats;

// 再エクスポート
pub use core::{OptimizedDataFrame, ColumnView};

// 入出力系の型を再エクスポート
pub use io::ParquetCompression;
pub use serialize::JsonOrient;

// 結合系の型を再エクスポート
pub use join::JoinType;

// グループ化系の型を再エクスポート
pub use group::{GroupBy, AggregateOp};

// 統計系の型を再エクスポート
pub use stats::{StatDescribe, StatResult};
