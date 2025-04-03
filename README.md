# PandRS

Rustで実装されたデータ分析用DataFrameライブラリです。Pythonの`pandas`ライブラリにインスパイアされた機能と設計を持ちます。

## 機能

- Series (1次元配列) とDataFrame (2次元表) データ構造
- 欠損値（NA）のサポート
- グループ化操作
- インデックスによる行ラベル
- CSVからのデータ読み込みと書き込み
- JSONからのデータ読み込みと書き込み
- フィルタリング、ソート、結合などの基本操作
- 数値データに対する集計関数 (合計、平均、最小、最大など)
- 文字列データに対する特殊操作
- マルチレベルインデックス (開発中)
- カテゴリカルデータ型 (開発中)
- 高度なデータフレーム操作 (開発中)

## 使用例

### DataFrameの作成と基本操作

```rust
use pandrs::{DataFrame, Series};

// シリーズの作成
let ages = Series::new(vec![30, 25, 40], Some("age".to_string()))?;
let heights = Series::new(vec![180, 175, 182], Some("height".to_string()))?;

// DataFrameにシリーズを追加
let mut df = DataFrame::new();
df.add_column("age".to_string(), ages)?;
df.add_column("height".to_string(), heights)?;

// CSVとして保存
df.to_csv("data.csv")?;

// CSVからDataFrameを読み込む
let df_from_csv = DataFrame::from_csv("data.csv", true)?;
```

### 数値操作

```rust
// 数値シリーズの作成
let numbers = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string()))?;

// 統計計算
let sum = numbers.sum();         // 150
let mean = numbers.mean()?;      // 30
let min = numbers.min()?;        // 10
let max = numbers.max()?;        // 50
```

## インストール

Cargo.tomlに以下を追加:

```toml
[dependencies]
pandrs = { git = "https://github.com/cool-japan/pandrs" }
```

### 欠損値（NA）の操作

```rust
// NA値を含むシリーズの作成
let data = vec![
    NA::Value(10), 
    NA::Value(20), 
    NA::NA,  // 欠損値
    NA::Value(40)
];
let series = NASeries::new(data, Some("values".to_string()))?;

// NA値の処理
println!("NAの数: {}", series.na_count());
println!("値の数: {}", series.value_count());

// NAの削除と埋め合わせ
let dropped = series.dropna()?;
let filled = series.fillna(0)?;
```

### グループ操作

```rust
// データとグループキー
let values = Series::new(vec![10, 20, 15, 30, 25], Some("values".to_string()))?;
let keys = vec!["A", "B", "A", "C", "B"];

// グループ化して集計
let group_by = GroupBy::new(
    keys.iter().map(|s| s.to_string()).collect(),
    &values,
    Some("by_category".to_string())
)?;

// 集計結果
let sums = group_by.sum()?;
let means = group_by.mean()?;
```

### 時系列データの操作

```rust
use pandrs::temporal::{TimeSeries, date_range, Frequency};
use chrono::NaiveDate;

// 日付範囲を生成
let dates = date_range(
    NaiveDate::from_str("2023-01-01")?,
    NaiveDate::from_str("2023-01-31")?,
    Frequency::Daily,
    true
)?;

// 時系列データを作成
let time_series = TimeSeries::new(values, dates, Some("daily_data".to_string()))?;

// 時間フィルタリング
let filtered = time_series.filter_by_time(
    &NaiveDate::from_str("2023-01-10")?,
    &NaiveDate::from_str("2023-01-20")?
)?;

// 移動平均の計算
let moving_avg = time_series.rolling_mean(3)?;

// リサンプリング（週次に変換）
let weekly = time_series.resample(Frequency::Weekly).mean()?;
```

### ピボットテーブルとグループ化

```rust
use pandrs::pivot::AggFunction;

// グループ化と集計
let grouped = df.groupby("category")?;
let category_sum = grouped.sum(&["sales"])?;

// ピボットテーブル
let pivot_result = df.pivot_table(
    "category",   // インデックス列
    "region",     // カラム列
    "sales",      // 値列
    AggFunction::Sum
)?;
```

## 開発計画

- [x] 基本的なDataFrame構造
- [x] Series実装
- [x] インデックス機能
- [x] CSV入出力
- [x] JSON入出力
- [x] 欠損値処理
- [x] グループ化処理
- [x] 時系列データ対応
  - [x] 日付範囲生成
  - [x] 時間フィルタリング
  - [x] 移動平均計算
  - [x] 周波数変換（リサンプリング）
- [x] ピボットテーブル
- [x] 結合操作の完全実装
  - [x] 内部結合 (内部一致)
  - [x] 左結合 (左側優先)
  - [x] 右結合 (右側優先)
  - [x] 外部結合 (すべての行)
- [x] 可視化機能の連携
  - [x] 折れ線グラフ
  - [x] 散布図
  - [x] テキストプロットによる出力
- [x] 並列処理サポート
  - [x] Series/NASeriesの並列変換
  - [x] DataFrameの並列処理
  - [x] 並列フィルタリング
  - [x] 並列集計
- [ ] マルチレベルインデックス（開発中）
- [ ] カテゴリカルデータ型（開発中）
- [ ] 高度なデータフレーム操作（開発中）
- [ ] メモリ使用効率の最適化

## 最近の改善

- テキストベースのプロットを用いた可視化機能を実装
  - 折れ線グラフ、散布図、ポイントプロットに対応
  - ターミナル出力とファイル出力に対応
  - データ可視化の多様なサンプルを追加
- Rayonを使用した並列処理機能を実装
  - Series/NASeriesの並列マップと並列フィルタリング
  - DataFrameの並列適用と並列フィルタリング
  - 並列集計と並列ソートのユーティリティ関数
  - マルチスレッドによる高性能な処理を実現
- 内部結合、左結合、右結合、外部結合を実装し、DataFrameの結合操作を完成
- レコード指向JSONおよび列指向JSONの出力を実装
- ピボットテーブル機能とグループ化操作を強化（集計関数とデータアクセス）
- RFC3339形式の日付パーシングに対応し、時系列データ操作の互換性を向上
- 移動平均計算における整数オーバーフロー問題を修正
- `DAILY`、`WEEKLY`などの完全形式の周波数指定に対応
- すべてのテストが成功するよう安定性を向上

## ライセンス

Apache License 2.0 で提供されています。