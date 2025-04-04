# PandRS

Rustで実装されたデータ分析用DataFrameライブラリです。Pythonの`pandas`ライブラリにインスパイアされた機能と設計を持ちます。

## 機能

- Series (1次元配列) とDataFrame (2次元表) データ構造
- 列指向ストレージと遅延評価による高性能な最適化実装
- 欠損値（NA）のサポート
- グループ化操作
- インデックスによる行ラベル
- CSVからのデータ読み込みと書き込み
- JSONからのデータ読み込みと書き込み
- ParquetファイルのI/O
- 文字列プールによるメモリ使用量の最適化
- フィルタリング、ソート、結合などの基本操作
- 数値データに対する集計関数 (合計、平均、最小、最大など)
- 文字列データに対する特殊操作
- マルチレベルインデックス
- カテゴリカルデータ型
- 高度なデータフレーム操作

## 使用例

### DataFrameの作成と基本操作

```rust
use pandrs::{DataFrame, Column, Int64Column};

// 列データを準備
let ages = vec![30, 25, 40];
let heights = vec![180, 175, 182];

// DataFrameにInt64Column（整数列）を追加
let mut df = DataFrame::new();
df.add_column("age", Column::Int64(Int64Column::new(ages)))?;
df.add_column("height", Column::Int64(Int64Column::new(heights)))?;

// CSVとして保存
df.to_csv("data.csv")?;

// CSVからDataFrameを読み込む
let df_from_csv = DataFrame::from_csv("data.csv", true)?;
```

### 数値操作

```rust
// 数値列の作成
let numbers = Int64Column::new(vec![10, 20, 30, 40, 50]);

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
// NA値を含む整数列の作成
let data = vec![10, 20, 0, 40]; // 0はNAとして扱う
let null_mask = vec![false, false, true, false]; // 3番目の要素がNA
let column = Int64Column::with_nulls(data, null_mask);

// NA値の処理
println!("NAの数: {}", column.na_count());
println!("値の数: {}", column.len() - column.na_count());

// NAの削除と埋め合わせ
let dropped = column.dropna()?;
let filled = column.fillna(0)?;
```

### グループ操作

```rust
// データの準備
let mut df = DataFrame::new();
df.add_column("values", Column::Int64(Int64Column::new(vec![10, 20, 15, 30, 25])))?;
df.add_column("category", Column::String(StringColumn::new(vec![
    "A".to_string(), "B".to_string(), "A".to_string(), 
    "C".to_string(), "B".to_string()
])))?;

// グループ化して集計
let group_by = df.groupby("category")?;

// 集計結果
let sums = group_by.sum(&["values"])?;
let means = group_by.mean(&["values"])?;
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
- [x] マルチレベルインデックス
  - [x] ヒエラルキカルなインデックス構造
  - [x] 複数レベルによるデータのグループ化
  - [x] レベル操作（入れ替え、選択）
- [x] カテゴリカルデータ型
- [x] 高度なデータフレーム操作
  - [x] 長形式・広形式変換（melt, stack, unstack）
  - [x] 条件付き集計
  - [x] データフレーム結合
- [ ] メモリ使用効率の最適化
- [x] Pythonバインディング
  - [x] PyO3を使用したPythonモジュール化
  - [x] numpyとpandasとの相互運用性
  - [x] Jupyter Notebookサポート

### マルチレベルインデックスの操作

```rust
use pandrs::{DataFrame, MultiIndex, Column, StringColumn};

// タプルからMultiIndexを作成
let tuples = vec![
    vec!["A".to_string(), "a".to_string()],
    vec!["A".to_string(), "b".to_string()],
    vec!["B".to_string(), "a".to_string()],
    vec!["B".to_string(), "b".to_string()],
];

// レベル名を設定
let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);
let multi_idx = MultiIndex::from_tuples(tuples, names)?;

// MultiIndexを使用したDataFrameを作成
let mut df = DataFrame::with_multi_index(multi_idx.clone());

// データを追加
let data = vec!["data1".to_string(), "data2".to_string(), "data3".to_string(), "data4".to_string()];
df.add_column("data", Column::String(StringColumn::new(data)))?;

// レベル操作
let level0_values = multi_idx.get_level_values(0)?;
let level1_values = multi_idx.get_level_values(1)?;

// レベルの入れ替え
let swapped_idx = multi_idx.swaplevel(0, 1)?;
```

### Pythonバインディングの使用例

```python
import pandrs as pr
import numpy as np
import pandas as pd

# DataFrameの作成
df = pr.DataFrame()
df.add_int_column('A', [1, 2, 3, 4, 5])
df.add_string_column('B', ['a', 'b', 'c', 'd', 'e'])
df.add_float_column('C', [1.1, 2.2, 3.3, 4.4, 5.5])

# pandasとの相互運用性
pd_df = df.to_pandas()  # PandRSからpandas DataFrameに変換
pr_df = pr.DataFrame.from_pandas(pd_df)  # pandas DataFrameからPandRSに変換

# CSV入出力
df.to_csv('data.csv')
df_loaded = pr.DataFrame.read_csv('data.csv')

# NumPy連携
column = df.get_int_column('A')
np_array = column.to_numpy()

# Jupyter Notebookサポート
from pandrs.jupyter import display_dataframe
display_dataframe(df, max_rows=10, max_cols=5)
```

## 最近の改善

- **列指向ストレージと遅延評価によるパフォーマンス最適化を実装（デフォルト実装に統合）**
  - DataFrame作成が約2.2倍高速化（従来実装と比較）
  - 型特化された列構造でSeries/列作成が約3.6倍高速化
  - グループ化集計が約3.8倍高速化
  - 遅延評価による計算の最適化
  - 大規模データセットでの効率的な処理

- **文字列処理の最適化**
  - グローバル文字列プールによるメモリ使用量の大幅削減
  - カテゴリカルエンコーディングによる効率化
  - 文字列→インデックス変換の1パス処理の実装
  - 重複文字列の排除によるパフォーマンス向上
  
- **並列処理の最適化**
  - 適応的並列化アルゴリズムの導入（データサイズに応じて自動選択）
  - チャンク処理の最適化
  - 効率的なスレッドプール管理

- **Pythonバインディングを実装**
  - PyO3を利用したPythonモジュール
  - NumPyとpandasとの相互運用性
  - Jupyter Notebook連携
  - 高性能なRustバックエンドとPythonの利便性を両立
  - 文字列プール最適化のPythonインターフェース

- **マルチレベルインデックスを実装**
  - 階層的なインデックス構造をサポート
  - タプルからのインデックス生成
  - レベル操作とレベル値の操作
  - DataFrameとの統合
  
- **テキストベースのプロットを用いた可視化機能を実装**
  - 折れ線グラフ、散布図、ポイントプロットに対応
  - ターミナル出力とファイル出力に対応
  - データ可視化の多様なサンプルを追加
  
- **Rayonを使用した並列処理機能を実装**
  - Series/NASeriesの並列マップと並列フィルタリング
  - DataFrameの並列適用と並列フィルタリング
  - 並列集計と並列ソートのユーティリティ関数
  - マルチスレッドによる高性能な処理を実現
  
- **その他の改善**
  - 内部結合、左結合、右結合、外部結合を実装し、DataFrameの結合操作を完成
  - レコード指向JSONおよび列指向JSONの出力を実装
  - ピボットテーブル機能とグループ化操作を強化（集計関数とデータアクセス）
  - RFC3339形式の日付パーシングに対応し、時系列データ操作の互換性を向上
  - 移動平均計算における整数オーバーフロー問題を修正
  - `DAILY`、`WEEKLY`などの完全形式の周波数指定に対応
  - すべてのテストが成功するよう安定性を向上

## ライセンス

Apache License 2.0 で提供されています。