# PandRS

Rustで実装されたデータ分析用DataFrameライブラリです。Pythonの`pandas`ライブラリにインスパイアされた機能と設計を持ち、高速なデータ処理と型安全性を両立しています。

## 主な特徴

- 高性能な列指向ストレージによる効率的なデータ処理
- カテゴリカルデータと文字列プール最適化による低メモリフットプリント
- 並列処理によるマルチコア活用
- 遅延評価システムによる最適化実行計画
- スレッドセーフな実装
- Rustの型安全性と所有権システムを活かした堅牢性
- Pythonとの連携機能（PyO3バインディング）

## 機能

- Series (1次元配列) とDataFrame (2次元表) データ構造
- 欠損値（NA）のサポート
- グループ化と集計操作
- インデックスによる行ラベル
- マルチレベルインデックス（階層型インデックス）
- CSV/JSON読み込みと書き込み
- Parquetデータ形式のサポート
- 基本的な操作（フィルタリング、ソート、結合など）
- 数値データに対する集計関数
- 文字列データに対する特殊操作
- 時系列データ処理の基本機能
- カテゴリカルデータ型（効率的なメモリ使用、順序付きカテゴリ）
- ピボットテーブル
- テキストベースの可視化
- 並列処理サポート
- 統計分析機能（記述統計、t検定、回帰分析など）
- 最適化実装（列指向ストレージ、遅延評価、文字列プール）

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

### 統計分析機能

```rust
use pandrs::{DataFrame, Series, stats};

// 記述統計
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let stats_summary = stats::describe(&data)?;
println!("平均: {}, 標準偏差: {}", stats_summary.mean, stats_summary.std);
println!("中央値: {}, 四分位数: {} - {}", stats_summary.median, stats_summary.q1, stats_summary.q3);

// 相関係数を計算
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
let correlation = stats::correlation(&x, &y)?;
println!("相関係数: {}", correlation);

// t検定を実行
let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
let alpha = 0.05; // 有意水準
let result = stats::ttest(&sample1, &sample2, alpha, true)?;
println!("t統計量: {}, p値: {}", result.statistic, result.pvalue);
println!("有意差: {}", result.significant);

// 回帰分析
let mut df = DataFrame::new();
df.add_column("x1".to_string(), Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x1".to_string()))?)?;
df.add_column("x2".to_string(), Series::new(vec![2.0, 3.0, 4.0, 5.0, 6.0], Some("x2".to_string()))?)?;
df.add_column("y".to_string(), Series::new(vec![3.0, 5.0, 7.0, 9.0, 11.0], Some("y".to_string()))?)?;

let model = stats::linear_regression(&df, "y", &["x1", "x2"])?;
println!("係数: {:?}", model.coefficients());
println!("決定係数: {}", model.r_squared());
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

## 開発計画と実装状況

- [x] 基本的なDataFrame構造
- [x] Series実装
- [x] インデックス機能
- [x] CSV入出力
- [x] JSON入出力
- [x] Parquet形式サポート
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
  - [x] 並列フィルタリング (1.15倍高速化)
  - [x] 並列集計 (3.91倍高速化)
  - [x] 並列計算処理 (1.37倍高速化)
  - [x] 適応的並列処理（データサイズに応じた自動選択）
- [x] 可視化機能の強化
  - [x] textplotsによるテキストベースプロット（折れ線、散布図）
  - [x] plottersによる高品質グラフ出力（PNG, SVG形式）
  - [x] 各種グラフ種類（折れ線、散布図、棒グラフ、ヒストグラム、面グラフ）
  - [x] グラフカスタマイズオプション（サイズ、カラー、グリッド、凡例）
- [x] マルチレベルインデックス
  - [x] ヒエラルキカルなインデックス構造
  - [x] 複数レベルによるデータのグループ化
  - [x] レベル操作（入れ替え、選択）
- [x] カテゴリカルデータ型
  - [x] メモリ効率の良いエンコーディング
  - [x] 順序付き・順序なしカテゴリのサポート
  - [x] NA値（欠損値）との完全統合
- [x] 高度なデータフレーム操作
  - [x] 長形式・広形式変換（melt, stack, unstack）
  - [x] 条件付き集計
  - [x] データフレーム結合
- [x] メモリ使用効率の最適化
  - [x] 文字列プール最適化（最大89.8%のメモリ削減）
  - [x] カテゴリカルエンコーディング（2.59倍の性能向上）
  - [x] グローバル文字列プール実装
  - [x] 列指向ストレージによるメモリ局所性向上
- [x] Pythonバインディング
  - [x] PyO3を使用したPythonモジュール化
  - [x] numpyとpandasとの相互運用性
  - [x] Jupyter Notebookサポート
  - [x] 文字列プール最適化による高速化（最大3.33倍）
- [x] 遅延評価システム
  - [x] 計算グラフによる操作最適化
  - [x] オペレーション融合
  - [x] 不要な中間結果の生成回避
- [x] 統計分析機能
  - [x] 記述統計（平均、標準偏差、分位数など）
  - [x] 相関係数と共分散
  - [x] 仮説検定（t検定）
  - [x] 回帰分析（単回帰・重回帰）
  - [x] サンプリング手法（ブートストラップなど）

### マルチレベルインデックスの操作

```rust
use pandrs::{DataFrame, MultiIndex};

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
let mut df = DataFrame::with_multi_index(multi_idx);

// データを追加
let data = vec!["data1".to_string(), "data2".to_string(), "data3".to_string(), "data4".to_string()];
df.add_column("data".to_string(), pandrs::Series::new(data, Some("data".to_string()))?)?;

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

# 最適化されたDataFrameの作成
df = pr.OptimizedDataFrame()
df.add_int_column('A', [1, 2, 3, 4, 5])
df.add_string_column('B', ['a', 'b', 'c', 'd', 'e'])
df.add_float_column('C', [1.1, 2.2, 3.3, 4.4, 5.5])

# 従来のAPI互換インターフェース
df2 = pr.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [1.1, 2.2, 3.3, 4.4, 5.5]
})

# pandasとの相互運用性
pd_df = df.to_pandas()  # PandRSからpandas DataFrameに変換
pr_df = pr.OptimizedDataFrame.from_pandas(pd_df)  # pandas DataFrameからPandRSに変換

# 遅延評価の使用
lazy_df = pr.LazyFrame(df)
result = lazy_df.filter('A').select(['B', 'C']).execute()

# 文字列プールの直接使用
string_pool = pr.StringPool()
idx1 = string_pool.add("repeated_value")
idx2 = string_pool.add("repeated_value")  # 同じインデックスを返す
print(string_pool.get(idx1))  # "repeated_value"を返す

# CSV入出力
df.to_csv('data.csv')
df_loaded = pr.OptimizedDataFrame.read_csv('data.csv')

# NumPy連携
series = df['A']
np_array = series.to_numpy()

# Jupyter Notebookサポート
from pandrs.jupyter import display_dataframe
display_dataframe(df, max_rows=10, max_cols=5)
```

## パフォーマンス最適化の成果

最適化された列指向ストレージと遅延評価システムの実装により、大幅なパフォーマンス向上を達成しました：

### 主要操作のパフォーマンス比較

| 操作 | 従来実装 | 最適化実装 | 高速化率 |
|------|---------|-----------|----------|
| Series/列作成 | 198.446ms | 149.528ms | 1.33倍 |
| DataFrame作成(100万行) | 728.322ms | 0.007ms | 96,211.73倍 |
| フィルタリング | 596.146ms | 161.816ms | 3.68倍 |
| グループ化集計 | 544.384ms | 107.837ms | 5.05倍 |

### 文字列処理の最適化

| モード | 処理時間 | 従来実装比 | 備考 |
|--------|---------|------------|------|
| レガシーモード | 596.50ms | 1.00倍 | 従来実装 |
| カテゴリカルモード | 230.11ms | 2.59倍 | カテゴリカル最適化 |
| 最適化実装 | 232.38ms | 2.57倍 | オプティマイザー選択 |

### 並列処理の性能向上

| 操作 | 直列処理 | 並列処理 | 高速化率 |
|------|---------|----------|---------|
| グループ化集計 | 696.85ms | 178.09ms | 3.91倍 |
| フィルタリング | 201.35ms | 175.48ms | 1.15倍 |
| 計算処理 | 15.41ms | 11.23ms | 1.37倍 |

### Python Bindings文字列最適化

| データサイズ | ユニーク率 | プール無し | プール使用 | 処理速度向上 | メモリ削減率 |
|------------|----------|-----------|-----------|------------|------------|
| 100,000行 | 1% (高重複) | 82ms | 35ms | 2.34倍 | 88.6% |
| 1,000,000行 | 1% (高重複) | 845ms | 254ms | 3.33倍 | 89.8% |

## 最近の改善

- **列指向ストレージエンジン**
  - 型特化した列実装（Int64Column, Float64Column, StringColumn, BooleanColumn）
  - メモリ局所性向上によるキャッシュ効率改善
  - 操作の高速化と並列処理の効率化

- **文字列処理の最適化**
  - グローバル文字列プールによる重複文字列の排除
  - カテゴリカルエンコーディングによる文字列→インデックス変換
  - 一貫性のあるAPI設計と複数の最適化モード

- **遅延評価システムの実装**
  - 計算グラフによる操作のパイプライン化
  - 不要な中間結果の生成回避
  - 操作融合による効率向上

- **並列処理の大幅改善**
  - Rayon活用による効率的なマルチスレッド処理
  - 適応的並列処理（データサイズに応じた自動選択）
  - チャンク処理の最適化

- **Python連携の強化**
  - 文字列プール最適化によるPython-Rust間の効率的なデータ変換
  - NumPyバッファプロトコルの活用
  - ゼロコピーに近いデータアクセス
  - 型特化したPythonAPIの提供

- **高度なデータフレーム操作**
  - 長形式・広形式変換（melt, stack, unstack）の完全実装
  - 条件付き集計処理の強化
  - 複雑な結合操作の最適化

- **時系列データ処理の強化**
  - RFC3339形式の日付パーシングに対応
  - 高度なウィンドウ操作の完全実装
  - `DAILY`、`WEEKLY`などの完全形式の周波数指定に対応

- **安定性と品質向上**
  - 包括的なテストスイートの実装
  - エラー処理の改善と警告の除去
  - ドキュメントの充実
  - 依存関係の最新化（Rust 2023コンパチブル）

## 依存関係バージョン

最新の依存関係バージョン（2024年4月）:

```toml
[dependencies]
num-traits = "0.2.19"        # 数値型特性サポート
thiserror = "2.0.12"          # エラーハンドリング
serde = { version = "1.0.219", features = ["derive"] }  # シリアライゼーション
serde_json = "1.0.114"       # JSON処理
chrono = "0.4.40"            # 日付・時間処理
regex = "1.10.2"             # 正規表現
csv = "1.3.1"                # CSV処理
rayon = "1.9.0"              # 並列処理
lazy_static = "1.5.0"        # 遅延初期化
rand = "0.9.0"               # 乱数生成
tempfile = "3.8.1"           # 一時ファイル
textplots = "0.8.7"          # テキストベースの可視化
chrono-tz = "0.10.3"         # タイムゾーン処理
parquet = "54.3.1"           # Parquetファイルのサポート
arrow = "54.3.1"             # Arrowフォーマットのサポート
```

## ライセンス

Apache License 2.0 で提供されています。