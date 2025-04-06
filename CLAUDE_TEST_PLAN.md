# PandRS テスト計画と実行戦略

大規模なコードベースでは効率的なテスト戦略が必要です。このファイルでは、テスト計画と実行状況を管理します。

## テスト分類と実行コマンド

### 1. モジュール別単体テスト

```bash
# コアモジュールのテスト
cargo test --test dataframe_test
cargo test --test series_test
cargo test --test index_test
cargo test --test na_test

# 機能別テスト
cargo test --test apply_test
cargo test --test transform_test
cargo test --test groupby_test
cargo test --test categorical_test
cargo test --test categorical_na_test
cargo test --test categorical_df_test
cargo test --test temporal_test
cargo test --test window_test
cargo test --test multi_index_test

# I/O関連のテスト
cargo test --test io_test

# ML関連モジュールのテスト
cargo test --test ml_basic_test

# 最適化実装のテスト
cargo test --test optimized_dataframe_test
cargo test --test optimized_series_test
cargo test --test optimized_transform_test
cargo test --test optimized_apply_test
cargo test --test optimized_groupby_test
cargo test --test optimized_io_test
cargo test --test optimized_lazy_test
cargo test --test optimized_multi_index_test
cargo test --test optimized_window_test
```

### 2. 例コード検証

例コードをグループに分割して検証します：

```bash
# グループ1: 基本機能
cargo check --example basic_usage
cargo check --example benchmark_comparison
cargo check --example optimized_basic_usage
cargo check --example string_optimization_benchmark

# グループ2: データ操作
cargo check --example categorical_example
cargo check --example categorical_na_example
cargo check --example na_example
cargo check --example optimized_categorical_example
cargo check --example optimized_transform_example
cargo check --example transform_example

# グループ3: グループ化と集計
cargo check --example groupby_example
cargo check --example optimized_groupby_example
cargo check --example pivot_example

# グループ4: 時系列と窓演算
cargo check --example time_series_example
cargo check --example window_operations_example
cargo check --example optimized_window_example
cargo check --example dataframe_window_example

# グループ5: インデックス操作
cargo check --example multi_index_example
cargo check --example optimized_multi_index_example

# グループ6: ML基本
cargo check --example ml_basic_example
cargo check --example ml_model_example
cargo check --example ml_pipeline_example

# グループ7: ML応用
cargo check --example ml_clustering_example
cargo check --example ml_anomaly_detection_example
cargo check --example ml_dimension_reduction_example
cargo check --example ml_feature_engineering_example

# グループ8: 可視化
cargo check --example visualization_example
cargo check --example visualization_plotters_example
cargo check --example plotters_simple_example
cargo check --example plotters_visualization_example

# グループ9: 並列処理
cargo check --example parallel_example
cargo check --example parallel_benchmark
cargo check --example lazy_parallel_example

# グループ10: その他
cargo check --example parquet_example
cargo check --example performance_bench
cargo check --example benchmark_million
```

## テスト戦略

1. **変更の影響範囲に応じたテスト**:
   - 特定モジュールの変更: そのモジュールのテストと直接関連する例コード
   - API変更: 影響を受けるすべてのモジュールのテスト
   - 基本的な機能追加/修正: 関連機能のテストと例コード

2. **段階的テスト実行**:
   - まず `cargo check` でコンパイルエラーを検出
   - 次に関連する単体テストを実行
   - 最後に関連する例コードを検証

3. **継続的テスト**:
   - PRを作成する前に関連するすべてのテストが通過していることを確認
   - CI/CDでの自動テストを待ってからマージ

## 現在の作業とテスト進捗 (2025/04/06)

### ML関連修正のテスト計画と結果

#### 単体テスト
- [x] cargo test --test ml_basic_test

#### 例コードコンパイルチェック
- [x] cargo check --example ml_dimension_reduction_example
- [x] cargo check --example ml_clustering_example
- [x] cargo check --example ml_anomaly_detection_example
- [x] cargo check --example ml_basic_example
- [x] cargo check --example ml_feature_engineering_example
- [x] cargo check --example ml_pipeline_example
- [ ] cargo check --example ml_model_example (大幅な修正が必要)

#### GitHub Actionsでのテスト
- [ ] 最新コミット(8e35eb7)のCIテスト完了確認

### 修正済みの問題
1. rand APIの更新（0.9.0対応）
   - `gen_range` → `random_range`
   - `thread_rng` → `rng`
   - `random` → `gen`対応

2. コンストラクタパターンの更新
   - `Float64Column::new(values, has_nulls, name)` → `Float64Column::with_name(values, name)`
   - `Int64Column::new(values, has_nulls, name)` → `Int64Column::with_name(values, name)`
   - `StringColumn::new(values, has_nulls, name)` → `StringColumn::with_name(values, name)`

3. 所有権問題の解決
   - 値の移動後に再利用しないよう修正
   - 必要に応じて`clone()`を使用

4. 文字列フォーマット修正
   - 非表示型を持つ値の出力: `{}` → `{:?}`
   - プレースホルダの修正

5. API対応
   - `Transformer`トレイトのインポート追加
   - 新しいAPIパターンへの対応
   - `head()`メソッドの代替処理の実装