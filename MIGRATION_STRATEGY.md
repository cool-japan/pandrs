# OptimizedDataFrame のリファクタリング戦略

PandRSプロジェクトでは、コードベースの肥大化に対処し、より保守性の高い構造に改善するためのリファクタリングを行います。特に`OptimizedDataFrame`クラスは約2600行と非常に大きくなっているため、モジュール分割と責務の明確化が必要です。

## 実装進捗

### 完了した移行

1. **入出力機能** (2024-04-06)
   - CSV、Parquet、Excel関連の操作を`split_dataframe/io.rs`へ移行
   - dataframe.rsのサイズを2694行から2644行に削減 (-1.9%)
   - すべてのIO関連テストが正常に動作

2. **グループ化操作** (2024-04-06)
   - `par_groupby`機能を`split_dataframe/group.rs`へ移行
   - dataframe.rsのサイズを2644行から2501行に削減 (-5.4%)
   - すべてのグループ化関連テストが正常に動作

3. **結合操作** (2024-04-06)
   - 内部結合、左結合、右結合、外部結合の操作を`split_dataframe/join.rs`へ移行
   - 457行のjoin_implメソッドを完全に削除
   - 結合操作のテスト（6種類）を新規追加
   - dataframe.rsのサイズを2501行から2286行に削減 (-8.6%)

4. **データ変換操作** (2024-04-06)
   - melt操作（広形式から長形式への変換）と append操作（縦方向結合）を`split_dataframe/data_ops.rs`へ移行
   - 526行を委譲パターンに置き換え
   - すべての変換操作関連テストが正常に動作
   - dataframe.rsのサイズを2286行から1884行に削減 (-17.6%)

5. **列操作機能の強化** (2024-04-06)
   - 列追加・削除・名前変更・参照などの操作を`split_dataframe/column_ops.rs`との連携で強化
   - 委譲パターンを用いた実装に更新
   - add_int_column、add_float_column、add_string_column、add_boolean_column等の専用メソッドを追加
   - remove_column、rename_column、get_valueメソッドの追加
   - データ型特化操作のサポート強化
   - すべての列操作関連テストが正常に動作
   - dataframe.rsのサイズを1884行から2018行に増加 (+7.1%)（新機能追加のため）

6. **インデックス操作の強化** (2024-04-06)
   - インデックス操作を`split_dataframe/index.rs`との連携で強化
   - set_index、get_index、set_default_index、set_index_directly、set_index_from_simple_indexの実装
   - reset_index（インデックスを列として追加する機能）の実装
   - get_row_by_index、select_by_indexなどインデックスを使用した行取得メソッドの実装
   - すべてのインデックス関連テストが正常に動作
   - dataframe.rsのサイズを2018行から2228行に増加 (+10.4%)（新機能追加のため）

7. **行操作機能の移行** (2024-04-07)
   - 行操作機能を`split_dataframe/row_ops.rs`へ移行
   - フィルタリング（filter → filter_rows）
   - 行選択（select_row_by_idx、select_rows_by_indices）、行取得（get_row）の実装
   - filter_by_indices、filter_by_mask、select_by_mask、insert_rowなどの新機能追加
   - すべての行操作関連テストが正常に動作
   - dataframe.rsのサイズをさらに削減

8. **選択操作の移行** (2024-04-07)
   - 列選択機能を`split_dataframe/select.rs`へ移行
   - select_columnsメソッドの完全実装
   - 存在しない列を指定した場合のエラーハンドリングを改善
   - すべての選択操作関連テストが正常に動作

9. **集計操作の移行** (2024-04-07)
   - 集計機能を`split_dataframe/aggregate.rs`へ移行
   - sum、mean、min、max、std、varなどの基本集計関数
   - 集計操作のメソッドチェーンサポート
   - すべての集計関連テストが正常に動作

10. **適用操作の移行** (2024-04-07)
    - 関数適用機能を`split_dataframe/apply.rs`へ移行
    - map_valuesとapply_function系メソッドの実装
    - 異なる型への変換をサポートする適用操作
    - すべての適用関連テストが正常に動作

11. **並列処理の強化** (2024-04-07)
    - 並列処理機能を`split_dataframe/parallel.rs`へ移行
    - 並列フィルタリング、並列マッピング、並列集計の実装
    - Rayonを使用した効率的な並列処理
    - すべての並列処理関連テストが正常に動作

12. **ソート機能の実装** (2024-04-07)
    - ソート機能を`split_dataframe/sort.rs`へ実装
    - 単一列ソート、複数列ソート、カスタムソートのサポート
    - 昇順・降順ソートオプション
    - ソートインデックスの生成と適用
    - すべてのソート関連テストが正常に動作

13. **JSONシリアライズ機能の強化** (2024-04-07)
    - JSON操作を`split_dataframe/serialize.rs`へ移行
    - レコード形式とカラム形式のJSONシリアライズ
    - デシリアライズの最適化
    - すべてのシリアライズ関連テストが正常に動作

14. **型変換とコード整理** (2024-04-07)
    - 型変換機能を`convert.rs`モジュールに移行
    - 標準DataFrameとOptimizedDataFrame間の相互変換
    - 全コードベースにおけるエラー処理とパターンマッチングの最適化
    - コンポーネント間の依存性を整理
    - dataframe.rsのサイズを2228行から1870行に削減 (-16.1%)
    - コード全体のクリーンアップと重複コードの削除
15. **コンバート機能の実装と完了** (2024-04-07)
    - 型変換機能と相互運用性の向上
    - 動的なパターンマッチング処理の最適化
    - エラーハンドリングの統一と強化
    - 依存関係の明確化とコード整理
    - リファクタリング完了による保守性の大幅向上
    - ビルドエラーとすべての警告の修正
    - すべてのテストの正常動作確認

**現在の削減率**: 元のサイズ2694行 → 現在2363行 (約12.3%削減)

## リファクタリング方針

### 1. モジュール分割によるコード整理

現在の`dataframe.rs`（約2500行）を以下の方針でリファクタリングします：

1. **機能ごとの分離**: 機能ごとに分割された`split_dataframe/`ディレクトリの実装を活用
   - 列操作: `column_ops.rs`
   - データ操作: `data_ops.rs`
   - インデックス操作: `index.rs`
   - 入出力: `io.rs` ✅
   - 結合操作: `join.rs` ✅
   - 統計処理: `stats.rs`
   - グループ化: `group.rs` ✅
   - 列ビュー: `column_view.rs`
   - コア機能: `core.rs`

2. **段階的アプローチ**:
   - 一度に全てを変更するのではなく、機能ブロックごとに段階的に移行
   - 各段階でテストを実行して機能の正常性を確認
   - 一定の機能が移行できたらコミットして進捗を記録

3. **トークン制限への配慮**:
   - AIツール(Claude Code)の25000トークン制限を考慮し、大きなファイルは部分的に処理
   - 一度に処理する機能ブロックを限定的にする

### 2. 具体的な実施手順

#### フェーズ1: `dataframe.rs`のスリム化

1. **基本データ構造の定義を維持**:
   - 構造体定義と基本コンストラクタは`dataframe.rs`に残す
   - 基本的なアクセサメソッドも`dataframe.rs`に残す

2. **次の移行予定**:
   - ✅ 入出力機能（CSV、JSON、Parquet、Excel、SQL）→ `split_dataframe/io.rs`
   - ✅ グループ化と集計 → `split_dataframe/group.rs`
   - ✅ 結合操作 → `split_dataframe/join.rs`
   - ✅ データ変換（melt、append）→ `split_dataframe/data_ops.rs`
   - ✅ 列操作（追加、削除、名前変更、値取得）→ `split_dataframe/column_ops.rs`
   - ✅ インデックス操作（設定、取得、選択）→ `split_dataframe/index.rs`
   - ✅ 行操作（フィルタリング、選択、head、tail、sample）→ `split_dataframe/row_ops.rs`
   - ✅ 統計処理 → `split_dataframe/stats.rs`（すでに完全に実装済み）
   - ✅ 関数適用（apply, applymap, par_apply）→ `split_dataframe/apply.rs`
   - ✅ 並列処理（par_filter）→ `split_dataframe/parallel.rs`
   - ✅ 選択操作（select, filter_by_indices）→ `split_dataframe/select.rs`
   - ✅ 集計操作（sum, mean, count, min, max）→ `split_dataframe/aggregate.rs`
   - ✅ ソート操作（sort_by, sort_by_columns）→ `split_dataframe/sort.rs`
   - ✅ シリアライズ操作（to_json, from_json）→ `split_dataframe/serialize.rs`
   - 次の候補:
     - 説明的アクセサメソッド（describe, info）→ `split_dataframe/describe.rs`

3. **移行プロセス**:
   - 各機能を`split_dataframe/`の対応するファイルに実装
   - `dataframe.rs`から対応する実装を削除またはシンプルな委譲に置き換え
   - 共通ユーティリティの識別と抽出

#### フェーズ2: テストと最適化

1. **テスト強化**:
   - 各モジュールに対応するテストの確認と追加
   - エッジケースの検証
   - パフォーマンス測定

2. **パフォーマンス最適化**:
   - モジュール間の相互作用の最適化
   - クリティカルパスの識別と改善
   - メモリ使用量の削減

### 3. コードの品質向上

1. **一貫性のある設計**:
   - 明確なモジュール境界と責務の定義
   - 一貫性のあるエラーハンドリング
   - 型安全性の維持

2. **ドキュメント**:
   - 各モジュールの目的と責務の明確化
   - パブリックAPIとモジュール間インターフェースの文書化
   - コード例の更新

## 期待される成果

1. **保守性の向上**:
   - 2600行の巨大なファイルが複数の小さなモジュールに分割
   - 機能ごとの責務が明確化され、理解しやすく
   - 変更の影響範囲が局所化

2. **拡張性の向上**:
   - 新機能の追加が特定のモジュールに集中
   - モジュール間の依存関係が明確

3. **パフォーマンスの改善**:
   - 機能ごとの最適化が容易
   - メモリ使用量の削減

4. **コードの信頼性向上**:
   - テストカバレッジの向上
   - エラー処理の一貫性

## 実装上の注意点

1. **後方互換性**: 既存のAPIと互換性を維持
2. **段階的実装**: 一度にすべてを変更せず、機能ブロックごとに移行
3. **テスト駆動**: 各変更後にテストを実行して機能の正常性を確認
4. **トークン制限への対応**: AIツール(Claude Code)の25000トークン制限を考慮した分割アプローチ

## スケジュール

1. **初期評価と計画**: 1週間
2. **フェーズ1（機能移行）**: 2-3週間
3. **フェーズ2（テストと最適化）**: 1-2週間
4. **最終レビューと調整**: 1週間

合計: 約5-7週間

このリファクタリング計画により、コードベースの品質と保守性が大幅に向上し、将来の拡張や機能追加が容易になります。
