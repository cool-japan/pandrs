# PandRS リファクタリング計画

## 実施状況（2025/4/6）

以下の作業を開始しました：

1. **リファクタリング計画を作成**:
   - 既存APIを維持しながら、実装を機能別に分割する方針を定義
   - 委譲パターンによる段階的移行を計画

2. **重複機能の特定**:
   - インデックス操作（`set_simple_index`, `get_index`など）が両方の実装に存在
   - 列操作（`rename_column`, `add_column`など）が両方の実装に存在

3. **移行準備**:
   - インデックス関連メソッドにTODOコメントを追加
   - 列操作関連メソッドにTODOコメントを追加
   - split_dataframe/index.rsに必要なメソッドを追加（`set_index_from_simple_index`）

4. **今後の実装**:
   - 結合機能など新規機能は、split_dataframeディレクトリに実装
   - dataframe.rsからは薄いラッパーで呼び出す

## 背景

OptimizedDataFrameの実装が`dataframe.rs`ファイルに集中し、ファイルが大きくなりすぎているため、管理が難しくなっています。一方で、`split_dataframe/`ディレクトリが既に用意されており、機能別に分割された実装を格納する構造が整えられています。

このリファクタリングでは、既存のAPIを維持しながら、実装を機能別に分割することを目指します。

## 方針

1. **既存APIはそのまま維持**
   - `dataframe.rs`の公開APIは変更せず、既存のコードやサンプルが引き続き動作するようにします
   - 新しい機能を追加する場合も、同じパターンに従います

2. **内部実装を移動**
   - `dataframe.rs`の各メソッドの内部実装を`split_dataframe/`の適切なファイルに移動します
   - 機能ごとに分類し、対応するファイルに実装を移動します

3. **委譲パターン**
   - `dataframe.rs`のメソッドは`split_dataframe/`の実装を呼び出すだけの薄いラッパーにします
   - これにより、実装の詳細を`split_dataframe/`に隠蔽しつつ、外部APIを維持します

## 実装パターン例

```rust
// dataframe.rs
impl OptimizedDataFrame {
    pub fn some_method(&self, args) -> Result<T> {
        // split_dataframe/module.rs の実装を呼び出すだけ
        use crate::optimized::split_dataframe::module;
        module::some_method_impl(self, args)
    }
}
```

```rust
// split_dataframe/module.rs
pub(crate) fn some_method_impl(df: &OptimizedDataFrame, args) -> Result<T> {
    // 実際の実装
    // ...
}
```

## 機能分類と移動先ファイル

| 機能カテゴリ | 対象メソッド | 移動先ファイル |
|------------|------------|-------------|
| 基本構造体とコンストラクタ | `new()`, `with_index()`, `with_multi_index()`, `with_range_index()` | `core.rs` |
| インデックス操作 | `set_multi_index()`, `set_column_as_index()`, `set_simple_index()`, `get_index()` | `index.rs` |
| 列操作 | `add_column()`, `rename_column()`, `remove_column()` | `column_ops.rs` |
| データ変換 | `stack()`, `unstack()`, `conditional_aggregate()` | `data_ops.rs` |
| 結合機能 | `inner_join()`, `left_join()`, `right_join()`, `outer_join()` | `join.rs` |
| 入出力 | `to_csv()`, `from_csv()`, `to_json()`, `from_json()`, `to_parquet()`, `from_parquet()` | `io.rs` |
| グループ化と集計 | `groupby()`, `par_groupby()` | `group.rs` |
| 統計機能 | `describe()`, `corr()` | `stats.rs` |

## 移行戦略

1. **段階的アプローチ**
   - 新しい機能（例：結合機能）から先に実装し、このパターンを適用する
   - 既存機能は徐々に移行する

2. **テスト戦略**
   - 各機能移行後に対応するテストを実行し、挙動が変わっていないことを確認する
   - リファクタリング専用のテストを追加し、API互換性を検証する

3. **ドキュメント**
   - コードコメントを更新し、内部実装が移動したことを記録する
   - 新しい機能追加時のコーディングスタイルガイドを更新する

## 期待される効果

1. コードの管理が容易になる
2. 機能ごとの関心の分離が実現する
3. 複数のファイルに分割されることで、コンテキスト内で処理しやすくなる
4. 既存の外部APIは維持されるため、移行コストがかからない

## 対応する具体的な移行計画

この計画は`MIGRATION_STRATEGY.md`で定められている移行計画と連携して実施します。
特にフェーズ1（ファサードパターンの実装）に関連する作業と並行して進めることができます。