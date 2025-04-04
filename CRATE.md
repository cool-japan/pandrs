# PandRS クレートバージョンアップ計画

## 一括アップデートブランチ戦略

このバージョンアップでは、ブランチ戦略を活用して一括でアップデートを実施します。

**ブランチ戦略**:
1. 新しい機能ブランチ `deps-update-2024` を作成
2. このブランチ上ですべてのクレートアップデートを一括実行
3. テスト・修正・最適化を完了
4. 問題がなければプルリクエストを作成してマスターブランチに統合

この方法により、以下の利点が得られます:
- 開発中のメインブランチへの影響なく作業可能
- すべての依存関係を一度に更新して相互作用を早期に発見
- 完全にテストされた状態で統合可能
- 問題が発生しても簡単に破棄可能

## 現在のバージョンと新バージョン

| クレート       | 現在のバージョン | 新バージョン     | 変更の規模    | 対応難易度 |
|--------------|--------------|--------------|------------|--------|
| num-traits   | 0.2.14       | 0.2.19       | パッチ       | 低     |
| chrono       | 0.4.19       | 0.4.40       | マイナー     | 中     |
| serde        | 1.0.x        | 1.0.219      | パッチ       | 低     |
| serde_json   | 1.0.64       | 1.0.114+     | パッチ       | 低     |
| thiserror    | 1.0.24       | 2.0.12       | メジャー     | 高     |
| csv          | 1.1.6        | 1.3.1        | マイナー     | 中     |
| rand         | 0.8.4        | 0.9.0        | マイナー     | 高     |
| chrono-tz    | 0.6.1        | 0.10.3       | マイナー     | 高     |
| textplots    | 0.6.3        | 0.8.7        | マイナー     | 中     |
| rayon        | 1.5.1        | 1.9.0        | マイナー     | 中     |
| regex        | 1.5.4        | 1.10.2       | マイナー     | 中     |
| lazy_static  | 1.4.0        | 1.5.0        | パッチ       | 低     |
| tempfile     | (確認必要)    | 3.8.1        | (確認必要)   | 低     |
| parquet      | (確認必要)    | 54.3.1       | メジャー     | 高     |
| arrow        | (確認必要)    | 54.3.1       | メジャー     | 高     |

## 一括アップデート実行手順

### ステップ 1: 準備とブランチ作成

```bash
# 現在のブランチが最新であることを確認
git checkout master
git pull

# 新しいブランチを作成
git checkout -b deps-update-2024

# 現在の依存関係バージョンを確認
cargo tree > deps-before-update.txt
```

### ステップ 2: 一括依存関係アップデート

すべての依存関係を一度にアップデートする方法:

1. **Cargo.tomlの編集**:
   すべての依存関係を一度に新しいバージョンに更新します。

```toml
# Cargo.toml の例
[dependencies]
num-traits = "0.2.19"
chrono = "0.4.40"
serde = "1.0.219"
serde_json = "1.0.114"
thiserror = "2.0.12"
csv = "1.3.1"
rand = "0.9.0"
chrono-tz = "0.10.3"
textplots = "0.8.7"
rayon = "1.9.0"
regex = "1.10.2"
lazy_static = "1.5.0"
tempfile = "3.8.1"
parquet = "54.3.1"
arrow = "54.3.1"
```

2. **Python Bindings の依存関係も更新**:
```bash
cd py_bindings
# Cargo.toml を同様に更新
```

3. **依存関係の更新**:
```bash
cargo update
cargo check # 初期的な構文チェック
```

### ステップ 3: エラー修正フェーズ

ビルドエラーに優先順位をつけて修正します。重要なのは最もクリティカルな変更（thiserror, parquet/arrow）を最初に対処することです。

1. **コンパイルエラーの収集**:
```bash
cargo build -v > build-errors.log 2>&1
```

2. **エラー修正の優先順位**:
   - まず `error.rs` のthiserrorの変更に対応
   - 次にparquet/arrowのAPIの大きな変更に対応
   - chrono-tzとrandの変更に対応
   - その他のマイナー変更に対応

3. **各モジュールごとの修正**:
   - 各モジュールを個別にチェックしてエラー修正
   ```bash
   cd src/dataframe
   cargo check
   # エラーを修正
   ```

### ステップ 4: 全体のテスト

すべてのコンパイルエラーが修正された後、テストを実行します:

```bash
# 単体テスト
cargo test

# サンプルコードのテスト
for example in $(ls examples/*.rs); do
  cargo run --example $(basename $example .rs)
done

# Python バインディングテスト
cd py_bindings
maturin develop
python -m unittest discover -s tests
```

### ステップ 5: パフォーマンス検証

アップデート前後のパフォーマンスを比較します:

```bash
# ベンチマークを実行
cargo bench

# バインディングのベンチマーク
cd py_bindings
python examples/benchmark.py
python examples/optimized_benchmark.py
```

### ステップ 6: 最終確認と統合

すべてのテストが通り、パフォーマンスが問題ないことを確認したら:

```bash
# 変更をコミット
git add .
git commit -m "Update all dependencies to latest versions"

# プッシュしてPRを作成
git push origin deps-update-2024
```

## クレート別注意点と対応方針

### thiserror (1.0.24 → 2.0.12)

**主な変更点**:
- エラー型定義のマクロ構文が大幅に変更
- `#[source]` 属性の動作変更と新しいエラー伝播オプション
- 表示フォーマットの機能拡張

**対応コード例**:
```rust
// 変更前
#[derive(Error, Debug)]
pub enum DataFrameError {
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

// 変更後
#[derive(Error, Debug)]
pub enum DataFrameError {
    #[error("Column not found: {name}")]
    ColumnNotFound { name: String },
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },
}
```

**src/error.rs** を中心に確認し、すべてのエラー型定義を新しい構文に更新します。

### parquet & arrow (→ 54.3.1)

**主な変更点**:
- 完全に統合されたAPIとデータ構造
- 新しいメモリモデルとバッファ管理
- 型システムの変更とスキーマ定義の更新

**対応方針**:
1. `src/io/parquet.rs` を集中的に確認
2. スキーマ定義とデータ型マッピングの更新
3. 新しいReader/Writerインターフェースへの移行

**対応コード例**:
```rust
// 変更後の例
use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
```

### chrono-tz (0.6.1 → 0.10.3)

**主な変更点**:
- タイムゾーンの指定方法の変更
- `chrono`との統合インターフェースの変更
- タイムゾーンデータベースの更新

**確認箇所**:
- `src/temporal/` ディレクトリ全体
- 特に `frequency.rs` と `date_range.rs`

### rand (0.8.4 → 0.9.0)

**主な変更点**:
- 乱数生成APIの変更
- 分布関数の更新
- シード管理の改善

**確認箇所**:
- サンプルデータ生成部分
- テストコード内の乱数使用部分

## 一括アップデートの利点と注意点

### 利点:
- 複数の依存関係間の互換性問題を一度に発見して解決
- 段階的な更新よりも効率的に作業を完了
- 新機能やパフォーマンス向上を一度に活用可能

### 注意点:
- 問題の切り分けが複雑になる可能性
- コンパイルエラーが多数発生する可能性
- 依存関係間の非互換性のリスク

### 対策:
- 詳細なエラーログの保持
- git diffを活用した変更の追跡
- 変更内容の文書化
- テスト網羅率の向上

## テスト戦略

一括アップデートに対応したテスト戦略:

1. **単体テスト**:
   - 既存テストの全実行
   - 変更した機能に対する新テストの追加

2. **統合テスト**:
   - すべてのサンプルコードの実行
   - エンドツーエンドのワークフロー検証

3. **並列作業**:
   - 基本的な機能のテスト
   - Pythonバインディングのテスト
   - パフォーマンステスト

4. **回帰テスト**:
   - 既知の動作が保持されていることを確認
   - エッジケースのカバー

## ドキュメント

- 変更した依存関係ごとにコード変更の詳細をドキュメント化
- パフォーマンス測定結果の記録
- 修正に時間がかかった箇所やトリッキーな問題の文書化
- 新しい機能や改善された挙動の記録

## まとめ

この一括アップデート戦略により、PandRSのすべての依存関係を効率的に最新化します。メインブランチへの影響なく作業を行い、テストを完了させた上で安全に統合することが可能です。大きな変更（thiserror、parquet/arrow）に注意しながらも、一度の作業で完了させることでメンテナンスの効率を大幅に向上させます。