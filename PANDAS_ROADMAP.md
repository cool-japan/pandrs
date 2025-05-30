# PandRS ロードマップ - pandas相当機能の実装計画

このロードマップは、PythonのPandasライブラリにインスパイアされた機能をRustで実装するためのガイドラインです。

## 現在実装済みの機能

- Series (1次元配列) とDataFrame (2次元表) データ構造
- 欠損値（NA）のサポート
- グループ化操作
- インデックスによる行ラベル
- CSV/JSON読み込みと書き込み
- Parquetデータ形式のサポート（依存関係追加済み、実装は今後拡充予定）
- 基本的な操作（フィルタリング、ソート、結合など）
- 数値データに対する集計関数
- 時系列データ処理の基本機能
- ピボットテーブル
- テキストベースの可視化
- 並列処理サポート
- カテゴリカルデータ型
- マルチレベルインデックス
- 文字列プール最適化
- 高性能な列指向ストレージ実装
- 統計機能（記述統計、相関/共分散、t検定、分散分析、ノンパラメトリック検定、カイ二乗検定、回帰分析、サンプリング）

## 短期実装目標 (1-3ヶ月)

### 統計機能の強化と拡充（2024年5-6月）

1. **既存統計モジュール（stats/）の拡充**
   - ✅ 記述統計機能（標本分散・標準偏差、分位数）
   - ✅ 共分散・相関分析
   - ✅ 仮説検定（t検定）
   - ✅ 回帰分析の基礎（単回帰・重回帰、最小二乗法）
   - ✅ サンプリング手法（ブートストラップ）
   - ✅ 分散分析（一元配置ANOVA）の実装
   - ✅ ノンパラメトリック検定（Mann-Whitney U検定）
   - ✅ カイ二乗検定の実装
   - ✅ 信頼区間と予測区間の強化

2. **既存機能との連携強化**
   - ✅ 独立したAPI関数としての提供
   - ✅ 公開APIインターフェースの整備（linear_regression関数の公開等）
   - ✅ DataFrameとSeriesに統計メソッドを追加
   - ✅ 並列処理との統合のためのインターフェース設計
   - ✅ 最適化実装（optimized/）との統合
   - カテゴリカルデータに対する特化統計処理

3. **機械学習評価指標モジュール**
   - ✅ 回帰モデル評価指標（MSE、MAE、RMSE、R²スコア）
   - ✅ 分類モデル評価指標（精度、適合率、再現率、F1スコア）
   - ✅ エラーハンドリングとドキュメンテーション

### データ構造と操作の拡張

1. ✅ **MultiIndex（マルチレベルインデックス）**
   - ✅ 階層型インデックス構造
   - ✅ レベルベースのデータアクセス
   - ✅ インデックスの入れ替え操作

2. ✅ **Categoricalデータ型**
   - ✅ カテゴリカルデータの効率的な表現
   - ✅ 順序付きカテゴリカルデータのサポート
   - ✅ カテゴリカルデータの操作（変換、集計など）

3. ✅ **データフレーム操作の拡張**
   - ✅ `apply`/`applymap` 相当の関数適用機能
   - ✅ 条件付き置換（`where`/`mask`/`replace`）
   - ✅ 重複行の検出と削除の改善

### データ入出力の拡充

1. ✅ **Excelサポート**
   - ✅ xlsxファイルの読み込みと書き込み
   - ✅ シート指定と操作
   - ✅ 基本的なExcel出力機能

2. ✅ **SQLインターフェース**
   - ✅ SQLiteからの読み込み（SQL文による問い合わせ）
   - ✅ SQLiteへの書き込み
   - ✅ 既存テーブルへの追加/置換オプション

3. ✅ **Parquetおよび列指向形式サポート**
   - ✅ 依存関係の追加（arrow 54.3.1, parquet 54.3.1）
   - ✅ Parquetファイルの読み込みと書き込み
   - ✅ 圧縮オプション（Snappy, GZIP, Brotli, LZO, LZ4, Zstd）
   - ✅ 列指向データ構造と統合

### 時系列データ処理の強化

1. ✅ **周期的インデックスの強化**
   - ✅ カスタム頻度（営業日など）
   - ✅ 四半期・年度計算のサポート
   - ✅ カレンダー機能の拡張（chrono-tz 0.10.3）

2. ✅ **時系列固有の操作**
   - ✅ 季節性分解
   - ✅ 移動平均の種類拡張
   - ✅ 時系列のシフトと差分演算の最適化

## 中期実装目標 (4-8ヶ月)

### 高度な分析機能

1. ✅ **ウィンドウ操作**
   - ✅ 固定・拡張・可変ウィンドウ処理
   - ✅ ウィンドウ集計関数
   - ✅ ローリング統計の多様化

2. **統計関数の充実**
   - ✅ 相関係数と共分散
   - ✅ 仮説検定（t検定）
   - ✅ 標本抽出と乱数生成（rand 0.9.0）
   - ✅ 回帰分析の基礎（単回帰・重回帰）
   - 🔄 高度な統計手法（仮説検定の拡充、ノンパラメトリック検定）

3. ✅ **文字列操作の強化**
   - ✅ 正規表現ベースの検索・置換（regex 1.10.2）
   - ✅ 文字列ベクトル操作の効率化
   - ✅ テキスト処理ユーティリティ

### データ可視化の拡充

1. ✅ **Plottersとの統合**
   - ✅ 高品質な可視化ライブラリ (plotters v0.3.7) の統合
   - ✅ PNGとSVG形式の出力サポート
   - ✅ グラフ種類の拡充（折れ線、棒、散布図、ヒストグラム、面グラフ）
   - ✅ カスタマイズオプション（サイズ、色、グリッド、凡例）
   - 🔄 DataFrame/Seriesからの直接プロット（一部実装済み）

2. **インタラクティブ可視化**
   - 🔄 WebAssemblyサポートによるブラウザ可視化（初期段階）
   - ダッシュボード機能
   - 動的グラフ生成

### メモリとパフォーマンスの最適化

1. ✅ **メモリ使用量の最適化**
   - ✅ ゼロコピー操作の追加
   - ✅ 列指向ストレージの最適化
   - 大規模データセットのディスクベース処理

2. ✅ **並列処理の強化**
   - ✅ データフレームレベルの並列処理（rayon 1.9.0）
   - ✅ 操作チェーンの並列最適化
   - GPUアクセラレーション（cuNDとの連携）

3. ✅ **コードベース最適化**
   - ✅ OptimizedDataFrameの機能別ファイル分割
   - ✅ コア機能、列操作、データ操作などの最適な分割
   - ✅ API互換性を確保した再エクスポート
   - 🔄 より精緻なモジュール構造の整理

## 長期実装目標 (9ヶ月以上)

### 高度なデータサイエンス機能

1. **機械学習との連携**
   - ✅ scikit-learn相当のデータ変換パイプライン
   - ✅ 特徴量エンジニアリング機能
     - ✅ 多項式特徴量生成
     - ✅ ビニング（離散化）
     - ✅ 欠損値補完
     - ✅ 特徴量選択
   - ✅ モデル学習・評価用のユーティリティ
     - ✅ 線形回帰・ロジスティック回帰モデル
     - ✅ モデル選択（交差検証、グリッドサーチ）
     - ✅ モデル評価指標
     - ✅ モデルの保存と読み込み

2. **次元削減と探索的データ分析**
   - ✅ PCA、t-SNEなどの実装
     - ✅ 主成分分析（PCA）
     - ✅ t-分布確率的近傍埋め込み（t-SNE）
   - ✅ クラスタリング機能
     - ✅ k-meansクラスタリング
     - ✅ 階層的クラスタリング
     - ✅ DBSCAN（密度ベースクラスタリング）
   - ✅ 異常検出
     - ✅ Isolation Forest（孤立森）
     - ✅ LOF（Local Outlier Factor）
     - ✅ One-Class SVM

3. 🔄 **大規模データ処理**
   - ✅ チャンク処理機能
   - 🔄 ストリーミングデータサポート
   - 分散処理フレームワークとの連携

### エコシステム連携

1. ✅ **Pythonバインディング**
   - ✅ PyO3を使用したPythonモジュール化
   - ✅ numpyとpandasとの相互運用性
   - ✅ Jupyter Notebookサポート

2. **R言語との連携**
   - RとRustの間の相互運用
   - tidyverseスタイルのインターフェース

3. **データベース統合**
   - 主要データベース向けのコネクタ
   - クエリオプティマイザ
   - ORM的機能

## 実装アプローチ

1. **段階的実装方針**
   - 最初にAPIを設計し、docテストを作成
   - 基本機能をシンプルに実装
   - パフォーマンスを段階的に最適化

2. **ユーザビリティ重視**
   - Pythonのpandasに慣れた人が直感的に使えるAPI
   - 型安全性とRustの強みを活かしたAPI設計
   - 充実したドキュメントとサンプル

3. **テスト戦略**
   - 各機能に対応する単体テスト
   - pandasとの互換性テスト
   - ベンチマークによるパフォーマンステスト

## 次のステップ

1. **コミュニティ構築**
   - 貢献ガイドラインの策定
   - ミルストーンとイシューの整理
   - 初心者向けイシューの作成

2. **ドキュメント整備**
   - APIドキュメントの充実
   - チュートリアルとクックブックの作成
   - ユースケースギャラリー

3. ✅ **依存関係の最新化**
   - ✅ すべての依存関係を最新バージョンに更新（2024年4月時点）
   - ✅ Rust 2023エコシステムとの互換性確保
   - ✅ セキュリティと性能向上のためのアップデート
   - ✅ rand 0.9.0 APIの変更対応（`gen_range` → `random_range`）
   - ✅ Parquet圧縮定数の新API対応

4. **パッケージング**
   - crates.ioへの公開と配布
   - バージョニング戦略
   - 依存関係の管理

## 主要依存関係（2024年4月時点最新）

```toml
[dependencies]
num-traits = "0.2.19"        # 数値型特性サポート
thiserror = "2.0.12"         # エラーハンドリング
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

---

この計画はPandRSがPythonのpandasライブラリと同等の機能を提供しつつも、Rustの特性を活かした高性能なデータ分析ライブラリを目指すためのロードマップです。実装の優先順位はコミュニティの関心やニーズに応じて調整されるべきです。