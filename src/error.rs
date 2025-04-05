use thiserror::Error;

/// エラー型の定義
#[derive(Error, Debug)]
pub enum Error {
    #[error("入出力エラー: {0}")]
    IoError(String),
    
    #[error("CSVエラー: {0}")]
    CsvError(String),
    
    #[error("JSONエラー: {0}")]
    JsonError(String),
    
    #[error("Parquetエラー: {0}")]
    ParquetError(String),
    
    #[error("インデックスが範囲外です: インデックス {index}, サイズ {size}")]
    IndexOutOfBounds { index: usize, size: usize },
    
    #[error("列が見つかりません: {0}")]
    ColumnNotFound(String),
    
    #[error("列名が重複しています: {0}")]
    DuplicateColumnName(String),
    
    #[error("行数が一致しません: 期待値 {expected}, 実際 {found}")]
    InconsistentRowCount { expected: usize, found: usize },
    
    #[error("列の型が一致しません: 列 {name}, 期待値 {expected:?}, 実際 {found:?}")]
    ColumnTypeMismatch {
        name: String,
        expected: crate::column::ColumnType,
        found: crate::column::ColumnType,
    },
    
    #[error("無効な正規表現です: {0}")]
    InvalidRegex(String),
    
    #[error("未実装機能です: {0}")]
    NotImplemented(String),
    
    #[error("データがありません: {0}")]
    EmptyData(String),
    
    #[error("操作に失敗しました: {0}")]
    OperationFailed(String),
    
    #[error("無効な入力です: {0}")]
    InvalidInput(String),
    
    #[error("長さが一致しません: 期待値 {expected}, 実際 {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    
    // 旧エラー型との互換性のために追加
    #[error("入出力エラー")]
    Io(#[source] std::io::Error),
    
    #[error("CSVエラー")]
    Csv(#[source] csv::Error),
    
    #[error("JSONエラー")]
    Json(#[source] serde_json::Error),
    
    #[error("インデックスエラー: {0}")]
    Index(String),
    
    #[error("列エラー: {0}")]
    Column(String),
    
    #[error("型変換エラー: {0}")]
    Cast(String),
    
    #[error("データ形式エラー: {0}")]
    Format(String),
    
    #[error("データ一貫性エラー: {0}")]
    Consistency(String),
    
    #[error("可視化エラー: {0}")]
    Visualization(String),
    
    #[error("並列処理エラー: {0}")]
    Parallel(String),
    
    #[error("空データエラー: {0}")]
    Empty(String),
    
    #[error("キーが見つかりません: {0}")]
    KeyNotFound(String),
    
    #[error("操作エラー: {0}")]
    Operation(String),
    
    #[error("次元不一致エラー: {0}")]
    DimensionMismatch(String),
    
    #[error("データ不足エラー: {0}")]
    InsufficientData(String),
    
    #[error("計算エラー: {0}")]
    ComputationError(String),
    
    #[error("無効な操作です: {0}")]
    InvalidOperation(String),
    
    #[error("無効な値です: {0}")]
    InvalidValue(String),
    
    #[error("その他のエラー: {0}")]
    Other(String),
}

// PandRSErrorとの後方互換性を維持する
pub type PandRSError = Error;

/// Resultの型エイリアス
pub type Result<T> = std::result::Result<T, Error>;

// 標準エラーからの変換（From実装は#[derive(Error)]で自動的に実装される）
impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Self {
        Error::Csv(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err)
    }
}

impl From<regex::Error> for Error {
    fn from(err: regex::Error) -> Self {
        Error::InvalidRegex(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

// Plottersエラーの変換
impl<E: std::error::Error + Send + Sync + 'static> From<plotters::drawing::DrawingAreaErrorKind<E>> for Error {
    fn from(err: plotters::drawing::DrawingAreaErrorKind<E>) -> Self {
        Error::Visualization(format!("プロット描画エラー: {}", err))
    }
}