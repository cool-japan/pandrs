use thiserror::Error;

/// エラー型の定義
#[derive(Error, Debug)]
pub enum PandRSError {
    #[error("入出力エラー: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV処理エラー: {0}")]
    Csv(#[from] csv::Error),

    #[error("JSON処理エラー: {0}")]
    Json(#[from] serde_json::Error),

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

    #[error("未実装機能: {0}")]
    NotImplemented(String),

    #[error("可視化エラー: {0}")]
    Visualization(String),

    #[error("並列処理エラー: {0}")]
    Parallel(String),

    #[error("空データエラー: {0}")]
    Empty(String),

    #[error("その他のエラー: {0}")]
    Other(String),
}

/// Resultの型エイリアス
pub type Result<T> = std::result::Result<T, PandRSError>;
