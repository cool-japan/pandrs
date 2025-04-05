//! 機械学習パイプラインモジュール
//!
//! scikit-learn相当のデータ変換パイプラインを提供します。

use crate::dataframe::DataFrame;
use crate::error::Result;

/// データ変換器のトレイト
pub trait Transformer {
    /// データを変換する
    fn transform(&self, df: &DataFrame) -> Result<DataFrame>;
    
    /// データを学習し、その後変換する
    fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame>;
    
    /// データから学習する
    fn fit(&mut self, df: &DataFrame) -> Result<()>;
}

/// データ変換ステップを連鎖させるパイプライン
pub struct Pipeline {
    transformers: Vec<Box<dyn Transformer>>,
}

impl Pipeline {
    /// 新しいパイプラインを作成
    pub fn new() -> Self {
        Pipeline {
            transformers: Vec::new(),
        }
    }
    
    /// 変換器をパイプラインに追加
    pub fn add_transformer<T: Transformer + 'static>(&mut self, transformer: T) -> &mut Self {
        self.transformers.push(Box::new(transformer));
        self
    }
    
    /// パイプラインの全ステップを実行して変換
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for transformer in &self.transformers {
            result = transformer.transform(&result)?;
        }
        
        Ok(result)
    }
    
    /// パイプラインを学習してから変換
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for transformer in &mut self.transformers {
            result = transformer.fit_transform(&result)?;
        }
        
        Ok(result)
    }
    
    /// パイプラインを学習
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let mut temp_df = df.clone();
        
        for transformer in &mut self.transformers {
            transformer.fit(&temp_df)?;
            temp_df = transformer.transform(&temp_df)?;
        }
        
        Ok(())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}