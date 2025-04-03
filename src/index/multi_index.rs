use crate::error::{PandRSError, Result};
use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// MultiIndex構造体
///
/// 複数レベルの階層化されたインデックスを表現します。
/// Pythonのpandasの`MultiIndex`と同様の機能を提供します。
#[derive(Debug, Clone)]
pub struct MultiIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// 各レベルのラベル
    levels: Vec<Vec<T>>,
    
    /// 各レベルの値のインデックスを示すコード
    codes: Vec<Vec<i32>>,
    
    /// 各レベルの名前
    names: Vec<Option<String>>,
    
    /// MultiIndexの値から位置へのマッピング
    map: HashMap<Vec<T>, usize>,
}

impl<T> MultiIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// 新しいMultiIndexを作成
    ///
    /// # 引数
    /// * `levels` - 各レベルの一意な値のリスト
    /// * `codes` - 各レベルでのインデックス位置を示すコード（-1は欠損値）
    /// * `names` - 各レベルの名前（オプション）
    pub fn new(
        levels: Vec<Vec<T>>,
        codes: Vec<Vec<i32>>,
        names: Option<Vec<Option<String>>>,
    ) -> Result<Self> {
        // 基本的な検証
        if levels.is_empty() {
            return Err(PandRSError::Index("少なくとも1つのレベルが必要です".into()));
        }
        
        if levels.len() != codes.len() {
            return Err(PandRSError::Index(
                "levelsとcodesの長さが一致しません".into()
            ));
        }
        
        // コードの有効性確認
        for (level_idx, level_codes) in codes.iter().enumerate() {
            let max_code = levels[level_idx].len() as i32 - 1;
            for &code in level_codes.iter() {
                if code > max_code && code != -1 {
                    return Err(PandRSError::Index(format!(
                        "レベル{}のコード{}は有効範囲外です",
                        level_idx, code
                    )));
                }
            }
        }
        
        // 行数の一貫性確認
        let n_rows = if !codes.is_empty() { codes[0].len() } else { 0 };
        for level_codes in &codes {
            if level_codes.len() != n_rows {
                return Err(PandRSError::Index(
                    "すべてのレベルは同じ行数を持つ必要があります".into()
                ));
            }
        }
        
        // 名前の数確認
        let names = match names {
            Some(n) => {
                if n.len() != levels.len() {
                    return Err(PandRSError::Index(
                        "namesはlevelsと同じ長さである必要があります".into()
                    ));
                }
                n
            }
            None => vec![None; levels.len()],
        };
        
        // マップの構築
        let mut map = HashMap::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut row_values = Vec::with_capacity(levels.len());
            
            for level_idx in 0..levels.len() {
                let code = codes[level_idx][i];
                if code == -1 {
                    // 欠損値の場合
                    return Err(PandRSError::NotImplemented(
                        "現在のMultiIndexは欠損値をサポートしていません".into()
                    ));
                } else {
                    row_values.push(levels[level_idx][code as usize].clone());
                }
            }
            
            if map.insert(row_values.clone(), i).is_some() {
                return Err(PandRSError::Index(
                    "MultiIndexの重複値は許可されていません".into()
                ));
            }
        }
        
        Ok(MultiIndex {
            levels,
            codes,
            names,
            map,
        })
    }
    
    /// Pythonのpandas.MultiIndex.from_tuplesのようにタプルリストからMultiIndexを作成
    pub fn from_tuples(tuples: Vec<Vec<T>>, names: Option<Vec<Option<String>>>) -> Result<Self> {
        if tuples.is_empty() {
            return Err(PandRSError::Index("空のタプルリストが渡されました".into()));
        }
        
        let n_levels = tuples[0].len();
        
        // すべてのタプルが同じ長さを持つことを確認
        for (i, tuple) in tuples.iter().enumerate() {
            if tuple.len() != n_levels {
                return Err(PandRSError::Index(format!(
                    "すべてのタプルは同じ長さを持つ必要があります。タプル{}は長さ{}ですが、最初のタプルは長さ{}です。",
                    i, tuple.len(), n_levels
                )));
            }
        }
        
        // 各レベルの一意な値を収集
        let mut unique_values: Vec<Vec<T>> = vec![Vec::new(); n_levels];
        let mut level_maps: Vec<HashMap<T, i32>> = vec![HashMap::new(); n_levels];
        
        // 各レベルのコードを構築
        let mut codes: Vec<Vec<i32>> = vec![vec![-1; tuples.len()]; n_levels];
        
        for (row_idx, tuple) in tuples.iter().enumerate() {
            for (level_idx, value) in tuple.iter().enumerate() {
                let level_map = &mut level_maps[level_idx];
                
                let code = match level_map.get(value) {
                    Some(&code) => code,
                    None => {
                        let new_code = unique_values[level_idx].len() as i32;
                        unique_values[level_idx].push(value.clone());
                        level_map.insert(value.clone(), new_code);
                        new_code
                    }
                };
                
                codes[level_idx][row_idx] = code;
            }
        }
        
        MultiIndex::new(unique_values, codes, names)
    }
    
    /// 特定の位置のタプルを取得
    pub fn get_tuple(&self, pos: usize) -> Option<Vec<T>> {
        if pos >= self.len() {
            return None;
        }
        
        let mut result = Vec::with_capacity(self.levels.len());
        for level_idx in 0..self.levels.len() {
            let code = self.codes[level_idx][pos];
            if code == -1 {
                // 欠損値は現在サポートしていない
                return None;
            }
            result.push(self.levels[level_idx][code as usize].clone());
        }
        
        Some(result)
    }
    
    /// タプルからその位置を取得
    pub fn get_loc(&self, key: &[T]) -> Option<usize> {
        if key.len() != self.levels.len() {
            return None;
        }
        
        // Vec<T>に変換してマップ検索
        let key_vec = key.to_vec();
        self.map.get(&key_vec).copied()
    }
    
    /// インデックスの長さ（行数）を取得
    pub fn len(&self) -> usize {
        if self.codes.is_empty() {
            0
        } else {
            self.codes[0].len()
        }
    }
    
    /// インデックスが空かどうか
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// レベル数を取得
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
    
    /// 各レベルの値を取得
    pub fn levels(&self) -> &[Vec<T>] {
        &self.levels
    }
    
    /// 各レベルのコードを取得
    pub fn codes(&self) -> &[Vec<i32>] {
        &self.codes
    }
    
    /// 各レベルの名前を取得
    pub fn names(&self) -> &[Option<String>] {
        &self.names
    }
    
    /// 特定のレベルに基づいてスライス
    pub fn get_level_values(&self, level: usize) -> Result<Index<T>> {
        if level >= self.levels.len() {
            return Err(PandRSError::Index(format!(
                "レベル{}は範囲外です。有効なレベルは0から{}です",
                level,
                self.levels.len() - 1
            )));
        }
        
        let mut values = Vec::with_capacity(self.len());
        for &code in &self.codes[level] {
            if code == -1 {
                return Err(PandRSError::NotImplemented(
                    "現在のレベル値の抽出は欠損値をサポートしていません".into()
                ));
            }
            values.push(self.levels[level][code as usize].clone());
        }
        
        Index::new(values)
    }
    
    /// レベルを交換して新しいMultiIndexを作成
    pub fn swaplevel(&self, i: usize, j: usize) -> Result<Self> {
        if i >= self.levels.len() || j >= self.levels.len() {
            return Err(PandRSError::Index(format!(
                "レベル指定が範囲外です。有効なレベルは0から{}です",
                self.levels.len() - 1
            )));
        }
        
        let mut new_levels = self.levels.clone();
        let mut new_codes = self.codes.clone();
        let mut new_names = self.names.clone();
        
        // レベルを交換
        new_levels.swap(i, j);
        new_codes.swap(i, j);
        new_names.swap(i, j);
        
        MultiIndex::new(new_levels, new_codes, Some(new_names))
    }
    
    /// 指定されたレベルの名前を設定
    pub fn set_names(&mut self, names: Vec<Option<String>>) -> Result<()> {
        if names.len() != self.levels.len() {
            return Err(PandRSError::Index(format!(
                "名前の数{}はレベル数{}と一致する必要があります",
                names.len(),
                self.levels.len()
            )));
        }
        
        self.names = names;
        Ok(())
    }
}

// 文字列MultiIndexのエイリアス
pub type StringMultiIndex = MultiIndex<String>;