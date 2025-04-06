use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::error::{PandRSError, Result};
use crate::index::{Index, StringIndex};
use crate::na::NA;
use crate::series::{Series, NASeries};

/// カテゴリカルデータの順序タイプ
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoricalOrder {
    /// 順序なし
    Unordered,
    /// カテゴリに定義された順序あり
    Ordered,
}

/// カテゴリカルデータを表現する構造体
///
/// メモリ効率のよいカテゴリカルデータ表現を提供します。
/// Pythonのpandasの`Categorical`と同様の機能を持ちます。
#[derive(Debug, Clone)]
pub struct Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    /// 実際のカテゴリ値へのコード（整数インデックス）
    codes: Vec<i32>,
    
    /// カテゴリの一意な値のリスト
    categories: Vec<T>,
    
    /// カテゴリから整数コードへのマップ
    category_map: HashMap<T, i32>,
    
    /// カテゴリの順序
    ordered: CategoricalOrder,
}

impl<T> Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    /// 新しいカテゴリカルデータを作成
    ///
    /// # 引数
    /// * `values` - 元のデータ値
    /// * `categories` - 使用するカテゴリのリスト（指定がない場合は一意な値から自動生成）
    /// * `ordered` - カテゴリの順序付けを行うかどうか
    pub fn new(
        values: Vec<T>,
        categories: Option<Vec<T>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        // カテゴリのセットアップ
        let (categories, category_map) = match categories {
            Some(cats) => {
                // 指定されたカテゴリのマップを作成
                let mut map = HashMap::with_capacity(cats.len());
                for (i, cat) in cats.iter().enumerate() {
                    if map.insert(cat.clone(), i as i32).is_some() {
                        return Err(PandRSError::Consistency(format!(
                            "カテゴリ '{:?}' が重複しています",
                            cat
                        )));
                    }
                }
                (cats, map)
            }
            None => {
                // データから一意なカテゴリを収集
                let mut unique_cats = Vec::new();
                let mut map = HashMap::new();
                
                for value in &values {
                    if !map.contains_key(value) {
                        map.insert(value.clone(), unique_cats.len() as i32);
                        unique_cats.push(value.clone());
                    }
                }
                
                (unique_cats, map)
            }
        };
        
        // 値からコードへの変換
        let mut codes = Vec::with_capacity(values.len());
        for value in values {
            match category_map.get(&value) {
                Some(&code) => codes.push(code),
                None => {
                    return Err(PandRSError::Consistency(format!(
                        "値 '{:?}' はカテゴリに含まれていません",
                        value
                    )));
                }
            }
        }
        
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered: ordered.unwrap_or(CategoricalOrder::Unordered),
        })
    }
    
    /// コードから直接Categoricalを作成（内部使用）
    fn from_codes(
        codes: Vec<i32>,
        categories: Vec<T>,
        ordered: CategoricalOrder,
    ) -> Result<Self> {
        // カテゴリのマップを構築
        let mut category_map = HashMap::with_capacity(categories.len());
        for (i, cat) in categories.iter().enumerate() {
            category_map.insert(cat.clone(), i as i32);
        }
        
        // コードの有効性チェック
        let max_code = categories.len() as i32 - 1;
        for &code in &codes {
            if code != -1 && (code < 0 || code > max_code) {
                return Err(PandRSError::Consistency(format!(
                    "コード {} は有効範囲外です",
                    code
                )));
            }
        }
        
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered,
        })
    }
    
    /// データの長さを取得
    pub fn len(&self) -> usize {
        self.codes.len()
    }
    
    /// データが空かどうか
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
    
    /// カテゴリの一覧を取得
    pub fn categories(&self) -> &[T] {
        &self.categories
    }
    
    /// コードを取得
    pub fn codes(&self) -> &[i32] {
        &self.codes
    }
    
    /// 順序情報を取得
    pub fn ordered(&self) -> &CategoricalOrder {
        &self.ordered
    }
    
    /// 特定のインデックスの値を取得
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.codes.len() {
            return None;
        }
        
        let code = self.codes[index];
        if code == -1 {
            // -1は欠損値を表す
            None
        } else {
            Some(&self.categories[code as usize])
        }
    }
    
    /// 順序を設定
    pub fn set_ordered(&mut self, ordered: CategoricalOrder) {
        self.ordered = ordered;
    }
    
    /// カテゴリの順序を変更
    pub fn reorder_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        if new_categories.len() != self.categories.len() {
            return Err(PandRSError::Consistency(format!(
                "新しいカテゴリの数 {} が現在のカテゴリ数 {} と一致しません",
                new_categories.len(),
                self.categories.len()
            )));
        }
        
        // 新しいカテゴリにすべての値が含まれているか確認
        let mut new_cat_set = HashMap::with_capacity(new_categories.len());
        for cat in &new_categories {
            new_cat_set.insert(cat, ());
        }
        
        for cat in &self.categories {
            if !new_cat_set.contains_key(cat) {
                return Err(PandRSError::Consistency(format!(
                    "カテゴリ '{:?}' が新しいカテゴリリストに含まれていません",
                    cat
                )));
            }
        }
        
        // 新しいカテゴリとマップを構築
        let mut new_map = HashMap::with_capacity(new_categories.len());
        let mut new_codes = vec![-1; self.codes.len()];
        
        for (i, cat) in new_categories.iter().enumerate() {
            new_map.insert(cat.clone(), i as i32);
        }
        
        // コードの再マッピング
        for (i, &old_code) in self.codes.iter().enumerate() {
            if old_code != -1 {
                let old_cat = &self.categories[old_code as usize];
                if let Some(&new_code) = new_map.get(old_cat) {
                    new_codes[i] = new_code;
                }
            }
        }
        
        self.categories = new_categories;
        self.category_map = new_map;
        self.codes = new_codes;
        
        Ok(())
    }
    
    /// カテゴリの追加
    pub fn add_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        let mut categories = self.categories.clone();
        let mut category_map = self.category_map.clone();
        
        // 新しいカテゴリを追加
        for cat in new_categories {
            if category_map.contains_key(&cat) {
                return Err(PandRSError::Consistency(format!(
                    "カテゴリ '{:?}' は既に存在します",
                    cat
                )));
            }
            
            let new_code = categories.len() as i32;
            category_map.insert(cat.clone(), new_code);
            categories.push(cat);
        }
        
        self.categories = categories;
        self.category_map = category_map;
        
        Ok(())
    }
    
    /// カテゴリの削除
    pub fn remove_categories(&mut self, categories_to_remove: &[T]) -> Result<()> {
        // 削除するコードのセット
        let mut codes_to_remove = Vec::new();
        
        for cat in categories_to_remove {
            if let Some(&code) = self.category_map.get(cat) {
                codes_to_remove.push(code);
            } else {
                return Err(PandRSError::Consistency(format!(
                    "カテゴリ '{:?}' は存在しません",
                    cat
                )));
            }
        }
        
        // 削除するカテゴリを使用しているデータを欠損値に変更
        for code in &mut self.codes {
            if codes_to_remove.contains(code) {
                *code = -1;
            }
        }
        
        // 新しいカテゴリリストとマップを構築
        let mut new_categories = Vec::new();
        let mut new_map = HashMap::new();
        
        for (i, cat) in self.categories.iter().enumerate() {
            if !categories_to_remove.contains(cat) {
                let new_code = new_categories.len() as i32;
                new_map.insert(cat.clone(), new_code);
                new_categories.push(cat.clone());
            }
        }
        
        // コードの再マッピング
        for code in &mut self.codes {
            if *code != -1 {
                let old_cat = &self.categories[*code as usize];
                *code = *new_map.get(old_cat).unwrap();
            }
        }
        
        self.categories = new_categories;
        self.category_map = new_map;
        
        Ok(())
    }
    
    /// 値をカテゴリに変換
    pub fn as_values(&self) -> Vec<Option<T>> {
        self.codes
            .iter()
            .map(|&code| {
                if code == -1 {
                    None
                } else {
                    Some(self.categories[code as usize].clone())
                }
            })
            .collect()
    }
    
    /// 一意なカテゴリの出現回数をカウント
    pub fn value_counts(&self) -> Result<Series<usize>> {
        let mut counts = vec![0; self.categories.len()];
        
        for &code in &self.codes {
            if code != -1 {
                counts[code as usize] += 1;
            }
        }
        
        // カウント結果からSeriesを構築
        let mut values = Vec::with_capacity(self.categories.len());
        let mut count_values = Vec::with_capacity(self.categories.len());
        
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                values.push(self.categories[i].clone());
                count_values.push(count);
            }
        }
        
        // インデックスとカウント値からSeriesを構築
        let index = Index::new(values)?;
        let result = Series::with_index(count_values, index, Some("count".to_string()))?;
        
        Ok(result)
    }
    
    /// Seriesに変換
    pub fn to_series(&self, name: Option<String>) -> Result<Series<T>> {
        // コードを値に変換
        let values: Vec<T> = self
            .codes
            .iter()
            .filter_map(|&code| {
                if code == -1 {
                    None
                } else {
                    Some(self.categories[code as usize].clone())
                }
            })
            .collect();
        
        Series::new(values, name)
    }
}

/// 文字列カテゴリカルのエイリアス
pub type StringCategorical = Categorical<String>;

impl<T> Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    /// NA値（欠損値）を含むベクトルからカテゴリカルデータを作成
    ///
    /// # 引数
    /// * `values` - NA値を含む可能性のあるデータ値
    /// * `categories` - 使用するカテゴリのリスト（指定がない場合は一意な値から自動生成）
    /// * `ordered` - カテゴリの順序付けを行うかどうか
    pub fn from_na_vec(
        values: Vec<NA<T>>,
        categories: Option<Vec<T>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        // カテゴリのセットアップ
        let (categories, category_map) = match categories {
            Some(cats) => {
                // 指定されたカテゴリのマップを作成
                let mut map = HashMap::with_capacity(cats.len());
                for (i, cat) in cats.iter().enumerate() {
                    if map.insert(cat.clone(), i as i32).is_some() {
                        return Err(PandRSError::Consistency(format!(
                            "カテゴリ '{:?}' が重複しています",
                            cat
                        )));
                    }
                }
                (cats, map)
            }
            None => {
                // データから一意なカテゴリを収集
                let mut unique_cats = Vec::new();
                let mut map = HashMap::new();
                
                for value in &values {
                    if let NA::Value(val) = value {
                        if !map.contains_key(val) {
                            map.insert(val.clone(), unique_cats.len() as i32);
                            unique_cats.push(val.clone());
                        }
                    }
                }
                
                (unique_cats, map)
            }
        };
        
        // 値からコードへの変換
        let mut codes = Vec::with_capacity(values.len());
        for value in values {
            match value {
                NA::Value(val) => {
                    match category_map.get(&val) {
                        Some(&code) => codes.push(code),
                        None => {
                            return Err(PandRSError::Consistency(format!(
                                "値 '{:?}' はカテゴリに含まれていません",
                                val
                            )));
                        }
                    }
                }
                NA::NA => codes.push(-1), // NA値は-1コードとして表現
            }
        }
        
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered: ordered.unwrap_or(CategoricalOrder::Unordered),
        })
    }
    
    /// カテゴリカルデータを`NA<T>`ベクトルに変換
    pub fn to_na_vec(&self) -> Vec<NA<T>> {
        self.codes
            .iter()
            .map(|&code| {
                if code == -1 {
                    NA::NA
                } else {
                    NA::Value(self.categories[code as usize].clone())
                }
            })
            .collect()
    }
    
    /// カテゴリカルデータをNASeriesに変換
    pub fn to_na_series(&self, name: Option<String>) -> Result<NASeries<T>> {
        let na_values = self.to_na_vec();
        NASeries::new(na_values, name)
    }
    
    /// 2つのカテゴリカルデータの和集合（すべての一意なカテゴリ）を作成
    pub fn union(&self, other: &Self) -> Result<Self> {
        // 自分のカテゴリをコピー
        let mut categories = self.categories.clone();
        let mut category_map = self.category_map.clone();
        
        // 相手のカテゴリで自分にないものを追加
        for cat in &other.categories {
            if !category_map.contains_key(cat) {
                let new_code = categories.len() as i32;
                category_map.insert(cat.clone(), new_code);
                categories.push(cat.clone());
            }
        }
        
        // 新しいコードリストを作成（元のデータを保持）
        let codes = self.codes.clone();
        
        // 新しいカテゴリカルを返す
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered: self.ordered.clone(),
        })
    }
    
    /// 2つのカテゴリカルデータの積集合（共通するカテゴリのみ）を作成
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        // 共通するカテゴリを収集
        let mut common_categories = Vec::new();
        let mut category_map = HashMap::new();
        
        for cat in &self.categories {
            if other.category_map.contains_key(cat) {
                let new_code = common_categories.len() as i32;
                category_map.insert(cat.clone(), new_code);
                common_categories.push(cat.clone());
            }
        }
        
        // 新しいコードリストを作成（元のデータを保持するが、共通カテゴリにないものはNA）
        let mut codes = Vec::with_capacity(self.codes.len());
        for &old_code in &self.codes {
            if old_code == -1 {
                codes.push(-1); // 元からNAならNA
            } else {
                let old_cat = &self.categories[old_code as usize];
                match category_map.get(old_cat) {
                    Some(&new_code) => codes.push(new_code),
                    None => codes.push(-1), // 共通カテゴリにない場合はNA
                }
            }
        }
        
        // 新しいカテゴリカルを返す
        Ok(Categorical {
            codes,
            categories: common_categories,
            category_map,
            ordered: self.ordered.clone(),
        })
    }
    
    /// 2つのカテゴリカルデータの差集合（自分のカテゴリから相手のカテゴリを引いたもの）を作成
    pub fn difference(&self, other: &Self) -> Result<Self> {
        // 差分カテゴリを収集
        let mut diff_categories = Vec::new();
        let mut category_map = HashMap::new();
        
        for cat in &self.categories {
            if !other.category_map.contains_key(cat) {
                let new_code = diff_categories.len() as i32;
                category_map.insert(cat.clone(), new_code);
                diff_categories.push(cat.clone());
            }
        }
        
        // 新しいコードリストを作成（差分カテゴリにないものはNA）
        let mut codes = Vec::with_capacity(self.codes.len());
        for &old_code in &self.codes {
            if old_code == -1 {
                codes.push(-1); // 元からNAならNA
            } else {
                let old_cat = &self.categories[old_code as usize];
                match category_map.get(old_cat) {
                    Some(&new_code) => codes.push(new_code),
                    None => codes.push(-1), // 差分カテゴリにない場合はNA
                }
            }
        }
        
        // 新しいカテゴリカルを返す
        Ok(Categorical {
            codes,
            categories: diff_categories,
            category_map,
            ordered: self.ordered.clone(),
        })
    }
}

impl<T> PartialEq for Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    fn eq(&self, other: &Self) -> bool {
        // コードの長さと内容が同じ
        if self.codes.len() != other.codes.len() {
            return false;
        }
        
        for (a, b) in self.codes.iter().zip(other.codes.iter()) {
            if a != b {
                return false;
            }
        }
        
        // カテゴリの長さと内容が同じ
        if self.categories.len() != other.categories.len() {
            return false;
        }
        
        for (a, b) in self.categories.iter().zip(other.categories.iter()) {
            if a != b {
                return false;
            }
        }
        
        // 順序情報も同じ
        self.ordered == other.ordered
    }
}

impl<T> PartialOrd for Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // 順序付きでない場合は比較不可
        if self.ordered != CategoricalOrder::Ordered || other.ordered != CategoricalOrder::Ordered {
            return None;
        }
        
        // カテゴリが異なる場合は比較不可
        if self.categories != other.categories {
            return None;
        }
        
        // コードを比較
        let len = self.codes.len().min(other.codes.len());
        for i in 0..len {
            let a = self.codes[i];
            let b = other.codes[i];
            
            // -1は欠損値なので最小と見なす
            match (a, b) {
                (-1, -1) => continue,
                (-1, _) => return Some(Ordering::Less),
                (_, -1) => return Some(Ordering::Greater),
                (_, _) => {
                    if a < b {
                        return Some(Ordering::Less);
                    } else if a > b {
                        return Some(Ordering::Greater);
                    }
                }
            }
        }
        
        // ここまで同じなら長さで比較
        self.codes.len().partial_cmp(&other.codes.len())
    }
}