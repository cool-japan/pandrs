use std::collections::HashMap;
use std::sync::Arc;

/// 文字列データを効率的に管理する文字列プール
#[derive(Debug, Clone)]
pub struct StringPool {
    strings: Arc<Vec<Arc<str>>>,
    hash_map: Arc<HashMap<Arc<str>, u32>>,
}

impl StringPool {
    /// 新しい空の文字列プールを作成する
    pub fn new() -> Self {
        Self {
            strings: Arc::new(Vec::new()),
            hash_map: Arc::new(HashMap::new()),
        }
    }
    
    /// 文字列ベクトルから新しい文字列プールを作成する
    pub fn from_strings(strings: Vec<String>) -> Self {
        let mut pool = Self::new_mut();
        
        for s in strings {
            pool.get_or_insert(&s);
        }
        
        pool.freeze()
    }
    
    /// 可変の文字列プールを作成する（内部で使用）
    fn new_mut() -> StringPoolMut {
        StringPoolMut {
            strings: Vec::new(),
            hash_map: HashMap::new(),
        }
    }
    
    /// 文字列の数を返す
    pub fn len(&self) -> usize {
        self.strings.len()
    }
    
    /// 文字列プールが空かどうかを返す
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
    
    /// インデックスを指定して文字列を取得する
    pub fn get(&self, index: u32) -> Option<&str> {
        self.strings.get(index as usize).map(|s| s.as_ref())
    }
    
    /// 文字列を検索し、そのインデックスを返す（なければNone）
    pub fn find(&self, s: &str) -> Option<u32> {
        self.hash_map.get(s).copied()
    }
    
    /// すべての文字列をベクトルとして取得する
    pub fn all_strings(&self) -> Vec<&str> {
        self.strings.iter().map(|s| s.as_ref()).collect()
    }
    
    /// 文字列プールと文字列インデックスのペアから、文字列ベクトルを構築する
    pub fn indices_to_strings(&self, indices: &[u32]) -> Vec<String> {
        indices.iter()
            .map(|&idx| self.get(idx).unwrap_or("").to_string())
            .collect()
    }
    
    /// 2つの文字列プールをマージする
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = Self::new_mut();
        
        // 自分のプールの文字列を追加
        for s in self.all_strings() {
            merged.get_or_insert(s);
        }
        
        // 他のプールの文字列を追加
        for s in other.all_strings() {
            merged.get_or_insert(s);
        }
        
        merged.freeze()
    }
}

/// 変更可能な文字列プール（構築中にのみ使用）
#[derive(Debug)]
struct StringPoolMut {
    strings: Vec<Arc<str>>,
    hash_map: HashMap<Arc<str>, u32>,
}

impl StringPoolMut {
    /// 文字列をプールに追加し、そのインデックスを返す
    fn get_or_insert(&mut self, s: &str) -> u32 {
        // 文字列をArz<str>に変換
        let arc_str: Arc<str> = s.into();
        
        // すでに存在する場合はそのインデックスを返す
        if let Some(&index) = self.hash_map.get(&arc_str) {
            return index;
        }
        
        // 新しいインデックスを割り当て
        let index = self.strings.len() as u32;
        self.strings.push(arc_str.clone());
        self.hash_map.insert(arc_str, index);
        
        index
    }
    
    /// 可変プールを不変プールに変換する
    fn freeze(self) -> StringPool {
        StringPool {
            strings: Arc::new(self.strings),
            hash_map: Arc::new(self.hash_map),
        }
    }
}