use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once, RwLock};
use lazy_static::lazy_static;

// グローバル文字列プールのシングルトンインスタンス
lazy_static! {
    pub static ref GLOBAL_STRING_POOL: GlobalStringPool = GlobalStringPool::new();
}

/// グローバル文字列プール（シングルトン）
#[derive(Debug)]
pub struct GlobalStringPool {
    pool: RwLock<StringPoolMut>,
}

impl GlobalStringPool {
    /// 新しいグローバル文字列プールを作成
    pub fn new() -> Self {
        Self {
            pool: RwLock::new(StringPoolMut {
                strings: Vec::new(),
                hash_map: HashMap::new(),
            }),
        }
    }
    
    /// 文字列をプールに追加し、そのインデックスを返す
    pub fn get_or_insert(&self, s: &str) -> u32 {
        // 読み取りロックで試行
        if let Ok(read_pool) = self.pool.read() {
            if let Some(&idx) = read_pool.hash_map.get(s) {
                return idx;
            }
        }
        
        // 見つからなかった場合は書き込みロックで追加
        if let Ok(mut write_pool) = self.pool.write() {
            // 再確認（他のスレッドが追加した可能性）
            if let Some(&idx) = write_pool.hash_map.get(s) {
                return idx;
            }
            
            // 新しいインデックスを割り当て
            let idx = write_pool.strings.len() as u32;
            let arc_str: Arc<str> = Arc::from(s.to_owned());
            write_pool.strings.push(arc_str.clone());
            write_pool.hash_map.insert(arc_str, idx);
            idx
        } else {
            // ロック失敗時のフォールバック（実際にはエラー処理すべき）
            0
        }
    }
    
    /// インデックスを指定して文字列を取得
    pub fn get(&self, index: u32) -> Option<String> {
        if let Ok(pool) = self.pool.read() {
            pool.strings.get(index as usize).map(|s| s.to_string())
        } else {
            None
        }
    }
    
    /// 登録済み文字列の数を返す
    pub fn len(&self) -> usize {
        if let Ok(pool) = self.pool.read() {
            pool.strings.len()
        } else {
            0
        }
    }
    
    /// グローバルプールに文字列ベクトルを追加し、インデックスのベクトルを返す
    pub fn add_strings(&self, strings: &[String]) -> Vec<u32> {
        strings.iter()
            .map(|s| self.get_or_insert(s))
            .collect()
    }
}

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
    
    /// 文字列ベクトルから新しい文字列プールを作成する（グローバルプールを使う最適化版）
    pub fn from_strings(strings: Vec<String>) -> Self {
        // グローバルプールを使う最適化を追加
        let indices = GLOBAL_STRING_POOL.add_strings(&strings);
        
        // ここで実際のプールを作るが、グローバルプールの参照を活用できる
        let mut pool = Self::new_mut();
        
        for (idx, s) in indices.iter().zip(strings.iter()) {
            let arc_str: Arc<str> = Arc::from(s.to_owned());
            pool.hash_map.insert(arc_str.clone(), *idx);
            if idx >= &(pool.strings.len() as u32) {
                // 必要に応じてサイズを拡張（実際には最適化可能）
                while pool.strings.len() <= *idx as usize {
                    pool.strings.push(arc_str.clone());
                }
            } else {
                pool.strings[*idx as usize] = arc_str;
            }
        }
        
        pool.freeze()
    }
    
    /// 文字列ベクトルから新しい文字列プールを作成する（従来の実装）
    pub fn from_strings_legacy(strings: Vec<String>) -> Self {
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