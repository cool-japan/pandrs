// 列指向ストレージプロトタイプ
// このファイルは、新しい列指向ストレージシステムのプロトタイプを実装します。
// このプロトタイプは、PERFORMANCE_IMPLEMENTATION_PLAN.mdで説明されている設計に基づいています。

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

// 列の型を表す列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ColumnType {
    Int64,
    Float64,
    String,
    Boolean,
    // 他の型も追加可能
}

// 列の共通トレイト
pub trait ColumnTrait: Debug + Send + Sync {
    fn len(&self) -> usize;
    #[allow(dead_code)]
    fn is_empty(&self) -> bool;
    #[allow(dead_code)]
    fn column_type(&self) -> ColumnType;
    #[allow(dead_code)]
    fn name(&self) -> Option<&str>;
    #[allow(dead_code)]
    fn clone_box(&self) -> Box<dyn ColumnTrait>;
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any;
}

// Int64型の列
#[derive(Debug, Clone)]
pub struct Int64Column {
    data: Arc<[i64]>,
    #[allow(dead_code)]
    null_mask: Option<Arc<[u8]>>,
    name: Option<String>,
}

impl Int64Column {
    pub fn new(data: Vec<i64>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[allow(dead_code)]
    pub fn values(&self) -> &[i64] {
        &self.data
    }

    // 高速な合計計算
    pub fn sum(&self) -> i64 {
        if self.data.is_empty() {
            return 0;
        }
        
        // 最適化された合計計算
        self.data.iter().sum()
    }

    // 高速な平均計算
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        
        let sum: i64 = self.sum();
        Some(sum as f64 / self.len() as f64)
    }
}

impl ColumnTrait for Int64Column {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Int64
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_box(&self) -> Box<dyn ColumnTrait> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Float64型の列
#[derive(Debug, Clone)]
pub struct Float64Column {
    data: Arc<[f64]>,
    #[allow(dead_code)]
    null_mask: Option<Arc<[u8]>>,
    name: Option<String>,
}

impl Float64Column {
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[allow(dead_code)]
    pub fn values(&self) -> &[f64] {
        &self.data
    }

    // 高速な合計計算
    pub fn sum(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        
        // 最適化された合計計算
        self.data.iter().sum()
    }

    // 高速な平均計算
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        
        let sum: f64 = self.sum();
        Some(sum / self.len() as f64)
    }
}

impl ColumnTrait for Float64Column {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Float64
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_box(&self) -> Box<dyn ColumnTrait> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// 文字列プール
#[derive(Debug, Clone)]
pub struct StringPool {
    strings: Vec<String>,
    hash_map: HashMap<String, u32>,
}

impl StringPool {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            hash_map: HashMap::new(),
        }
    }

    pub fn get_or_insert(&mut self, s: impl Into<String>) -> u32 {
        let s = s.into();
        if let Some(&idx) = self.hash_map.get(&s) {
            return idx;
        }
        
        let idx = self.strings.len() as u32;
        self.strings.push(s.clone());
        self.hash_map.insert(s, idx);
        idx
    }

    pub fn get(&self, idx: u32) -> Option<&str> {
        self.strings.get(idx as usize).map(|s| s.as_str())
    }
}

// 文字列型の列
#[derive(Debug, Clone)]
pub struct StringColumn {
    // 文字列プール(共有)
    string_pool: Arc<StringPool>,
    // 文字列プールへのインデックス
    indices: Arc<[u32]>,
    #[allow(dead_code)]
    null_mask: Option<Arc<[u8]>>,
    name: Option<String>,
}

impl StringColumn {
    pub fn new(data: Vec<String>) -> Self {
        let mut pool = StringPool::new();
        let indices: Vec<u32> = data.iter()
            .map(|s| pool.get_or_insert(s))
            .collect();
        
        Self {
            string_pool: Arc::new(pool),
            indices: indices.into(),
            null_mask: None,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn values<'a>(&'a self) -> impl Iterator<Item = &'a str> + 'a {
        self.indices.iter()
            .filter_map(move |&idx| self.string_pool.get(idx))
    }
}

impl ColumnTrait for StringColumn {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::String
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_box(&self) -> Box<dyn ColumnTrait> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// 列の型消去版
#[derive(Debug, Clone)]
pub enum Column {
    Int64(Int64Column),
    Float64(Float64Column),
    String(StringColumn),
    // 他の型も追加可能
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int64(col) => col.len(),
            Column::Float64(col) => col.len(),
            Column::String(col) => col.len(),
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    pub fn column_type(&self) -> ColumnType {
        match self {
            Column::Int64(_) => ColumnType::Int64,
            Column::Float64(_) => ColumnType::Float64,
            Column::String(_) => ColumnType::String,
        }
    }

    #[allow(dead_code)]
    pub fn name(&self) -> Option<&str> {
        match self {
            Column::Int64(col) => col.name(),
            Column::Float64(col) => col.name(),
            Column::String(col) => col.name(),
        }
    }

    // Int64列として取得（型が一致する場合のみ）
    pub fn as_int64(&self) -> Option<&Int64Column> {
        if let Column::Int64(col) = self {
            Some(col)
        } else {
            None
        }
    }

    // Float64列として取得（型が一致する場合のみ）
    pub fn as_float64(&self) -> Option<&Float64Column> {
        if let Column::Float64(col) = self {
            Some(col)
        } else {
            None
        }
    }

    // String列として取得（型が一致する場合のみ）
    #[allow(dead_code)]
    pub fn as_string(&self) -> Option<&StringColumn> {
        if let Column::String(col) = self {
            Some(col)
        } else {
            None
        }
    }
}

// 型変換
impl From<Int64Column> for Column {
    fn from(col: Int64Column) -> Self {
        Column::Int64(col)
    }
}

impl From<Float64Column> for Column {
    fn from(col: Float64Column) -> Self {
        Column::Float64(col)
    }
}

impl From<StringColumn> for Column {
    fn from(col: StringColumn) -> Self {
        Column::String(col)
    }
}

// 最適化されたDataFrame
#[derive(Debug, Clone)]
pub struct OptimizedDataFrame {
    // 列データ
    columns: Vec<Column>,
    // 列名→インデックスのマッピング
    column_indices: HashMap<String, usize>,
    // 列の順序
    column_names: Vec<String>,
}

impl OptimizedDataFrame {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
        }
    }

    // 列の追加
    pub fn add_column<C: Into<Column>>(&mut self, name: impl Into<String>, column: C) -> Result<(), String> {
        let name = name.into();
        let column = column.into();
        
        // 列名の重複チェック
        if self.column_indices.contains_key(&name) {
            return Err(format!("列名 '{}' は既に存在します", name));
        }
        
        // 行数の整合性チェック
        let column_len = column.len();
        if !self.columns.is_empty() && column_len != self.row_count() {
            return Err(format!(
                "列の長さ ({}) がDataFrameの行数 ({}) と一致しません",
                column_len,
                self.row_count()
            ));
        }
        
        // 列の追加
        let column_idx = self.columns.len();
        self.columns.push(column);
        self.column_indices.insert(name.clone(), column_idx);
        self.column_names.push(name);
        
        Ok(())
    }

    // 行数の取得
    pub fn row_count(&self) -> usize {
        if self.columns.is_empty() {
            0
        } else {
            self.columns[0].len()
        }
    }

    // 列数の取得
    #[allow(dead_code)]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    // 列名の取得
    #[allow(dead_code)]
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    // 列の取得
    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.column_indices.get(name).map(|&idx| &self.columns[idx])
    }

    // 列の型安全な取得（Int64）
    pub fn get_int64_column(&self, name: &str) -> Option<&Int64Column> {
        self.get_column(name).and_then(|col| col.as_int64())
    }

    // 列の型安全な取得（Float64）
    pub fn get_float64_column(&self, name: &str) -> Option<&Float64Column> {
        self.get_column(name).and_then(|col| col.as_float64())
    }

    // 列の型安全な取得（String）
    #[allow(dead_code)]
    pub fn get_string_column(&self, name: &str) -> Option<&StringColumn> {
        self.get_column(name).and_then(|col| col.as_string())
    }
}

#[allow(dead_code)]
fn demonstrate_vector_operations() {
    println!("=== ベクトル操作の最適化パフォーマンス比較 ===");
    
    // 大量のデータ
    let n = 10_000_000;
    let int_data: Vec<i64> = (0..n).collect();
    let float_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    
    // 最適化された列操作
    let start = Instant::now();
    let int_col = Int64Column::new(int_data.clone());
    let sum1 = int_col.sum();
    let mean1 = int_col.mean().unwrap();
    let optimized_duration = start.elapsed();
    
    // 単純なベクトル操作（従来のループベース）
    let start = Instant::now();
    let mut sum2 = 0i64;
    let mut count = 0usize;
    for &val in &int_data {
        sum2 += val;
        count += 1;
    }
    let mean2 = sum2 as f64 / count as f64;
    let simple_duration = start.elapsed();
    
    println!("整数ベクトル (長さ = {})", n);
    println!("  最適化列: 合計時間 = {:?}, 合計 = {}, 平均 = {:.2}", optimized_duration, sum1, mean1);
    println!("  単純ループ: 合計時間 = {:?}, 合計 = {}, 平均 = {:.2}", simple_duration, sum2, mean2);
    println!("  高速化: {:.2}倍", simple_duration.as_secs_f64() / optimized_duration.as_secs_f64());
    
    // 浮動小数点数のケース
    let start = Instant::now();
    let float_col = Float64Column::new(float_data.clone());
    let sum1 = float_col.sum();
    let mean1 = float_col.mean().unwrap();
    let optimized_duration = start.elapsed();
    
    // 単純なベクトル操作
    let start = Instant::now();
    let mut sum2 = 0.0f64;
    let mut count = 0usize;
    for &val in &float_data {
        sum2 += val;
        count += 1;
    }
    let mean2 = sum2 / count as f64;
    let simple_duration = start.elapsed();
    
    println!("\n浮動小数点ベクトル (長さ = {})", n);
    println!("  最適化列: 合計時間 = {:?}, 合計 = {:.2}, 平均 = {:.2}", optimized_duration, sum1, mean1);
    println!("  単純ループ: 合計時間 = {:?}, 合計 = {:.2}, 平均 = {:.2}", simple_duration, sum2, mean2);
    println!("  高速化: {:.2}倍", simple_duration.as_secs_f64() / optimized_duration.as_secs_f64());
}

#[allow(dead_code)]
fn demonstrate_string_pooling() {
    println!("\n=== 文字列プールのメモリ効率比較 ===");
    
    // 重複の多い文字列データを生成
    let n = 1_000_000;
    let unique_strings = ["Red", "Green", "Blue", "Yellow", "Black", "White", "Orange", "Purple", "Brown", "Gray"];
    let string_data: Vec<String> = (0..n)
        .map(|i| unique_strings[i % unique_strings.len()].to_string())
        .collect();
    
    // メモリ使用量を概算（単純な文字列ベクトル）
    let simple_size = string_data.iter().map(|s| s.capacity() + std::mem::size_of::<String>()).sum::<usize>();
    
    // 文字列プールを使用したカラム
    let start = Instant::now();
    let string_col = StringColumn::new(string_data.clone());
    let string_pool_creation = start.elapsed();
    
    // 文字列プールのメモリサイズ概算
    let pool_unique_strings_size = string_col.string_pool.strings.iter().map(|s| s.capacity() + std::mem::size_of::<String>()).sum::<usize>();
    let indices_size = string_col.indices.len() * std::mem::size_of::<u32>();
    let pool_size = pool_unique_strings_size + indices_size;
    
    println!("文字列データ (長さ = {}, ユニーク値 = {})", n, unique_strings.len());
    println!("  単純なベクトル: メモリ使用量 ≈ {:.2} MB", simple_size as f64 / 1024.0 / 1024.0);
    println!("  文字列プール: メモリ使用量 ≈ {:.2} MB", pool_size as f64 / 1024.0 / 1024.0);
    println!("  メモリ削減: {:.2}倍", simple_size as f64 / pool_size as f64);
    println!("  プール作成時間: {:?}", string_pool_creation);
    
    // アクセス速度の比較
    let start = Instant::now();
    let mut _simple_count = 0;
    for s in &string_data {
        if s == "Blue" {
            _simple_count += 1;
        }
    }
    let simple_duration = start.elapsed();
    
    let start = Instant::now();
    let mut _pooled_count = 0;
    for s in string_col.values() {
        if s == "Blue" {
            _pooled_count += 1;
        }
    }
    let pooled_duration = start.elapsed();
    
    println!("  単純なベクトル検索時間: {:?}", simple_duration);
    println!("  プールドベクトル検索時間: {:?}", pooled_duration);
    println!("  検索高速化: {:.2}倍", simple_duration.as_secs_f64() / pooled_duration.as_secs_f64());
}

#[allow(dead_code)]
fn demonstrate_dataframe_operations() {
    println!("\n=== 最適化DataFrameのデモ ===");
    
    // 最適化されたDataFrameを作成
    let mut df = OptimizedDataFrame::new();
    
    // 列を追加
    let n = 1_000_000;
    
    let start = Instant::now();
    let int_col = Int64Column::new((0..n).collect()).with_name("id");
    let float_col = Float64Column::new((0..n).map(|i| i as f64 * 0.5).collect()).with_name("value");
    
    // 文字列列（カテゴリデータ）
    let categories = ["A", "B", "C", "D", "E"];
    let string_data: Vec<String> = (0..n)
        .map(|i| categories[i as usize % categories.len()].to_string())
        .collect();
    
    let string_col = StringColumn::new(string_data).with_name("category");
    
    // DataFrameに列を追加
    df.add_column("id", int_col).unwrap();
    df.add_column("value", float_col).unwrap();
    df.add_column("category", string_col).unwrap();
    
    let creation_time = start.elapsed();
    
    println!("最適化DataFrame ({}行 x {}列)", df.row_count(), df.column_count());
    println!("  列名: {:?}", df.column_names());
    println!("  作成時間: {:?}", creation_time);
    
    // 型安全な列アクセス
    let start = Instant::now();
    let id_col = df.get_int64_column("id").unwrap();
    let id_sum = id_col.sum();
    
    let value_col = df.get_float64_column("value").unwrap();
    let value_mean = value_col.mean().unwrap();
    
    let access_time = start.elapsed();
    
    println!("  id列の合計: {}", id_sum);
    println!("  value列の平均: {:.2}", value_mean);
    println!("  アクセス・計算時間: {:?}", access_time);
}

#[allow(dead_code)]
fn main() {
    demonstrate_vector_operations();
    demonstrate_string_pooling();
    demonstrate_dataframe_operations();
}