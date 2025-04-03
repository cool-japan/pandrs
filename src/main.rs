use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;
use std::iter::Sum;
use std::fmt::Debug;
use std::cmp::PartialOrd;

// num-traits クレートの cast 関数をインポート
// Cargo.toml に `num-traits = "0.2"` を追加する必要があります
use num_traits::cast;

// データフレーム構造体
#[derive(Debug, Clone)]
pub struct DataFrame<T: Debug + Clone> {
    data: HashMap<String, Vec<T>>,
    columns: Vec<String>, // 列の順序を保持
    row_count: usize,
}

// DataFrameの基本メソッド実装
impl<T: Debug + Clone> DataFrame<T> {
    pub fn new() -> Self {
        DataFrame {
            data: HashMap::new(),
            columns: Vec::new(),
            row_count: 0,
        }
    }

    // 指定された列名とデータでDataFrameを生成する
    pub fn from_vec(data: HashMap<String, Vec<T>>) -> Result<Self, Box<dyn Error>> {
        if data.is_empty() {
            return Ok(DataFrame::new());
        }
        // HashMapのキーから列名を取得 (順序は保証されない)
        // `mut` を削除 (warning: variable does not need to be mutable)
        let columns: Vec<String> = data.keys().cloned().collect();
        // 必要に応じて列の順序を安定させる (例: ソート)
        // let mut columns = data.keys().cloned().collect::<Vec<_>>(); // mut が必要になる
        // columns.sort();

        let first_column_name = columns.first().ok_or("Data map is empty after collecting keys")?;
        let row_count = data[first_column_name].len();

        // 全ての列の長さが一致するか確認
        for column_name in &columns {
             let column_data = data.get(column_name).ok_or("Internal error: Column name mismatch")?;
             if column_data.len() != row_count {
                 return Err(format!(
                    "Column '{}' has length {} but expected {}",
                    column_name,
                    column_data.len(),
                    row_count
                 ).into());
             }
        }
        // DataFrameを構築 (取得した列リストを使用)
        Ok(DataFrame { data, columns, row_count })
    }

    // DataFrameに行を追加する
    pub fn add_row(&mut self, mut row: HashMap<String, T>) -> Result<(), Box<dyn Error>> {
        // DataFrameが空の場合、最初の行としてデータを挿入し、列の順序も設定
        if self.columns.is_empty() {
             // HashMapのキーの順序は不定なため、ここで順序を決める必要がある
             self.columns = row.keys().cloned().collect();
             // 必要ならソート:
             // self.columns.sort();
             for column_name in &self.columns {
                let value = row.remove(column_name)
                    .ok_or_else(|| format!("Internal error: value for column '{}' disappeared", column_name))?;
                 self.data.insert(column_name.clone(), vec![value]);
             }
             self.row_count = 1;
             return Ok(());
        }

        // 列数がDataFrameと一致するか確認
        if row.len() != self.columns.len() {
            return Err(format!(
                "Row has {} columns, but DataFrame expects {}",
                row.len(),
                self.columns.len()
            ).into());
        }

        // DataFrameの列順序に従って値を追加
        for column_name in &self.columns {
             match row.remove(column_name) { // HashMapから値を取り出して消費
                 Some(value) => {
                     // data HashMapに対応するVecが存在することは保証されているはず
                     self.data.get_mut(column_name).unwrap().push(value);
                 }
                 None => {
                     // row に DataFrame の列が含まれていない場合
                     return Err(format!("Missing value for column '{}' in the provided row", column_name).into());
                 }
             }
        }
        self.row_count += 1;
        Ok(())
    }

    // 指定された列のデータを返す
    pub fn get_column(&self, column_name: &str) -> Option<&Vec<T>> {
        self.data.get(column_name)
    }

    // 指定された列のデータを変更可能な参照で返す
    pub fn get_column_mut(&mut self, column_name: &str) -> Option<&mut Vec<T>> {
        self.data.get_mut(column_name)
    }

    // 指定された列が存在するか確認する
    pub fn contains_column(&self, column_name: &str) -> bool {
        self.data.contains_key(column_name)
    }

    // DataFrameの列名を返す (定義された順序で)
    pub fn get_columns(&self) -> &Vec<String> {
        &self.columns
    }

    // DataFrameの行数を返す
    pub fn get_row_count(&self) -> usize {
        self.row_count
    }

    // DataFrameをフィルタリングする
    pub fn filter<F>(&self, predicate: F) -> Result<Self, Box<dyn Error>>
    where
        F: Fn(&HashMap<&String, &T>) -> bool,
    {
        let mut filtered_data: HashMap<String, Vec<T>> = HashMap::new();
        for column_name in &self.columns {
            filtered_data.insert(column_name.clone(), Vec::new());
        }
        let mut new_row_count = 0;

        for i in 0..self.row_count {
            // 行データをHashMapとして作成 (参照を使用)
            let mut row_map: HashMap<&String, &T> = HashMap::with_capacity(self.columns.len());
            for column_name in &self.columns {
                row_map.insert(column_name, &self.data[column_name][i]);
            }
            // 条件を満たすかチェック
            if predicate(&row_map) {
                // 条件を満たす行のデータを新しいDataFrameに追加
                for column_name in &self.columns {
                    filtered_data
                        .get_mut(column_name)
                        .unwrap() // キーは存在するはず
                        .push(self.data[column_name][i].clone()); // cloneが必要
                }
                new_row_count += 1;
            }
        }
        // 新しいDataFrameを構築して返す
        Ok(DataFrame {
             data: filtered_data,
             columns: self.columns.clone(), // 列の順序を維持
             row_count: new_row_count,
        })
    }

    // DataFrameをソートする (元のDataFrameは変更せず、新しいDataFrameを返す)
    pub fn sort_by<F>(&self, compare: F) -> Self
    where
        F: Fn(&HashMap<&String, &T>, &HashMap<&String, &T>) -> std::cmp::Ordering,
    {
        if self.row_count == 0 { return self.clone(); } // 空ならクローンを返す

        // ソート後のデータを格納するHashMapを準備
        let mut sorted_data: HashMap<String, Vec<T>> = HashMap::new();
        for column_name in &self.columns {
            sorted_data.insert(column_name.clone(), Vec::with_capacity(self.row_count));
        }

        // 行のインデックス (0..row_count) を作成
        let mut row_indices: Vec<usize> = (0..self.row_count).collect();

        // インデックスをソートするためのヘルパー関数 (元のデータへの参照を使う)
        let get_row_map = |index: usize| -> HashMap<&String, &T> {
            let mut row_map = HashMap::with_capacity(self.columns.len());
            for column_name in &self.columns {
                row_map.insert(column_name, &self.data[column_name][index]);
            }
            row_map
        };

        // インデックス自体を比較関数を使ってソート
        row_indices.sort_unstable_by(|&a, &b| { // パフォーマンスのためunstableで良い場合が多い
             let row_map_a = get_row_map(a);
             let row_map_b = get_row_map(b);
             compare(&row_map_a, &row_map_b)
        });

        // ソートされたインデックス順にデータを新しいHashMapに移す
        for index in row_indices {
            for column_name in &self.columns {
                sorted_data
                    .get_mut(column_name)
                    .unwrap()
                    .push(self.data[column_name][index].clone()); // cloneが必要
            }
        }
        // 新しいDataFrameを構築
        DataFrame { data: sorted_data, columns: self.columns.clone(), row_count: self.row_count }
    }

    // 2つのDataFrameを結合する (Inner Join)
    pub fn join(&self, other: &Self, join_column: &str) -> Result<Self, Box<dyn Error>>
    where
        T: Eq + std::hash::Hash, // HashMapのキーとして使うため Hash も必要
    {
        // 結合列が存在するか確認
        if !self.contains_column(join_column) { return Err(format!("Join column '{}' not found in left DataFrame", join_column).into()); }
        if !other.contains_column(join_column) { return Err(format!("Join column '{}' not found in right DataFrame", join_column).into()); }

        // 結合後のデータを格納するHashMapと列リストを準備
        let mut joined_data: HashMap<String, Vec<T>> = HashMap::new();
        let mut joined_columns = self.columns.clone(); // 左側の列順序を維持
        let mut right_col_rename_map = HashMap::new(); // 右側の重複列名のリネーム用

        // 結合後の列リストを作成 (右側の列を追加、重複名はリネーム)
        for column_name in &other.columns {
            if column_name != join_column {
                let new_col_name = if self.contains_column(column_name) {
                    // 重複する場合はリネーム (例: "col" -> "col_right")
                    let mut rename = format!("{}_right", column_name);
                    let mut count = 1;
                    // リネーム後も重複する場合に備えて連番を追加 (稀なケース)
                    while self.contains_column(&rename) || right_col_rename_map.contains_key(&rename) {
                         rename = format!("{}_right{}", column_name, count);
                         count += 1;
                    }
                    rename
                } else {
                    column_name.clone() // 重複しない場合はそのまま
                };
                // 元の名前 -> 新しい名前 のマッピングを保存
                right_col_rename_map.insert(column_name.clone(), new_col_name.clone());
                joined_columns.push(new_col_name); // 結合後の列リストに追加
            }
        }
        // 結合後のデータ用HashMapを初期化
        for column_name in &joined_columns {
            joined_data.insert(column_name.clone(), Vec::new());
        }

        // 結合列のデータを取得 (unwrapは存在確認済みのため安全)
        let left_join_col_data = self.get_column(join_column).unwrap();
        let right_join_col_data = other.get_column(join_column).unwrap();

        // パフォーマンス向上のため、右側のDataFrameの結合キーで行インデックスのマップを作成
        let mut right_indices_map: HashMap<&T, Vec<usize>> = HashMap::new();
        for (j, right_val) in right_join_col_data.iter().enumerate() {
            right_indices_map.entry(right_val).or_default().push(j);
        }

        // 結合処理
        let mut new_row_count = 0;
        // 左側のDataFrameを行ごとに処理
        for i in 0..self.row_count {
            let left_value = &left_join_col_data[i];
            // 右側のインデックスマップを使って、結合キーが一致する行を検索
            if let Some(matching_indices) = right_indices_map.get(left_value) {
                // 一致する右側の行が見つかった場合、その各行に対して結合結果を生成
                for &j in matching_indices {
                    new_row_count += 1;
                    // 左側のデータを結合後データに追加
                    for column_name in &self.columns {
                        joined_data.get_mut(column_name).unwrap().push(self.data[column_name][i].clone());
                    }
                    // 右側のデータ（結合列を除く）を結合後データに追加
                    for original_right_col in &other.columns {
                        if original_right_col != join_column {
                             // リネームされた列名を使って追加
                             let final_col_name = right_col_rename_map.get(original_right_col).unwrap();
                             joined_data.get_mut(final_col_name).unwrap().push(other.data[original_right_col][j].clone());
                        }
                    }
                }
            }
        }
        // 結合後のDataFrameを構築
        Ok(DataFrame { data: joined_data, columns: joined_columns, row_count: new_row_count })
    }
}


// 数値型に限定したDataFrameの拡張
impl<T> DataFrame<T>
where
    T: FromStr + Clone + Debug + Sum<T> + Default + Copy + std::ops::Div<Output = T> + PartialOrd + 'static,
    T: num_traits::NumCast, // NumCast トレイト境界 (meanで使用)
{
    // 指定された列の合計を計算する
    pub fn sum(&self, column_name: &str) -> Result<T, Box<dyn Error>> {
        let column = self.get_column(column_name)
            .ok_or_else(|| format!("Column '{}' not found for sum()", column_name))?;
        if column.is_empty() {
            Ok(T::default()) // 空ならデフォルト値 (通常 0)
        } else {
            Ok(column.iter().copied().sum()) // イテレータのsumを使用 (T: Sum<T> + Copy)
        }
    }

    // 指定された列の平均を計算する
    pub fn mean(&self, column_name: &str) -> Result<T, Box<dyn Error>> {
        let column = self.get_column(column_name)
             .ok_or_else(|| format!("Column '{}' not found for mean()", column_name))?;
        let n = column.len(); // 行数 (usize)
        if n == 0 {
            return Err(format!("Cannot calculate mean of an empty column '{}'", column_name).into());
        }
        // 合計を計算
        let sum_val = self.sum(column_name)?;

        // 行数(usize)を型Tに安全にキャスト (T: num_traits::NumCast)
        let n_t = cast::<usize, T>(n)
            .ok_or_else(|| format!(
                "Failed to cast column length {} (usize) to the required numeric type '{}'",
                n,
                std::any::type_name::<T>() // T: 'static
            ))?;

        // オプション: キャスト後の値がゼロかチェック (T: num_traits::Zero が必要)
        // use num_traits::Zero;
        // if n_t.is_zero() && n != 0 {
        //     return Err("Division by zero occurred after casting column length to type T".into());
        // }

        // 合計 / 行数 (T: Div<Output = T>)
        Ok(sum_val / n_t)
    }
}

// 文字列型 (String) に限定したDataFrameの拡張
impl DataFrame<String> { // String は Debug + Clone + Eq + Hash + FromStr + 'static を満たす
    // 指定された列の文字列を連結する
    pub fn concat(&self, column_name: &str, separator: &str) -> Result<String, Box<dyn Error>> {
        let column = self.get_column(column_name)
            .ok_or_else(|| format!("Column '{}' not found for concat()", column_name))?;
        Ok(column.join(separator)) // Vec<String> の join メソッドを利用
    }
}

// ファイルからDataFrameを読み込む関数
pub fn read_csv<T: Debug + Clone + FromStr + 'static>(
    file_path: &Path,
    has_header: bool,
) -> Result<DataFrame<T>, Box<dyn Error>> {
    // ファイルを開く
    let file = File::open(file_path)
        .map_err(|e| format!("Failed to open file '{}': {}", file_path.display(), e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines(); // 行のイテレータ

    // データ格納用
    let mut data: HashMap<String, Vec<T>> = HashMap::new();
    let mut columns: Vec<String> = Vec::new(); // 列名の順序を保持
    let mut row_count = 0;
    let mut expected_columns = 0; // 期待される列数

    // ヘッダー行の処理
    if has_header {
        if let Some(header_line_result) = lines.next() {
            let header_line = header_line_result.map_err(|e| format!("Failed to read header line from '{}': {}", file_path.display(), e))?;
            // ヘッダーをカンマで分割し、トリムして列名リストを作成
            columns = header_line.split(',').map(|s| s.trim().to_string()).collect();
            // ヘッダーが空または無効な場合はエラー
            if columns.is_empty() || columns.iter().all(String::is_empty) { return Err(format!("CSV file '{}' has an empty or invalid header", file_path.display()).into()); }
            expected_columns = columns.len();
            // データ格納用のVecを列ごとに初期化
            for column in &columns { data.insert(column.clone(), Vec::new()); }
        } else {
            // ヘッダーがあると指定されたがファイルが空の場合
            return Err(format!("CSV file '{}' is empty or failed to read header", file_path.display()).into());
        }
    }

    // データ行の処理
    for (line_idx, line_result) in lines.enumerate() {
        let line_number = line_idx + if has_header { 2 } else { 1 }; // エラー表示用の行番号 (1-based)
        let line = line_result.map_err(|e| format!("Failed to read line {} from '{}': {}", line_number, file_path.display(), e))?;

        // 空行はスキップ
        if line.trim().is_empty() { continue; }

        // 行をカンマで分割し、トリム
        let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        // ヘッダーがない場合、最初のデータ行で列情報を設定
        if !has_header && row_count == 0 {
            expected_columns = values.len();
            if expected_columns == 0 { return Err(format!("First data line (line {}) is empty, cannot determine columns", line_number).into()); }
            // 列名を "column_0", "column_1", ... として生成
            columns = (0..expected_columns).map(|i| format!("column_{}", i)).collect();
            // データ格納用のVecを初期化
            for column_name in &columns { data.insert(column_name.clone(), Vec::new()); }
        }

        // 列数が期待値と一致するかチェック
        if values.len() != expected_columns { return Err(format!("Row {} has {} values, but expected {} columns", line_number, values.len(), expected_columns).into()); }

        // 各値をパースして対応する列のVecに追加
        for (col_idx, value_str) in values.iter().enumerate() {
            let column_name = &columns[col_idx];
            // 文字列を型Tにパース (T: FromStr)
            let parsed_value = value_str.parse::<T>().map_err(|_| {
                format!("Failed to parse value '{}' in column '{}' (row {}) as type '{}'", value_str, column_name, line_number, std::any::type_name::<T>()) // T: 'static
            })?;
            // data HashMap内のVecに値を追加 (unwrapはキーが存在するはずなので安全)
            data.get_mut(column_name).unwrap().push(parsed_value);
        }
        row_count += 1; // 処理した行数をカウント
    }

    // CSV読み込み後の最終チェック
    if has_header && row_count == 0 {
        // ヘッダーありだがデータ行がなかった場合 -> ヘッダーのみの空DataFrameを返す
    }
    if !has_header && row_count == 0 {
        // ヘッダーなしでデータ行もなかった場合 -> 完全な空DataFrameを返す
        return Ok(DataFrame::new());
    }

    // 最終的なDataFrameを構築して返す
    Ok(DataFrame { data, columns, row_count })
}


// --- main関数 ---
fn main() -> Result<(), Box<dyn Error>> {
    // === i32 DataFrame ===
    println!("--- i32 DataFrame Example ---");
    let mut df_i32: DataFrame<i32> = DataFrame::new();
    df_i32.add_row(HashMap::from([("id".to_string(), 1), ("age".to_string(), 30), ("height".to_string(), 180)]))?;
    df_i32.add_row(HashMap::from([("id".to_string(), 2), ("age".to_string(), 25), ("height".to_string(), 175)]))?;
    df_i32.add_row(HashMap::from([("id".to_string(), 3), ("age".to_string(), 30), ("height".to_string(), 185)]))?;
    println!("DataFrame (i32):\n{:?}", df_i32);
    println!("Sum of age: {:?}", df_i32.sum("age")?);
    println!("Mean of age (using num_cast): {:?}", df_i32.mean("age")?); // Uses safe cast
    let filtered_df_i32 = df_i32.filter(|row| **row.get(&"age".to_string()).expect("Age missing") > 25)?;
    println!("Filtered DataFrame (age > 25):\n{:?}", filtered_df_i32);
    let sorted_df_i32 = df_i32.sort_by(|a, b| {
        let age_a = **a.get(&"age".to_string()).expect("Age missing");
        let age_b = **b.get(&"age".to_string()).expect("Age missing");
        match age_a.cmp(&age_b) {
             std::cmp::Ordering::Equal => {
                 let height_a = **a.get(&"height".to_string()).expect("Height missing");
                 let height_b = **b.get(&"height".to_string()).expect("Height missing");
                 height_b.cmp(&height_a) // height desc
             }
             other => other,
        }
    });
    println!("Sorted DataFrame (by age asc, height desc):\n{:?}", sorted_df_i32);
    println!("-----------------------------");
    println!();

    // === String DataFrame ===
    println!("--- String DataFrame Example ---");
    let mut df_string = DataFrame::<String>::new();
    df_string.add_row(HashMap::from([("name".to_string(), "Alice".to_string()), ("city".to_string(), "New York".to_string())]))?;
    df_string.add_row(HashMap::from([("name".to_string(), "Bob".to_string()), ("city".to_string(), "Los Angeles".to_string())]))?;
    df_string.add_row(HashMap::from([("name".to_string(), "Charlie".to_string()), ("city".to_string(), "New York".to_string())]))?;
    let mut df_string2 = DataFrame::<String>::new();
    df_string2.add_row(HashMap::from([("name".to_string(), "Alice".to_string()), ("state".to_string(), "NY".to_string()), ("zip".to_string(), "10001".to_string())]))?;
    df_string2.add_row(HashMap::from([("name".to_string(), "Bob".to_string()), ("state".to_string(), "CA".to_string()), ("zip".to_string(), "90001".to_string())]))?;
    df_string2.add_row(HashMap::from([("name".to_string(), "David".to_string()), ("state".to_string(), "TX".to_string()), ("zip".to_string(), "75001".to_string())]))?;
    println!("DataFrame 1 (String):\n{:?}", df_string);
    println!("DataFrame 2 (String):\n{:?}", df_string2);
    let joined_df_string = df_string.join(&df_string2, "name")?;
    println!("Joined DataFrame (inner join on name):\n{:?}", joined_df_string);
    println!("Concatenated names: {:?}", df_string.concat("name", ", ")?);
    println!("-----------------------------");
    println!();

    // === CSV Reading (f64) ===
    println!("--- CSV Reading Example (f64) ---");
    let file_path_str = "data_temp.csv";
    let file_path = Path::new(file_path_str);
    // Create dummy CSV
    {
        let mut file = File::create(file_path)?;
        writeln!(file, "value_a,value_b,label")?;
        writeln!(file, "1.1, 2.2, apple")?;
        writeln!(file, "3.3, 4.4, banana")?;
        writeln!(file, " 5.5 ,6.6, apple")?;
        writeln!(file, "")?; // empty line test
    }
    println!("Attempting to read: {}", file_path_str);
    // Try reading as f64 (should fail on 'label' column)
    let df_from_csv_f64_result: Result<DataFrame<f64>, Box<dyn Error>> = read_csv(file_path, true);
    match df_from_csv_f64_result {
        Ok(df) => { println!("Successfully read CSV as f64 (This shouldn't happen):\n{:?}", df); }
        Err(e) => { println!("Correctly failed to read CSV as f64:"); eprintln!("Error: {}", e); } // Expected
    }
    println!();
    // Try reading as String (should succeed)
    let df_from_csv_string_result: Result<DataFrame<String>, Box<dyn Error>> = read_csv(file_path, true);
     match df_from_csv_string_result {
        Ok(df) => {
            println!("Successfully read CSV as String:\n{:?}", df);
            if df.contains_column("label") { println!("Concatenated labels: {:?}", df.concat("label", " | ")?); }
        }
        Err(e) => { eprintln!("Failed to read CSV as String (This shouldn't happen): {}", e); }
    }
    // Clean up dummy file
    if let Err(e) = std::fs::remove_file(file_path) { eprintln!("Warning: Failed to remove temporary file '{}': {}", file_path_str, e); }
    println!("-----------------------------");

    Ok(()) // Main finishes successfully
}

