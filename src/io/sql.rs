use std::collections::HashMap;
use std::path::Path;

use rusqlite::{Connection, Row, Statement};

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;

/// SQLクエリの実行結果からデータフレームを作成する
///
/// # 引数
///
/// * `query` - 実行するSQLクエリ
/// * `db_path` - データベースファイルのパス (SQLite用)
///
/// # 戻り値
///
/// * `Result<DataFrame>` - クエリ結果を含むデータフレーム、またはエラー
///
/// # 例
///
/// ```no_run
/// use pandrs::io::read_sql;
///
/// let df = read_sql("SELECT name, age FROM users WHERE age > 30", "users.db").unwrap();
/// ```
pub fn read_sql<P: AsRef<Path>>(query: &str, db_path: P) -> Result<DataFrame> {
    // データベース接続
    let mut conn = Connection::open(db_path)
        .map_err(|e| Error::IoError(format!("データベースに接続できませんでした: {}", e)))?;
    
    // クエリを準備
    let mut stmt = conn.prepare(query)
        .map_err(|e| Error::IoError(format!("SQLクエリの準備に失敗しました: {}", e)))?;
    
    // 列名を取得
    let column_names: Vec<String> = stmt.column_names().iter()
        .map(|&name| name.to_string())
        .collect();
    
    // 列ごとのデータを格納するためのマップ
    let mut column_data: HashMap<String, Vec<String>> = HashMap::new();
    for name in &column_names {
        column_data.insert(name.clone(), Vec::new());
    }
    
    // クエリを実行して結果を取得
    let mut rows = stmt.query([])
        .map_err(|e| Error::IoError(format!("SQLクエリの実行に失敗しました: {}", e)))?;
    
    // 各行のデータを処理
    while let Some(row) = rows.next()
        .map_err(|e| Error::IoError(format!("SQLクエリの結果取得に失敗しました: {}", e)))? {
        
        for (idx, name) in column_names.iter().enumerate() {
            let value = get_row_value(row, idx)?;
            if let Some(data) = column_data.get_mut(name) {
                data.push(value);
            }
        }
    }
    
    // データフレームを作成
    let mut df = DataFrame::new();
    
    // 列データからシリーズを作成してデータフレームに追加
    for name in column_names {
        if let Some(data) = column_data.get(&name) {
            if let Some(series) = infer_series_from_strings(&name, data)? {
                df.add_column(name.clone(), series)?;
            }
        }
    }
    
    Ok(df)
}

/// SQL文を実行する（結果を返さない）
///
/// # 引数
///
/// * `sql` - 実行するSQL文
/// * `db_path` - データベースファイルのパス (SQLite用)
///
/// # 戻り値
///
/// * `Result<usize>` - 影響を受けた行数、またはエラー
///
/// # 例
///
/// ```no_run
/// use pandrs::io::execute_sql;
///
/// let affected_rows = execute_sql("UPDATE users SET status = 'active' WHERE last_login > '2023-01-01'", "users.db").unwrap();
/// println!("影響を受けた行数: {}", affected_rows);
/// ```
pub fn execute_sql<P: AsRef<Path>>(sql: &str, db_path: P) -> Result<usize> {
    // データベース接続
    let mut conn = Connection::open(db_path)
        .map_err(|e| Error::IoError(format!("データベースに接続できませんでした: {}", e)))?;
    
    // SQL文を実行
    let affected_rows = conn.execute(sql, [])
        .map_err(|e| Error::IoError(format!("SQL文の実行に失敗しました: {}", e)))?;
    
    Ok(affected_rows)
}

/// データフレームをSQLデータベースのテーブルに書き込む
///
/// # 引数
///
/// * `df` - 書き込むデータフレーム
/// * `table_name` - テーブル名
/// * `db_path` - データベースファイルのパス (SQLite用)
/// * `if_exists` - テーブルが存在する場合の処理 ("fail", "replace", "append")
///
/// # 戻り値
///
/// * `Result<()>` - 成功した場合はOk(())、失敗した場合はエラー
///
/// # 例
///
/// ```no_run
/// use pandrs::dataframe::DataFrame;
/// use pandrs::io::write_to_sql;
/// use pandrs::series::Series;
///
/// let mut df = DataFrame::new();
/// df.add_series(Series::new(vec![1, 2, 3], Some("id".to_string())).unwrap()).unwrap();
/// df.add_series(Series::new(vec!["Alice", "Bob", "Charlie"], Some("name".to_string())).unwrap()).unwrap();
///
/// write_to_sql(&df, "users", "users.db", "replace").unwrap();
/// ```
pub fn write_to_sql<P: AsRef<Path>>(
    df: &DataFrame,
    table_name: &str,
    db_path: P,
    if_exists: &str,
) -> Result<()> {
    // データベース接続
    let mut conn = Connection::open(db_path)
        .map_err(|e| Error::IoError(format!("データベースに接続できませんでした: {}", e)))?;
    
    // テーブルが存在するか確認
    let table_exists = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
        .map_err(|e| Error::IoError(format!("テーブル確認クエリの準備に失敗しました: {}", e)))?
        .exists(&[&table_name])
        .map_err(|e| Error::IoError(format!("テーブル存在確認に失敗しました: {}", e)))?;
    
    // if_exists に基づいてテーブルを処理
    if table_exists {
        match if_exists {
            "fail" => {
                return Err(Error::IoError(format!("テーブル '{}' は既に存在します", table_name)));
            },
            "replace" => {
                // テーブルを削除
                conn.execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
                    .map_err(|e| Error::IoError(format!("テーブルの削除に失敗しました: {}", e)))?;
                
                // 新しいテーブルを作成
                create_table_from_df(&conn, df, table_name)?;
            },
            "append" => {
                // テーブルは既に存在するので、このままデータを追加
            },
            _ => {
                return Err(Error::IoError(format!("不明なif_exists値: {}", if_exists)));
            }
        }
    } else {
        // テーブルが存在しない場合は新規作成
        create_table_from_df(&conn, df, table_name)?;
    }
    
    // データの挿入
    // カラム名のリスト
    let column_names = df.column_names();
    let columns = column_names.join(", ");
    
    // プレースホルダーのリスト
    let placeholders: Vec<String> = (0..column_names.len())
        .map(|_| "?".to_string())
        .collect();
    let placeholders = placeholders.join(", ");
    
    // INSERT文を準備
    let insert_sql = format!("INSERT INTO {} ({}) VALUES ({})", table_name, columns, placeholders);
    
    // トランザクション開始
    {
        let tx = conn.transaction()
            .map_err(|e| Error::IoError(format!("トランザクションの開始に失敗しました: {}", e)))?;
            
        // 各行のデータを挿入（トランザクション内でステートメントを準備して実行）
        for row_idx in 0..df.row_count() {
            // 行データを取得
            let mut row_values: Vec<String> = Vec::new();
            for col_name in column_names.iter() {
                // 列の値を文字列として取得する
                if let Some(column) = df.get_column(col_name) {
                    let value = if let Some(val) = column.get(row_idx) {
                        val.clone()
                    } else {
                        String::new()
                    };
                    row_values.push(value);
                }
            }
            
            // INSERT実行
            let params: Vec<&dyn rusqlite::ToSql> = row_values.iter()
                .map(|s| s as &dyn rusqlite::ToSql)
                .collect();
            
            tx.execute(&insert_sql, params.as_slice())
                .map_err(|e| Error::IoError(format!("データの挿入に失敗しました: {}", e)))?;
        }
        
        // トランザクションをコミット
        tx.commit()
            .map_err(|e| Error::IoError(format!("トランザクションのコミットに失敗しました: {}", e)))?;
    }
    
    Ok(())
}

/// データフレームから新しいテーブルを作成する内部ヘルパー関数
fn create_table_from_df(conn: &Connection, df: &DataFrame, table_name: &str) -> Result<()> {
    // 列名と型のリストを作成
    let mut columns = Vec::new();
    
    for col_name in df.column_names() {
        // 各列を文字列シリーズとして取得し、データ型を判断
        if let Some(series) = df.get_column(col_name) {
            let sql_type = series_to_sql_type(&series)?;
            columns.push(format!("{} {}", col_name, sql_type));
        }
    }
    
    // CREATE TABLE文を作成して実行
    let create_sql = format!("CREATE TABLE {} ({})", table_name, columns.join(", "));
    conn.execute(&create_sql, [])
        .map_err(|e| Error::IoError(format!("テーブルの作成に失敗しました: {}", e)))?;
    
    Ok(())
}

/// シリーズからSQLデータ型を推測する
fn series_to_sql_type(series: &Series<String>) -> Result<String> {
    // シリーズの名前からデータ型を推測
    let series_type = series.name().map_or("unknown", |s| s).split_whitespace().next().unwrap_or("");
    
    match series_type {
        "i64" | "Int64" => Ok("INTEGER".to_string()),
        "f64" | "Float64" => Ok("REAL".to_string()),
        "bool" | "Boolean" => Ok("INTEGER".to_string()),  // SQLiteではブール値は整数として保存
        _ => Ok("TEXT".to_string()),  // デフォルトはTEXT
    }
}

/// SQL行から値を取得する
fn get_row_value(row: &Row, idx: usize) -> Result<String> {
    let value: Option<String> = row.get::<_, Option<String>>(idx)
        .map_err(|e| Error::IoError(format!("行データの取得に失敗しました: {}", e)))?;
    
    Ok(value.unwrap_or_else(|| "NULL".to_string()))
}

/// 文字列のベクトルからデータ型を推測してシリーズを作成する
fn infer_series_from_strings(name: &str, data: &[String]) -> Result<Option<Series<String>>> {
    if data.is_empty() {
        return Ok(None);
    }
    
    // 整数かどうかチェック
    let all_integers = data.iter().all(|s| {
        s.trim().parse::<i64>().is_ok() || s.trim().is_empty() || s.trim() == "NULL"
    });
    
    if all_integers {
        let values: Vec<i64> = data.iter()
            .map(|s| {
                if s.trim().is_empty() || s.trim() == "NULL" {
                    0
                } else {
                    s.trim().parse::<i64>().unwrap_or(0)
                }
            })
            .collect();
        let series = Series::new(values, Some(name.to_string()))?;
        let string_series = series.to_string_series()?;
        return Ok(Some(string_series));
    }
    
    // 浮動小数点数かどうかチェック
    let all_floats = data.iter().all(|s| {
        s.trim().parse::<f64>().is_ok() || s.trim().is_empty() || s.trim() == "NULL"
    });
    
    if all_floats {
        let values: Vec<f64> = data.iter()
            .map(|s| {
                if s.trim().is_empty() || s.trim() == "NULL" {
                    0.0
                } else {
                    s.trim().parse::<f64>().unwrap_or(0.0)
                }
            })
            .collect();
        let series = Series::new(values, Some(name.to_string()))?;
        let string_series = series.to_string_series()?;
        return Ok(Some(string_series));
    }
    
    // ブール値かどうかチェック
    let all_booleans = data.iter().all(|s| {
        let s = s.trim().to_lowercase();
        s == "true" || s == "false" || s == "1" || s == "0" || s.is_empty() || s == "null"
    });
    
    if all_booleans {
        let values: Vec<bool> = data.iter()
            .map(|s| {
                let s = s.trim().to_lowercase();
                s == "true" || s == "1"
            })
            .collect();
        let series = Series::new(values, Some(name.to_string()))?;
        let string_series = series.to_string_series()?;
        return Ok(Some(string_series));
    }
    
    // それ以外は文字列として扱う
    let values: Vec<String> = data.iter()
        .map(|s| {
            if s.trim() == "NULL" {
                "".to_string()
            } else {
                s.clone()
            }
        })
        .collect();
    
    Ok(Some(Series::new(values, Some(name.to_string()))?))
}