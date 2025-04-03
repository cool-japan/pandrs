use std::io::{Read};
use std::fs::File;
use std::path::Path;
use pandrs::{DataFrame, Series};

// CSVファイル操作のテスト (一時ファイルを利用)
#[test]
fn test_csv_io() {
    // テスト用一時ファイルパス
    let temp_path = Path::new("temp_test.csv");
    
    // テスト用DataFrameを作成
    let mut df = DataFrame::new();
    let names = Series::new(
        vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()],
        Some("name".to_string())
    ).unwrap();
    let ages = Series::new(vec![30, 25, 35], Some("age".to_string())).unwrap();
    
    df.add_column("name".to_string(), names).unwrap();
    df.add_column("age".to_string(), ages).unwrap();
    
    // CSVに書き出し
    let write_result = df.to_csv(&temp_path);
    
    // テスト後に一時ファイルを削除するため
    struct CleanupGuard<'a>(&'a Path);
    impl<'a> Drop for CleanupGuard<'a> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }
    let _guard = CleanupGuard(temp_path);
    
    // 書き出しが成功したことを確認
    assert!(write_result.is_ok());
    
    // ファイルが存在することを確認
    assert!(temp_path.exists());
    
    // ファイルの内容を確認
    let mut file = File::open(temp_path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    
    // CSVのヘッダーと内容を確認
    let lines: Vec<&str> = contents.lines().collect();
    assert!(lines.len() >= 1);  // 少なくともヘッダー行がある
    
    // 最小限のチェック: ヘッダーに列名が含まれていることを確認
    assert!(lines[0].contains("name"));
    assert!(lines[0].contains("age"));
    
    // TODO: CSVからの読み込みテストも追加する
    // 現在実装が不完全なため、完全なテストは保留
}

// JSONファイル操作のテスト (まだ実装中)
#[test]
fn test_json_io() {
    // JSON I/O機能がまだ完全に実装されていないため、
    // 簡単な構造チェックのみ行う
    
    use pandrs::io::json::{JsonOrient};
    
    // レコード形式と列形式が定義されていることを確認
    let _record_orient = JsonOrient::Records;
    let _column_orient = JsonOrient::Columns;
    
    // 将来的にはここにJSONのI/Oテストを追加
}