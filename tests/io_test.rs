use pandrs::{DataFrame, Series, PandRSError};
use std::fs::File;
use std::io::Read;
use std::path::Path;

// CSVファイル操作のテスト (一時ファイルを利用)
#[test]
fn test_csv_io() -> Result<(), PandRSError> {
    // テスト用一時ファイルパス
    let temp_path = Path::new("temp_test.csv");

    // テスト用DataFrameを作成
    let mut df = DataFrame::new();
    let names = Series::new(
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ],
        Some("name".to_string()),
    )?;
    let ages = Series::new(vec![30, 25, 35], Some("age".to_string()))?;

    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;

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
    assert!(lines.len() >= 1); // 少なくともヘッダー行がある

    // 最小限のチェック: ヘッダーに列名が含まれていることを確認
    assert!(lines[0].contains("name"));
    assert!(lines[0].contains("age"));

    // CSVからの読み込みテストも追加
    // from_csvにヘッダーがあることを指定
    let df_from_csv = DataFrame::from_csv(&temp_path, true)?;
    
    // 読み込んだDataFrameの検証
    assert_eq!(df_from_csv.column_names().len(), 2, "列数が一致すること");
    assert!(df_from_csv.contains_column("name"), "name列が存在すること");
    assert!(df_from_csv.contains_column("age"), "age列が存在すること");
    
    let row_count = df_from_csv.row_count();
    assert_eq!(row_count, 3, "行数が元のデータと一致すること");
    
    // name列の値を確認 - DataFrame::get_column_string_valuesを使用
    let name_values = df_from_csv.get_column_string_values("name")?;
    assert!(name_values[0].contains("Alice"), "最初の行のname列の値が正しいこと");
    assert!(name_values[1].contains("Bob"), "2行目のname列の値が正しいこと");
    assert!(name_values[2].contains("Charlie"), "3行目のname列の値が正しいこと");
    
    // age列の値を確認 - DataFrame::get_column_string_valuesを使用して文字列内容を確認
    let age_str_values = df_from_csv.get_column_string_values("age")?;
    assert!(age_str_values[0].contains("30"), "最初の行のage列の値が正しいこと");
    assert!(age_str_values[1].contains("25"), "2行目のage列の値が正しいこと");
    assert!(age_str_values[2].contains("35"), "3行目のage列の値が正しいこと");
    
    Ok(())
}

// JSONファイル操作のテスト (まだ実装中)
#[test]
fn test_json_io() {
    // JSON I/O機能がまだ完全に実装されていないため、
    // 簡単な構造チェックのみ行う

    use pandrs::io::json::JsonOrient;

    // レコード形式と列形式が定義されていることを確認
    let _record_orient = JsonOrient::Records;
    let _column_orient = JsonOrient::Columns;

    // 将来的にはここにJSONのI/Oテストを追加
}
