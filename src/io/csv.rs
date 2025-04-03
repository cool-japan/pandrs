use csv::{ReaderBuilder, Writer};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::error::{PandRSError, Result};
use crate::series::{Series, CategoricalOrder, StringCategorical};
use crate::DataFrame;

/// CSVファイルからDataFrameを読み込む
pub fn read_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<DataFrame> {
    let file = File::open(path.as_ref()).map_err(PandRSError::Io)?;

    // CSVリーダーを設定
    let mut rdr = ReaderBuilder::new()
        .has_headers(has_header)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(file);

    let mut df = DataFrame::new();

    // ヘッダー行を取得
    let headers: Vec<String> = if has_header {
        rdr.headers()
            .map_err(PandRSError::Csv)?
            .iter()
            .map(|h| h.to_string())
            .collect()
    } else {
        // ヘッダーがない場合は、最初の行から推測して"column_0", "column_1"などとする
        if let Some(first_record_result) = rdr.records().next() {
            let first_record = first_record_result.map_err(PandRSError::Csv)?;
            (0..first_record.len())
                .map(|i| format!("column_{}", i))
                .collect()
        } else {
            // ファイルが空の場合
            return Ok(DataFrame::new());
        }
    };

    // データを列ごとに収集
    let mut columns: HashMap<String, Vec<String>> = HashMap::new();
    for header in &headers {
        columns.insert(header.clone(), Vec::new());
    }

    // 各行を処理
    for result in rdr.records() {
        let record = result.map_err(PandRSError::Csv)?;
        for (i, header) in headers.iter().enumerate() {
            if i < record.len() {
                columns.get_mut(header).unwrap().push(record[i].to_string());
            } else {
                // 行の長さが足りない場合、空文字列を追加
                columns.get_mut(header).unwrap().push(String::new());
            }
        }
    }

    // 列をDataFrameに追加
    for header in headers {
        if let Some(values) = columns.remove(&header) {
            let series = Series::new(values, Some(header.clone()))?;
            df.add_column(header, series)?;
        }
    }

    Ok(df)
}

/// DataFrameをCSVファイルに書き込む
pub fn write_csv<P: AsRef<Path>>(df: &DataFrame, path: P) -> Result<()> {
    let file = File::create(path.as_ref()).map_err(PandRSError::Io)?;
    let mut wtr = Writer::from_writer(file);

    // ヘッダー行を書き込む
    wtr.write_record(df.column_names())
        .map_err(PandRSError::Csv)?;

    // 各行のデータを書き込む
    let row_count = df.row_count();
    
    // 行がない場合は何もせず終了
    if row_count == 0 {
        wtr.flush().map_err(PandRSError::Io)?;
        return Ok(());
    }
    
    for i in 0..row_count {
        let mut row = Vec::new();
        
        for col_name in df.column_names() {
            if let Some(series) = df.get_column(col_name) {
                if i < series.len() {
                    row.push(series.values()[i].to_string());
                } else {
                    row.push(String::new());
                }
            }
        }
        
        wtr.write_record(&row).map_err(PandRSError::Csv)?;
    }

    wtr.flush().map_err(PandRSError::Io)?;
    Ok(())
}
