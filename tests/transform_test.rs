#[cfg(test)]
mod tests {
    use pandrs::{DataFrame, MeltOptions, Series, StackOptions, UnstackOptions};

    // DataBoxの文字列をクリーンな値に変換するヘルパー関数
    fn clean_databox_value(value: &str) -> String {
        let trimmed = value.trim_start_matches("DataBox(\"").trim_end_matches("\")");
        let value_str = if trimmed.starts_with("DataBox(") {
            trimmed.trim_start_matches("DataBox(").trim_end_matches(")")
        } else {
            trimmed
        };
        value_str.trim_matches('"').to_string()
    }

    #[test]
    fn test_melt() {
        // テスト用データフレーム作成
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "A".to_string(),
            Series::new(vec!["a1", "a2"], Some("A".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "B".to_string(),
            Series::new(vec!["b1", "b2"], Some("B".to_string())).unwrap(),
        )
        .unwrap();

        // melt操作
        let options = MeltOptions {
            id_vars: Some(vec!["id".to_string()]),
            value_vars: Some(vec!["A".to_string(), "B".to_string()]),
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
        };

        let melted = df.melt(&options).unwrap();

        // 検証
        assert_eq!(melted.column_count(), 3); // id, variable, value
        assert_eq!(melted.row_count(), 4);    // 2行 x 2列 = 4行

        // 列名の確認
        let columns = melted.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"variable".to_string()));
        assert!(columns.contains(&"value".to_string()));

        // データの確認
        let id_col = melted.get_column("id").unwrap();
        let var_col = melted.get_column("variable").unwrap();
        let val_col = melted.get_column("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[0].to_string()), "A");
        assert_eq!(clean_databox_value(&val_col.values()[0].to_string()), "a1");

        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[1].to_string()), "B");
        assert_eq!(clean_databox_value(&val_col.values()[1].to_string()), "b1");

        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "2");
        assert_eq!(clean_databox_value(&var_col.values()[2].to_string()), "A");
        assert_eq!(clean_databox_value(&val_col.values()[2].to_string()), "a2");

        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "2");
        assert_eq!(clean_databox_value(&var_col.values()[3].to_string()), "B");
        assert_eq!(clean_databox_value(&val_col.values()[3].to_string()), "b2");
    }

    #[test]
    fn test_stack() {
        // テスト用データフレーム作成
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "A".to_string(),
            Series::new(vec!["a1", "a2"], Some("A".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "B".to_string(),
            Series::new(vec!["b1", "b2"], Some("B".to_string())).unwrap(),
        )
        .unwrap();

        // stack操作
        let options = StackOptions {
            columns: Some(vec!["A".to_string(), "B".to_string()]),
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
            dropna: false,
        };

        let stacked = df.stack(&options).unwrap();

        // 検証
        assert_eq!(stacked.column_count(), 3); // id, variable, value
        assert_eq!(stacked.row_count(), 4);    // 2行 x 2列 = 4行

        // データの確認
        let id_col = stacked.get_column("id").unwrap();
        let var_col = stacked.get_column("variable").unwrap();
        let val_col = stacked.get_column("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[0].to_string()), "A");
        assert_eq!(clean_databox_value(&val_col.values()[0].to_string()), "a1");

        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[1].to_string()), "B");
        assert_eq!(clean_databox_value(&val_col.values()[1].to_string()), "b1");
    }

    #[test]
    fn test_unstack() {
        // テスト用の長形式データフレーム作成
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(
                vec!["1", "1", "2", "2"],
                Some("id".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "variable".to_string(),
            Series::new(
                vec!["A", "B", "A", "B"],
                Some("variable".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Series::new(
                vec!["a1", "b1", "a2", "b2"],
                Some("value".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        // unstack操作
        let options = UnstackOptions {
            var_column: "variable".to_string(),
            value_column: "value".to_string(),
            index_columns: Some(vec!["id".to_string()]),
            fill_value: None,
        };

        let unstacked = df.unstack(&options).unwrap();

        // 検証
        assert_eq!(unstacked.column_count(), 3); // id, A, B
        assert_eq!(unstacked.row_count(), 2);    // 2行（id別）

        // 列名の確認
        let columns = unstacked.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"A".to_string()));
        assert!(columns.contains(&"B".to_string()));

        // データの確認
        let id_col = unstacked.get_column("id").unwrap();
        let a_col = unstacked.get_column("A").unwrap();
        let b_col = unstacked.get_column("B").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&a_col.values()[0].to_string()), "a1");
        assert_eq!(clean_databox_value(&b_col.values()[0].to_string()), "b1");

        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&a_col.values()[1].to_string()), "a2");
        assert_eq!(clean_databox_value(&b_col.values()[1].to_string()), "b2");
    }

    #[test]
    fn test_conditional_aggregate() {
        // テスト用データフレーム作成
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec!["食品", "電化製品", "食品", "衣類"],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "sales".to_string(),
            Series::new(
                vec!["1000", "1500", "800", "1200"],
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        // 条件付き集計: 売上が1000以上の行のみでカテゴリ別の合計を計算
        let result = df
            .conditional_aggregate(
                "category",
                "sales",
                |row| {
                    if let Some(sales_str) = row.get("sales") {
                        if let Ok(sales) = sales_str.parse::<i32>() {
                            return sales >= 1000;
                        }
                    }
                    false
                },
                |values| {
                    let sum: i32 = values
                        .iter()
                        .filter_map(|v| v.parse::<i32>().ok())
                        .sum();
                    sum.to_string()
                },
            )
            .unwrap();

        // 検証
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 3); // 食品、電化製品、衣類

        // 集計結果の確認
        let cat_col = result.get_column("category").unwrap();
        let agg_col = result.get_column("sales_agg").unwrap();

        // カテゴリごとの集計値を確認
        // 注: 結果の順序は実装依存のため、各カテゴリについて個別にチェック
        for i in 0..result.row_count() {
            let category = clean_databox_value(&cat_col.values()[i].to_string());
            let agg_value = clean_databox_value(&agg_col.values()[i].to_string());

            if category == "食品" {
                assert_eq!(agg_value, "1000"); // 1000以上の食品は一つだけ
            } else if category == "電化製品" {
                assert_eq!(agg_value, "1500");
            } else if category == "衣類" {
                assert_eq!(agg_value, "1200");
            } else {
                panic!("予期しないカテゴリ: {}", category);
            }
        }
    }

    #[test]
    fn test_concat() {
        // 1つ目のデータフレーム
        let mut df1 = DataFrame::new();
        df1.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df1.add_column(
            "value".to_string(),
            Series::new(vec!["a", "b"], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        // 2つ目のデータフレーム
        let mut df2 = DataFrame::new();
        df2.add_column(
            "id".to_string(),
            Series::new(vec!["3", "4"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "value".to_string(),
            Series::new(vec!["c", "d"], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        // 結合操作
        let concat_df = DataFrame::concat(&[&df1, &df2], true).unwrap();

        // 検証
        assert_eq!(concat_df.column_count(), 2);
        assert_eq!(concat_df.row_count(), 4);

        // 列の確認
        let id_col = concat_df.get_column("id").unwrap();
        let value_col = concat_df.get_column("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&value_col.values()[0].to_string()), "a");
        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&value_col.values()[1].to_string()), "b");
        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "3");
        assert_eq!(clean_databox_value(&value_col.values()[2].to_string()), "c");
        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "4");
        assert_eq!(clean_databox_value(&value_col.values()[3].to_string()), "d");
    }

    #[test]
    fn test_concat_different_columns() {
        // 1つ目のデータフレーム
        let mut df1 = DataFrame::new();
        df1.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df1.add_column(
            "A".to_string(),
            Series::new(vec!["a1", "a2"], Some("A".to_string())).unwrap(),
        )
        .unwrap();

        // 2つ目のデータフレーム（異なる列を持つ）
        let mut df2 = DataFrame::new();
        df2.add_column(
            "id".to_string(),
            Series::new(vec!["3", "4"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "B".to_string(),
            Series::new(vec!["b3", "b4"], Some("B".to_string())).unwrap(),
        )
        .unwrap();

        // 結合操作
        let concat_df = DataFrame::concat(&[&df1, &df2], true).unwrap();

        // 検証
        assert_eq!(concat_df.column_count(), 3); // id, A, B
        assert_eq!(concat_df.row_count(), 4);

        // 列の確認
        let id_col = concat_df.get_column("id").unwrap();
        let a_col = concat_df.get_column("A").unwrap();
        let b_col = concat_df.get_column("B").unwrap();

        // id列
        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "3");
        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "4");

        // A列（df2にはA列がないので空値が入る）
        assert_eq!(clean_databox_value(&a_col.values()[0].to_string()), "a1");
        assert_eq!(clean_databox_value(&a_col.values()[1].to_string()), "a2");
        assert_eq!(clean_databox_value(&a_col.values()[2].to_string()), "");
        assert_eq!(clean_databox_value(&a_col.values()[3].to_string()), "");

        // B列（df1にはB列がないので空値が入る）
        assert_eq!(clean_databox_value(&b_col.values()[0].to_string()), "");
        assert_eq!(clean_databox_value(&b_col.values()[1].to_string()), "");
        assert_eq!(clean_databox_value(&b_col.values()[2].to_string()), "b3");
        assert_eq!(clean_databox_value(&b_col.values()[3].to_string()), "b4");
    }
}