use chrono::NaiveDate;
use pandrs::temporal::{date_range, Frequency, Temporal, TimeSeries};
use pandrs::NA;

#[test]
fn test_date_range_creation() {
    // 日付範囲の作成
    let start = NaiveDate::from_str("2023-01-01").unwrap();
    let end = NaiveDate::from_str("2023-01-10").unwrap();

    // 毎日の日付範囲
    let daily_range = date_range(start, end, Frequency::Daily, true).unwrap();
    assert_eq!(daily_range.len(), 10); // 両端を含む10日間
    assert_eq!(daily_range[0], start);
    assert_eq!(daily_range[9], end);

    // 週次の日付範囲
    let weekly_range = date_range(start, end, Frequency::Weekly, true).unwrap();
    assert_eq!(weekly_range.len(), 2); // 2週間（部分的な週も含む）
}

#[test]
fn test_frequency_parsing() {
    assert_eq!(Frequency::from_str("D"), Some(Frequency::Daily));
    assert_eq!(Frequency::from_str("day"), Some(Frequency::Daily));
    assert_eq!(Frequency::from_str("DAILY"), Some(Frequency::Daily));

    assert_eq!(Frequency::from_str("W"), Some(Frequency::Weekly));
    assert_eq!(Frequency::from_str("M"), Some(Frequency::Monthly));
    assert_eq!(Frequency::from_str("Q"), Some(Frequency::Quarterly));
    assert_eq!(Frequency::from_str("Y"), Some(Frequency::Yearly));

    assert_eq!(Frequency::from_str("invalid"), None);
}

#[test]
fn test_time_series_creation() {
    // 日付範囲を作成
    let dates = date_range(
        NaiveDate::from_str("2023-01-01").unwrap(),
        NaiveDate::from_str("2023-01-05").unwrap(),
        Frequency::Daily,
        true,
    )
    .unwrap();

    // 値を作成
    let values = vec![
        NA::Value(10.0),
        NA::Value(20.0),
        NA::NA,
        NA::Value(30.0),
        NA::Value(40.0),
    ];

    // TimeSeries を作成
    let ts = TimeSeries::new(values, dates, Some("test".to_string())).unwrap();

    assert_eq!(ts.len(), 5);
    assert_eq!(ts.name(), Some(&"test".to_string()));
}

#[test]
fn test_time_series_filter() {
    // 日付範囲を作成
    let dates = date_range(
        NaiveDate::from_str("2023-01-01").unwrap(),
        NaiveDate::from_str("2023-01-10").unwrap(),
        Frequency::Daily,
        true,
    )
    .unwrap();

    // 値を作成（値の内容は重要ではない）
    let values = (0..10).map(|i| NA::Value(i as f64)).collect();

    // TimeSeries を作成
    let ts = TimeSeries::new(values, dates, None).unwrap();

    // 期間でフィルタリング
    let start_filter = NaiveDate::from_str("2023-01-03").unwrap();
    let end_filter = NaiveDate::from_str("2023-01-07").unwrap();

    let filtered = ts.filter_by_time(&start_filter, &end_filter).unwrap();

    // 1/3 から 1/7 までの 5 日間
    assert_eq!(filtered.len(), 5);
    assert_eq!(filtered.timestamps()[0], start_filter);
    assert_eq!(filtered.timestamps()[4], end_filter);
}

#[test]
fn test_rolling_mean() {
    // 日付範囲を作成
    let dates = date_range(
        NaiveDate::from_str("2023-01-01").unwrap(),
        NaiveDate::from_str("2023-01-05").unwrap(),
        Frequency::Daily,
        true,
    )
    .unwrap();

    // 値を作成
    let values = vec![
        NA::Value(10.0),
        NA::Value(20.0),
        NA::Value(30.0),
        NA::Value(40.0),
        NA::Value(50.0),
    ];

    // TimeSeries を作成
    let ts = TimeSeries::new(values, dates, None).unwrap();

    // 3日間の移動平均
    let window = 3;
    let ma = ts.rolling_mean(window).unwrap();

    assert_eq!(ma.len(), 5);

    // 最初の window-1 個の要素は NA
    assert!(ma.values()[0].is_na());
    assert!(ma.values()[1].is_na());

    // 移動平均の計算結果を確認
    if let NA::Value(v) = ma.values()[2] {
        assert_eq!(v, (10.0 + 20.0 + 30.0) / 3.0);
    } else {
        panic!("Expected a value at index 2");
    }

    if let NA::Value(v) = ma.values()[3] {
        assert_eq!(v, (20.0 + 30.0 + 40.0) / 3.0);
    } else {
        panic!("Expected a value at index 3");
    }

    if let NA::Value(v) = ma.values()[4] {
        assert_eq!(v, (30.0 + 40.0 + 50.0) / 3.0);
    } else {
        panic!("Expected a value at index 4");
    }
}
