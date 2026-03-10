#![allow(clippy::result_large_err)]
//! Integration tests for Arrow Flight RPC support.
//!
//! Conversion tests run under `--features distributed`.
//! Full server/client round-trip tests require `--features flight`.

#[cfg(feature = "distributed")]
mod conversion_tests {
    use pandrs::dataframe::DataFrame;
    use pandrs::distributed::flight::conversion::{
        dataframe_to_record_batch, record_batch_to_dataframe, record_batches_to_dataframe,
    };
    use pandrs::series::base::Series;

    fn make_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(
                vec!["1".to_string(), "2".to_string(), "3".to_string()],
                Some("id".to_string()),
            )
            .expect("series"),
        )
        .expect("add id");
        df.add_column(
            "name".to_string(),
            Series::new(
                vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()],
                Some("name".to_string()),
            )
            .expect("series"),
        )
        .expect("add name");
        df.add_column(
            "score".to_string(),
            Series::new(
                vec!["9.5".to_string(), "8.0".to_string(), "7.3".to_string()],
                Some("score".to_string()),
            )
            .expect("series"),
        )
        .expect("add score");
        df
    }

    #[test]
    fn test_dataframe_to_record_batch_column_count() {
        let df = make_test_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        assert_eq!(batch.num_columns(), 3, "Expected 3 columns in RecordBatch");
    }

    #[test]
    fn test_dataframe_to_record_batch_row_count() {
        let df = make_test_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        assert_eq!(batch.num_rows(), 3, "Expected 3 rows in RecordBatch");
    }

    #[test]
    fn test_dataframe_to_record_batch_column_names() {
        let df = make_test_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let schema = batch.schema();
        let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(col_names.contains(&"id"));
        assert!(col_names.contains(&"name"));
        assert!(col_names.contains(&"score"));
    }

    #[test]
    fn test_record_batch_to_dataframe_column_names() {
        let df = make_test_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let df2 = record_batch_to_dataframe(&batch).expect("to_dataframe");
        assert_eq!(
            df2.column_names(),
            df.column_names(),
            "Column names must survive round-trip"
        );
    }

    #[test]
    fn test_record_batch_to_dataframe_row_count() {
        let df = make_test_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let df2 = record_batch_to_dataframe(&batch).expect("to_dataframe");
        assert_eq!(
            df2.row_count(),
            df.row_count(),
            "Row count must survive round-trip"
        );
    }

    #[test]
    fn test_record_batches_to_dataframe_concatenation() {
        let df = make_test_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");

        // Concatenate the same batch twice
        let combined =
            record_batches_to_dataframe(&[batch.clone(), batch.clone()]).expect("concat");

        assert_eq!(
            combined.row_count(),
            df.row_count() * 2,
            "Concatenated DataFrame should have 2x rows"
        );
        assert_eq!(
            combined.column_names(),
            df.column_names(),
            "Column names must be preserved after concatenation"
        );
    }

    #[test]
    fn test_empty_dataframe_conversion() {
        let df = DataFrame::new();
        let batch = dataframe_to_record_batch(&df).expect("empty df to_record_batch");
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 0);

        let df2 = record_batch_to_dataframe(&batch).expect("empty batch to_dataframe");
        assert_eq!(df2.row_count(), 0);
        assert_eq!(df2.column_count(), 0);
    }

    #[test]
    fn test_integer_column_type_detection() {
        let mut df = DataFrame::new();
        df.add_column(
            "count".to_string(),
            Series::new(
                vec!["10".to_string(), "20".to_string(), "30".to_string()],
                Some("count".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let schema = batch.schema();
        let field = schema.field(0);

        // Integer strings should be detected as Int64
        assert_eq!(
            *field.data_type(),
            arrow::datatypes::DataType::Int64,
            "Integer column should be encoded as Int64"
        );
    }

    #[test]
    fn test_float_column_type_detection() {
        let mut df = DataFrame::new();
        df.add_column(
            "ratio".to_string(),
            Series::new(
                vec!["1.5".to_string(), "2.7".to_string(), "3.14".to_string()],
                Some("ratio".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let schema = batch.schema();
        let field = schema.field(0);

        assert_eq!(
            *field.data_type(),
            arrow::datatypes::DataType::Float64,
            "Float column should be encoded as Float64"
        );
    }

    #[test]
    fn test_string_column_type_detection() {
        let mut df = DataFrame::new();
        df.add_column(
            "label".to_string(),
            Series::new(
                vec!["foo".to_string(), "bar".to_string(), "baz".to_string()],
                Some("label".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let schema = batch.schema();
        let field = schema.field(0);

        assert_eq!(
            *field.data_type(),
            arrow::datatypes::DataType::Utf8,
            "String column should be encoded as Utf8"
        );
    }
}

// ---------------------------------------------------------------------------
// Full server/client round-trip (requires `flight` feature + tokio runtime)
// ---------------------------------------------------------------------------

#[cfg(feature = "flight")]
mod flight_rpc_tests {
    use pandrs::dataframe::DataFrame;
    use pandrs::distributed::flight::{PandRsFlightClient, PandRsFlightServer};
    use pandrs::series::base::Series;
    use std::time::Duration;

    fn make_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "value".to_string(),
            Series::new(
                vec!["42".to_string(), "99".to_string()],
                Some("value".to_string()),
            )
            .expect("series"),
        )
        .expect("add");
        df
    }

    /// Find a free TCP port by binding to port 0 and recording the assigned address.
    fn free_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.local_addr().expect("addr").port()
    }

    #[tokio::test]
    async fn test_flight_server_starts_and_client_connects() {
        let port = free_port();
        let server = PandRsFlightServer::new(port);
        let df = make_df();
        server.register_dataframe("test_df", &df).expect("register");

        let _handle = server.serve_background().expect("serve_background");

        // Give the server a moment to bind
        tokio::time::sleep(Duration::from_millis(200)).await;

        let endpoint = format!("http://127.0.0.1:{port}");
        let mut client = PandRsFlightClient::new(&endpoint);
        client.connect().await.expect("connect");

        let datasets = client.list_datasets().await.expect("list_datasets");
        assert!(
            datasets.contains(&"test_df".to_string()),
            "Server should list 'test_df' but got: {datasets:?}"
        );
    }

    #[tokio::test]
    async fn test_flight_get_dataframe_roundtrip() {
        let port = free_port();
        let server = PandRsFlightServer::new(port);
        let df = make_df();
        server
            .register_dataframe("round_trip", &df)
            .expect("register");
        let _handle = server.serve_background().expect("serve_background");

        tokio::time::sleep(Duration::from_millis(200)).await;

        let endpoint = format!("http://127.0.0.1:{port}");
        let mut client = PandRsFlightClient::new(&endpoint);
        client.connect().await.expect("connect");

        let fetched = client
            .get_dataframe("round_trip")
            .await
            .expect("get_dataframe");
        assert_eq!(
            fetched.row_count(),
            df.row_count(),
            "Row count should match after round-trip"
        );
        assert_eq!(
            fetched.column_names(),
            df.column_names(),
            "Column names should match after round-trip"
        );
    }

    #[tokio::test]
    async fn test_flight_put_and_get_dataframe() {
        let port = free_port();
        let server = PandRsFlightServer::new(port);
        let _handle = server.serve_background().expect("serve_background");

        tokio::time::sleep(Duration::from_millis(200)).await;

        let endpoint = format!("http://127.0.0.1:{port}");
        let mut client = PandRsFlightClient::new(&endpoint);
        client.connect().await.expect("connect");

        let df = make_df();
        client
            .put_dataframe("pushed_df", &df)
            .await
            .expect("put_dataframe");

        // Allow server to process DoPut
        tokio::time::sleep(Duration::from_millis(50)).await;

        let fetched = client
            .get_dataframe("pushed_df")
            .await
            .expect("get_dataframe after put");
        assert_eq!(
            fetched.row_count(),
            df.row_count(),
            "Row count should match after put+get"
        );
    }

    #[test]
    fn test_server_list_datasets_local() {
        let server = PandRsFlightServer::new(59000);
        assert!(server.list_datasets().is_empty());

        let df = make_df();
        server.register_dataframe("ds1", &df).expect("register");
        let names = server.list_datasets();
        assert_eq!(names, vec!["ds1".to_string()]);

        server.unregister("ds1").expect("unregister");
        assert!(server.list_datasets().is_empty());
    }
}
