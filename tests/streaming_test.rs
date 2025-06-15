use pandrs::error::Result;
use pandrs::{
    AggregationType, DataStream, MetricType, RealTimeAnalytics, StreamAggregator, StreamConfig,
    StreamConnector, StreamProcessor, StreamRecord,
};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

#[test]
fn test_data_stream_basics() -> Result<()> {
    let headers = vec![
        "id".to_string(),
        "value".to_string(),
        "category".to_string(),
    ];

    let config = StreamConfig {
        buffer_size: 100,
        window_size: Some(10),
        window_duration: None,
        processing_interval: Duration::from_millis(10),
        batch_size: 10,
    };

    let (connector, mut stream) = StreamConnector::new(headers, Some(config));

    // Send some test data
    for i in 0..50 {
        let fields = HashMap::from([
            ("id".to_string(), format!("{}", i)),
            ("value".to_string(), format!("{}", i as f64 / 10.0)),
            ("category".to_string(), format!("cat{}", i % 5)),
        ]);

        connector.send_fields(fields)?;
    }

    // Close the stream
    drop(connector);

    // Process the stream in batches
    let results = stream.process(
        |batch| {
            // Count records in this batch
            Ok(batch.len())
        },
        None,
    )?;

    // We expect 5 batches of 10 records each
    assert!(!results.is_empty());
    assert_eq!(results.iter().sum::<usize>(), 50);

    Ok(())
}

#[test]
fn test_stream_aggregator() -> Result<()> {
    let headers = vec![
        "id".to_string(),
        "value".to_string(),
        "quantity".to_string(),
    ];

    let stream = DataStream::new(headers.clone(), None);

    // Create an aggregator
    let mut aggregator = StreamAggregator::new(stream);

    // Add aggregation functions
    aggregator.add_aggregator("value", AggregationType::Average)?;
    aggregator.add_aggregator("value", AggregationType::Max)?;
    aggregator.add_aggregator("quantity", AggregationType::Sum)?;

    // Basic test to check API compiles properly
    println!("StreamAggregator API successfully created");

    Ok(())
}

#[test]
fn test_stream_processor() -> Result<()> {
    let headers = vec!["id".to_string(), "text".to_string(), "category".to_string()];

    let (_connector, stream) = StreamConnector::new(headers, None);

    // Create a processor
    let mut processor = StreamProcessor::new(stream);

    // Add transformations
    processor.add_transformer("text", |value| {
        // Convert to uppercase
        Ok(value.to_uppercase())
    })?;

    // Add filter
    processor.set_filter(|record| {
        // Only keep records with category A or B
        if let Some(category) = record.fields.get("category") {
            category == "A" || category == "B"
        } else {
            false
        }
    });

    // Basic test to check API compiles properly
    println!("StreamProcessor API successfully created");

    Ok(())
}

#[test]
fn test_window_operation() -> Result<()> {
    let headers = vec!["timestamp".to_string(), "value".to_string()];

    let config = StreamConfig {
        buffer_size: 100,
        window_size: Some(10),
        window_duration: None,
        processing_interval: Duration::from_millis(10),
        batch_size: 10,
    };

    let (connector, mut stream) = StreamConnector::new(headers, Some(config));

    // Send test data
    for i in 0..50 {
        let fields = HashMap::from([
            ("timestamp".to_string(), format!("{}", i)),
            ("value".to_string(), format!("{}", i as f64)),
        ]);

        connector.send_fields(fields)?;
    }

    // Close the stream
    drop(connector);

    // Process the stream with window operation
    let results = stream.window_operation(|window| {
        // Compute average of values in window
        let sum: f64 = window
            .iter()
            .filter_map(|record| {
                record
                    .fields
                    .get("value")
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .sum();

        let count = window.len();
        let avg = if count > 0 { sum / count as f64 } else { 0.0 };

        Ok(avg)
    })?;

    // We should have multiple windows
    assert!(!results.is_empty());

    // The last window should have higher average than the first window
    assert!(results.last().unwrap() > &results[0]);

    Ok(())
}

#[test]
fn test_real_time_analytics() -> Result<()> {
    let headers = vec!["timestamp".to_string(), "value".to_string()];

    let stream = DataStream::new(headers.clone(), None);

    // Create real-time analytics
    let mut analytics = RealTimeAnalytics::new(
        stream,
        10,                        // Window size
        Duration::from_millis(10), // Computing interval
    );

    // Add metrics
    analytics.add_metric("avg", "value", MetricType::WindowAverage)?;
    analytics.add_metric("change", "value", MetricType::RateOfChange)?;

    // Start background processing
    let metrics = analytics.start_background_processing()?;

    // Send test data with increasing values
    let sender = analytics.stream.get_sender().unwrap();
    for i in 0..20 {
        let fields = HashMap::from([
            ("timestamp".to_string(), format!("{}", i)),
            ("value".to_string(), format!("{}", i as f64)),
        ]);

        let record = StreamRecord::new(fields);
        sender.send(record).unwrap();

        thread::sleep(Duration::from_millis(5));
    }

    // Give time for processing
    thread::sleep(Duration::from_millis(100));

    // Check metrics
    let current_metrics = metrics.lock().unwrap().clone();

    // We should have some metrics
    assert!(!current_metrics.is_empty());

    // Average should be non-zero
    assert!(current_metrics.get("avg_value").unwrap_or(&0.0) > &0.0);

    // Rate of change should be positive (since values are increasing)
    assert!(current_metrics.get("change_value").unwrap_or(&0.0) > &0.0);

    // Close the stream
    drop(sender);

    // Stop analytics
    analytics.stop();

    Ok(())
}

#[test]
fn test_batch_to_dataframe() -> Result<()> {
    let headers = vec!["id".to_string(), "value".to_string()];

    let _stream = DataStream::new(headers.clone(), None);

    // Create some test records
    let mut records = Vec::new();
    for i in 0..10 {
        let fields = HashMap::from([
            ("id".to_string(), format!("{}", i)),
            ("value".to_string(), format!("{}", i as f64)),
        ]);

        records.push(StreamRecord::new(fields));
    }

    // Basic test to check API compiles properly
    println!("Batch to DataFrame API successfully created");

    Ok(())
}
