#[cfg(feature = "streaming")]
use pandrs::error::Result;
#[cfg(feature = "streaming")]
use pandrs::streaming::{
    AggregationType, DataStream, MetricType, RealTimeAnalytics, StreamAggregator, StreamConfig,
    StreamConnector, StreamProcessor, StreamRecord,
};
#[cfg(feature = "streaming")]
#[allow(clippy::single_component_path_imports)]
use rand;
#[cfg(feature = "streaming")]
use std::collections::HashMap;
#[cfg(feature = "streaming")]
use std::thread;
#[cfg(feature = "streaming")]
use std::time::{Duration, Instant};

#[cfg(not(feature = "streaming"))]
fn main() {
    println!("This example requires the 'streaming' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example streaming_example --features streaming");
}

#[cfg(feature = "streaming")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // This example demonstrates various streaming data processing capabilities
    println!("Streaming Data Processing Example");
    println!("--------------------------------");

    // Example 1: Process a CSV file as a stream
    println!("\nExample 1: Reading CSV as a stream");
    csv_stream_example()?;

    // Example 2: Stream aggregation
    println!("\nExample 2: Stream aggregation");
    stream_aggregation_example()?;

    // Example 3: Real-time analytics
    println!("\nExample 3: Real-time analytics");
    realtime_analytics_example()?;

    // Example 4: Custom stream connector
    println!("\nExample 4: Custom stream connector");
    custom_connector_example()?;

    Ok(())
}

// Example showing how to process a CSV file as a stream
#[cfg(feature = "streaming")]
#[allow(clippy::result_large_err)]
fn csv_stream_example() -> Result<()> {
    // Create a stream from a CSV file (replace with actual file path)
    let config = StreamConfig {
        buffer_size: 1000,
        window_size: Some(100),
        window_duration: None,
        processing_interval: Duration::from_millis(10),
        batch_size: 100,
    };

    // In a real application, you would use an actual CSV file
    // For this example, we'll create a simulated stream
    let headers = vec![
        "timestamp".to_string(),
        "value".to_string(),
        "category".to_string(),
    ];

    // Create a stream connector and data stream
    let (connector, mut stream) = StreamConnector::new(headers.clone(), Some(config));

    // Start a thread to generate data
    thread::spawn(move || {
        let categories = ["A", "B", "C"];

        for i in 0..500 {
            let fields = HashMap::from([
                ("timestamp".to_string(), format!("{}", i)),
                ("value".to_string(), format!("{}", i as f64 / 10.0)),
                ("category".to_string(), categories[i % 3].to_string()),
            ]);

            // Send to stream
            let _ = connector.send_fields(fields);

            // Simulate delay between records
            thread::sleep(Duration::from_millis(1));
        }

        // Close the stream
        drop(connector);
    });

    // Process the stream with a window operation
    let results = stream.window_operation(|window| {
        // For this example, we'll just compute the average of the 'value' column
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

        // Print window stats
        println!("Window size: {}, Average value: {:.2}", count, avg);

        Ok(avg)
    })?;

    println!("Processed {} windows", results.len());
    println!(
        "Final window average: {:.2}",
        results.last().unwrap_or(&0.0)
    );

    Ok(())
}

// Example demonstrating stream aggregation
#[cfg(feature = "streaming")]
#[allow(clippy::result_large_err)]
fn stream_aggregation_example() -> Result<()> {
    let headers = vec![
        "timestamp".to_string(),
        "value".to_string(),
        "quantity".to_string(),
    ];

    let config = StreamConfig {
        buffer_size: 1000,
        window_size: None,
        window_duration: None,
        processing_interval: Duration::from_millis(10),
        batch_size: 50,
    };

    let stream = DataStream::new(headers.clone(), Some(config));

    // Create an aggregator
    let mut aggregator = StreamAggregator::new(stream);

    // Add aggregation functions
    aggregator.add_aggregator("value", AggregationType::Average)?;
    aggregator.add_aggregator("value", AggregationType::Max)?;
    aggregator.add_aggregator("quantity", AggregationType::Sum)?;

    // Simulate data in a separate thread
    let sender = aggregator.stream.get_sender().unwrap();
    thread::spawn(move || {
        for i in 0..300 {
            let fields = HashMap::from([
                ("timestamp".to_string(), format!("{}", i)),
                ("value".to_string(), format!("{}", (i % 100) as f64 / 10.0)),
                ("quantity".to_string(), format!("{}", i % 10 + 1)),
            ]);

            let record = StreamRecord::new(fields);
            let _ = sender.send(record);

            thread::sleep(Duration::from_millis(1));
        }

        // Close the stream
        drop(sender);
    });

    // Process and get aggregates
    let aggregates = aggregator.process()?;

    println!("Aggregation results:");
    for (column, value) in aggregates {
        println!("  {}: {:.2}", column, value);
    }

    Ok(())
}

// Example demonstrating real-time analytics
#[cfg(feature = "streaming")]
#[allow(clippy::result_large_err)]
fn realtime_analytics_example() -> Result<()> {
    let headers = vec![
        "timestamp".to_string(),
        "temperature".to_string(),
        "humidity".to_string(),
    ];

    let stream = DataStream::new(headers.clone(), None);

    // Create real-time analytics
    let mut analytics = RealTimeAnalytics::new(
        stream,
        20,                        // Window size
        Duration::from_millis(50), // Computing interval
    );

    // Add metrics to compute
    analytics.add_metric("avg", "temperature", MetricType::WindowAverage)?;
    analytics.add_metric("change", "temperature", MetricType::RateOfChange)?;
    analytics.add_metric(
        "ema",
        "temperature",
        MetricType::ExponentialMovingAverage(0.3),
    )?;
    analytics.add_metric("std", "temperature", MetricType::StandardDeviation)?;
    analytics.add_metric("p90", "temperature", MetricType::Percentile(0.9))?;

    // Start background processing
    let metrics = analytics.start_background_processing()?;

    // Simulate data in a separate thread
    let sender = analytics.stream.get_sender().unwrap();
    thread::spawn(move || {
        for i in 0..100 {
            // Simulate temperatures with some noise
            let base_temp = 20.0 + (i as f64 / 10.0).sin() * 5.0;
            // Using the rand crate for random numbers
            let noise = rand::random::<f64>() * 2.0 - 1.0;
            let temp = base_temp + noise;

            let fields = HashMap::from([
                ("timestamp".to_string(), format!("{}", i)),
                ("temperature".to_string(), format!("{:.1}", temp)),
                ("humidity".to_string(), format!("{:.1}", 60.0 + noise * 5.0)),
            ]);

            let record = StreamRecord::new(fields);
            let _ = sender.send(record);

            thread::sleep(Duration::from_millis(20));
        }

        // Close the stream
        drop(sender);
    });

    // Monitor metrics
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(2500) {
        let current = metrics.lock().unwrap().clone();

        if !current.is_empty() {
            println!("Real-time metrics:");
            for (name, value) in current {
                println!("  {}: {:.2}", name, value);
            }
            println!();
        }

        thread::sleep(Duration::from_millis(500));
    }

    // Stop processing
    analytics.stop();

    Ok(())
}

// Example demonstrating a custom stream connector
#[cfg(feature = "streaming")]
#[allow(clippy::result_large_err)]
fn custom_connector_example() -> Result<()> {
    let headers = vec![
        "timestamp".to_string(),
        "event".to_string(),
        "source".to_string(),
    ];

    // Create a stream connector and stream
    let (connector, stream) = StreamConnector::new(headers, None);

    // Create a stream processor
    let mut processor = StreamProcessor::new(stream);

    // Add transformations
    processor.add_transformer("event", |value| {
        // Convert to uppercase
        Ok(value.to_uppercase())
    })?;

    // Add filter for events
    processor.set_filter(|record| {
        // Filter out events not from "system" source
        if let Some(source) = record.fields.get("source") {
            source == "system"
        } else {
            false
        }
    });

    // Simulate data in a separate thread
    thread::spawn(move || {
        let sources = ["user", "system", "application"];
        let events = ["login", "logout", "error", "warning", "info"];

        for i in 0..100 {
            let source = sources[i % sources.len()];
            let event = events[i % events.len()];

            let fields = HashMap::from([
                ("timestamp".to_string(), format!("{}", i)),
                ("event".to_string(), event.to_string()),
                ("source".to_string(), source.to_string()),
            ]);

            let _ = connector.send_fields(fields);

            thread::sleep(Duration::from_millis(10));
        }

        // Close the stream
        drop(connector);
    });

    // Process the stream
    let results = processor.process()?;

    println!("Processed {} batches", results.len());

    // Print summary of processed data
    let mut event_counts = HashMap::new();

    for df in &results {
        for row_idx in 0..df.row_count() {
            if let Ok(event) = df.get_string_value("event", row_idx) {
                *event_counts.entry(event.to_string()).or_insert(0) += 1;
            }
        }
    }

    println!("Event counts (from 'system' source only):");
    for (event, count) in event_counts {
        println!("  {}: {}", event, count);
    }

    Ok(())
}
