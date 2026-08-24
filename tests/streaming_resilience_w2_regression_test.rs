//! Regression tests for the Wave-2 resilience/streaming fixes.
//!
//! Covers, per the audit fix list for `src/config/resilience.rs`,
//! `src/streaming/{mod,window,backpressure}.rs`, and `src/io/streaming.rs`:
//! - A full CSV streamed with an idle gap does not lose data (previously,
//!   `DataStream::process`/`window_operation` treated an idle gap longer
//!   than `processing_interval` as end-of-stream and silently discarded
//!   whatever the producer sent afterward).
//! - A count-based window completes exactly at its configured size.
//! - Variance is never negative, even for a large-mean/small-spread
//!   dataset that defeats the naive `E[x^2] - mean^2` formula.
//! - `BackpressureBuffer::try_push`'s `PushOutcome` distinguishes
//!   `Enqueued`/`Dropped`/`WouldBlock` instead of collapsing them into a
//!   bare `bool`.
//! - The resilience retry mechanism actually retries a transiently-failing
//!   operation instead of performing exactly one attempt regardless of
//!   configuration.

use pandrs::streaming::backpressure::{
    BackpressureConfigBuilder, BackpressureStrategy, PushOutcome,
};
use pandrs::streaming::window::{WindowAggregation, WindowConfigBuilder, WindowedAggregator};
use pandrs::streaming::{
    BackpressureBuffer, DataStream, StreamConfig, StreamConnector, StreamRecord,
};
use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

/// Writes `lines` (already including a header line as `lines[0]`) to a
/// fresh temp file and returns its path, per project convention of using
/// `std::env::temp_dir()` rather than a hardcoded path.
fn write_temp_csv(name: &str, lines: &[String]) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "pandrs_streaming_w2_regression_{name}_{}_{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    let mut file = std::fs::File::create(&path).expect("failed to create temp CSV file");
    for line in lines {
        writeln!(file, "{line}").expect("failed to write temp CSV file");
    }
    path
}

/// A CSV streamed with an idle gap (simulated via `delay_ms`) longer than
/// `processing_interval` must not lose any records. Before the fix,
/// `DataStream::process` treated `receiver.is_empty()` after any single
/// `recv_timeout` timeout as end-of-stream, so the first idle gap silently
/// discarded the rest of the file.
#[test]
fn test_csv_stream_with_idle_gap_does_not_lose_data() {
    let mut lines = vec!["id,value".to_string()];
    for i in 0..30 {
        lines.push(format!("{i},{}", i * 10));
    }
    let path = write_temp_csv("idle_gap", &lines);

    // processing_interval is much shorter than the deliberate per-record
    // delay injected below, so `process` will see several
    // `RecvTimeoutError::Timeout`s (idle gaps) in the middle of the stream.
    let config = StreamConfig {
        buffer_size: 1000,
        window_size: None,
        window_duration: None,
        processing_interval: Duration::from_millis(5),
        batch_size: 4,
    };

    // Delay every record by slightly more than `processing_interval` so
    // idle gaps are essentially guaranteed between (most) records.
    let mut stream = DataStream::read_from_csv(&path, Some(config), Some(8))
        .expect("failed to open temp CSV as a stream");

    let mut total_records = 0usize;
    let results = stream
        .process(
            |batch| {
                total_records += batch.len();
                Ok(batch.len())
            },
            None,
        )
        .expect("stream processing failed");

    let _ = std::fs::remove_file(&path);

    assert_eq!(
        total_records, 30,
        "all 30 records must be processed despite idle gaps between them, got {total_records}"
    );
    assert_eq!(
        results.iter().sum::<usize>(),
        30,
        "batch-size results must also sum to all 30 records"
    );
}

/// The same idle-gap scenario, but for `window_operation` rather than
/// `process`, since it has its own independent receive loop.
#[test]
fn test_csv_stream_with_idle_gap_window_operation_does_not_lose_data() {
    let mut lines = vec!["id,value".to_string()];
    for i in 0..12 {
        lines.push(format!("{i},{}", i * 10));
    }
    let path = write_temp_csv("idle_gap_window", &lines);

    let config = StreamConfig {
        buffer_size: 1000,
        window_size: Some(4),
        window_duration: None,
        processing_interval: Duration::from_millis(5),
        batch_size: 4,
    };

    let mut stream = DataStream::read_from_csv(&path, Some(config), Some(8))
        .expect("failed to open temp CSV as a stream");

    let mut total_records_seen = 0usize;
    let results = stream
        .window_operation(|window| {
            total_records_seen += window.len();
            Ok(window.len())
        })
        .expect("window processing failed");

    let _ = std::fs::remove_file(&path);

    assert!(!results.is_empty(), "expected at least one window to fire");
    // Count-based (window_size=4) tumbling windows over 12 records: 3
    // complete windows of 4, with no records lost to the idle gaps.
    assert_eq!(
        results.iter().sum::<usize>(),
        12,
        "all 12 records must be accounted for across emitted windows, got windows {results:?}"
    );
}

/// A count-based window must complete exactly when it reaches its
/// configured size, with the correct aggregate value -- not, as the
/// previous fabricated-time-extent implementation could produce, on a
/// bogus per-second cadence unrelated to the actual record count.
#[test]
fn test_count_window_completes_at_configured_size() {
    let config = WindowConfigBuilder::new()
        .count(5, None)
        .build()
        .expect("valid window config");

    let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

    let mut fields = HashMap::new();
    for i in 1..5 {
        fields.insert("value".to_string(), i.to_string());
        let record = StreamRecord::new(fields.clone());
        let results = agg.process(&record).expect("processing should succeed");
        assert!(
            results.is_empty(),
            "window must not complete before reaching its configured size (record {i})"
        );
    }

    fields.insert("value".to_string(), "5".to_string());
    let record = StreamRecord::new(fields);
    let results = agg.process(&record).expect("processing should succeed");

    assert_eq!(
        results.len(),
        1,
        "window must complete exactly once it reaches its configured count"
    );
    assert_eq!(results[0].count, 5);
    assert_eq!(results[0].values["value"], 1.0 + 2.0 + 3.0 + 4.0 + 5.0);
}

/// Variance must never be negative, including for data whose mean is large
/// relative to its spread -- the case that made the previous
/// `E[x^2] - mean^2` formula cancel catastrophically and go negative
/// (`StdDev` happened to mask this with a `.max(0.0)` floor; `Variance`
/// itself did not).
#[test]
fn test_variance_never_negative_for_large_mean_small_spread() {
    let config = WindowConfigBuilder::new()
        .global()
        .include_partial_windows(true)
        .build()
        .expect("valid window config");

    let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Variance);

    let mut fields = HashMap::new();
    for v in [1.0e9, 1.0e9 + 1.0, 1.0e9 + 2.0, 1.0e9 + 1.0] {
        fields.insert("value".to_string(), v.to_string());
        agg.process(&StreamRecord::new(fields.clone()))
            .expect("processing should succeed");
    }

    let results = agg.flush();
    assert_eq!(results.len(), 1);
    let variance = results[0].values["value"];
    assert!(
        variance >= 0.0,
        "variance must never be negative regardless of the data's magnitude, got {variance}"
    );
}

/// `try_push` must distinguish "enqueued" from "dropped" from "would block"
/// rather than collapsing all three into a bare `bool` that a caller could
/// not meaningfully act on.
#[test]
fn test_backpressure_push_outcome_semantics() {
    let config = BackpressureConfigBuilder::new()
        .high_watermark(2)
        .low_watermark(1)
        .strategy(BackpressureStrategy::DropNewest)
        .build();
    let buffer = BackpressureBuffer::new(config);

    let record = |v: &str| {
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), v.to_string());
        StreamRecord::new(fields)
    };

    assert_eq!(
        buffer.try_push(record("1")).expect("push should succeed"),
        PushOutcome::Enqueued
    );
    assert_eq!(
        buffer.try_push(record("2")).expect("push should succeed"),
        PushOutcome::Enqueued
    );
    // Buffer is now at the high watermark; DropNewest declines further pushes.
    assert_eq!(
        buffer.try_push(record("3")).expect("push should succeed"),
        PushOutcome::Dropped
    );
    assert!(!PushOutcome::Dropped.is_enqueued());
    assert!(PushOutcome::Enqueued.is_enqueued());

    let block_config = BackpressureConfigBuilder::new()
        .high_watermark(1)
        .low_watermark(0)
        .strategy(BackpressureStrategy::Block)
        .build();
    let block_buffer = BackpressureBuffer::new(block_config);
    assert_eq!(
        block_buffer
            .try_push(record("a"))
            .expect("push should succeed"),
        PushOutcome::Enqueued
    );
    // try_push (unlike the blocking `push`) never retries; over the high
    // watermark with `Block` strategy it reports `WouldBlock` directly so
    // the caller can decide what to do (rather than a `false` that looked
    // identical to a strategy-driven drop).
    assert_eq!(
        block_buffer
            .try_push(record("b"))
            .expect("push should succeed"),
        PushOutcome::WouldBlock
    );
}

/// `StreamConnector::send_fields` must reject a field name that isn't one of
/// the stream's declared headers, rather than silently accepting it. Before
/// this check existed, such a record would be sent successfully but the
/// unrecognized field would vanish without any error the first time it went
/// through `DataStream::batch_to_dataframe`, which only ever looks up
/// `headers` in each record's field map -- an unrecognized key is simply
/// never read.
#[test]
fn test_stream_connector_rejects_field_not_in_declared_headers() {
    let headers = vec!["id".to_string(), "value".to_string()];
    let (connector, _stream) = StreamConnector::new(headers, None);

    let mut fields = HashMap::new();
    fields.insert("id".to_string(), "1".to_string());
    fields.insert("valeu".to_string(), "10".to_string()); // typo'd column name
    let result = connector.send_fields(fields);

    assert!(
        result.is_err(),
        "a field name not in the declared headers must be rejected, not silently dropped"
    );
}

/// The mirror-image success case: every field name matching a declared
/// header must still be accepted (proving the new validation is a genuine
/// check, not an unconditional rejection).
#[test]
fn test_stream_connector_accepts_fields_matching_declared_headers() {
    let headers = vec!["id".to_string(), "value".to_string()];
    let (connector, _stream) = StreamConnector::new(headers, None);

    let mut fields = HashMap::new();
    fields.insert("id".to_string(), "1".to_string());
    fields.insert("value".to_string(), "10".to_string());

    assert!(
        connector.send_fields(fields).is_ok(),
        "fields whose names all match declared headers must be accepted"
    );
}

#[cfg(feature = "resilience")]
mod resilience_retry_regression {
    use pandrs::config::resilience::{BackoffStrategy, RetryConfig, RetryMechanism};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// The retry mechanism must actually retry: 2 simulated transient
    /// failures followed by a success must yield exactly 3 invocations of
    /// the operation closure. The previous `is_retryable` check compared
    /// `retryable_errors` entries (written in `PascalCase`, matching the
    /// `Error` enum's Rust variant identifiers) against the error's
    /// `Display` text (a free-form human sentence, e.g. `"Connection
    /// error: ..."` rather than `"ConnectionError..."`), so `starts_with`
    /// could never match and every operation failed after exactly one
    /// attempt regardless of `max_attempts`/`backoff_strategy`/`jitter`.
    #[tokio::test]
    async fn test_retry_mechanism_actually_retries() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 1,
            backoff_strategy: BackoffStrategy::Fixed,
            jitter: false,
            retryable_errors: vec!["transient".to_string()],
            ..Default::default()
        };
        let retry = RetryMechanism::new(config);

        let invocations = Arc::new(AtomicU32::new(0));
        let invocations_clone = invocations.clone();

        let result = retry
            .execute(move || {
                let n = invocations_clone.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(format!("transient failure #{n}"))
                } else {
                    Ok("success")
                }
            })
            .await;

        assert_eq!(result.expect("should eventually succeed"), "success");
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            3,
            "expected exactly 2 failed attempts followed by 1 successful attempt"
        );
    }

    /// A non-retryable error must fail after exactly one attempt, proving
    /// the retry decision is genuinely conditional on `retryable_errors`
    /// rather than either "always retries" or "never retries".
    #[tokio::test]
    async fn test_retry_mechanism_does_not_retry_unlisted_errors() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 1,
            retryable_errors: vec!["transient".to_string()],
            ..Default::default()
        };
        let retry = RetryMechanism::new(config);

        let invocations = Arc::new(AtomicU32::new(0));
        let invocations_clone = invocations.clone();

        let result: pandrs::error::Result<()> = retry
            .execute(move || {
                invocations_clone.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>("permanent failure".to_string())
            })
            .await;

        assert!(result.is_err());
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    /// The crate's own `Error` type is classified structurally (by variant),
    /// not by string-matching its `Display` message, and using it with
    /// `RetryMechanism` (via the default `retryable_errors`, which are the
    /// literal `Display` prefixes of pandrs's transient `Error` variants)
    /// genuinely retries.
    #[tokio::test]
    async fn test_retry_mechanism_retries_pandrs_error_with_default_config() {
        use pandrs::error::Error;

        let retry = RetryMechanism::new(RetryConfig::default());
        let invocations = Arc::new(AtomicU32::new(0));
        let invocations_clone = invocations.clone();

        let result = retry
            .execute(move || {
                let n = invocations_clone.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 2 {
                    Err(Error::ConnectionError("refused".to_string()))
                } else {
                    Ok(())
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
    }
}
