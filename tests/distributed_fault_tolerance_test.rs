#[cfg(feature = "distributed")]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use pandrs::distributed::config::DistributedConfig;
    use pandrs::distributed::datafusion::{DataFusionContext, FaultTolerantDataFusionContext};
    use pandrs::distributed::execution::{ExecutionPlan, Operation};
    use pandrs::distributed::fault_tolerance::{
        FailureInfo, FailureType, FaultToleranceHandler, RecoveryStrategy, RetryPolicy,
    };
    use pandrs::error::{Error, Result};

    // Create a test context with sample data
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn create_test_context() -> Result<DataFusionContext> {
        let config = DistributedConfig::new().with_concurrency(2);
        let mut context = DataFusionContext::new(config);

        // Create a simple dataframe with test data
        let mut df = pandrs::dataframe::DataFrame::new();

        df.add_column(
            "id".to_string(),
            pandrs::series::Series::new((1..=20).collect::<Vec<i64>>(), Some("id".to_string()))?,
        )?;

        df.add_column(
            "value".to_string(),
            pandrs::series::Series::new(
                (1..=20).map(|x| x as f64 * 2.0).collect::<Vec<f64>>(),
                Some("value".to_string()),
            )?,
        )?;

        // Convert to partitions and register
        use pandrs::distributed::datafusion::conversion::dataframe_to_record_batches;
        use pandrs::distributed::partition::{Partition, PartitionSet};

        let batches = dataframe_to_record_batches(&df, 5)?;
        let mut partitions = Vec::new();

        for (i, batch) in batches.iter().enumerate() {
            let partition = Partition::new(i, batch.clone());
            partitions.push(Arc::new(partition));
        }

        let partition_set = PartitionSet::new(partitions, batches[0].schema());

        context.register_dataset("test_data", partition_set)?;

        Ok(context)
    }

    // Test the RetryPolicy implementation
    #[test]
    fn test_retry_policy() {
        // Fixed retry policy
        let fixed = RetryPolicy::Fixed {
            max_retries: 3,
            delay_ms: 100,
        };

        assert_eq!(fixed.max_retries(), 3);
        assert_eq!(fixed.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(fixed.delay_for_attempt(2), Duration::from_millis(100));

        // Exponential retry policy
        let exp = RetryPolicy::Exponential {
            max_retries: 5,
            initial_delay_ms: 100,
            max_delay_ms: 1000,
            backoff_factor: 2.0,
        };

        assert_eq!(exp.max_retries(), 5);
        assert_eq!(exp.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(exp.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(exp.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(exp.delay_for_attempt(3), Duration::from_millis(800));
        // Should be capped at max_delay_ms
        assert_eq!(exp.delay_for_attempt(4), Duration::from_millis(1000));
    }

    // Test the FailureType implementation
    #[test]
    fn test_failure_type() {
        assert!(FailureType::Network.is_retriable());
        assert!(FailureType::Timeout.is_retriable());
        assert!(!FailureType::Data.is_retriable());
        assert!(!FailureType::Memory.is_retriable());

        // Test error conversion
        let network_err = Error::IoError("network error".to_string());
        assert_eq!(FailureType::from_error(&network_err), FailureType::Network);

        let timeout_err = Error::Timeout("timeout".to_string());
        assert_eq!(FailureType::from_error(&timeout_err), FailureType::Timeout);
    }

    // Test the FailureInfo implementation
    #[test]
    fn test_failure_info() {
        let mut failure = FailureInfo::new(FailureType::Network, "connection refused");
        assert_eq!(failure.failure_type, FailureType::Network);
        assert_eq!(failure.error_message, "connection refused");
        assert_eq!(failure.node_id, None);
        assert_eq!(failure.recovered, false);
        assert_eq!(failure.retry_attempts, 0);

        // Test with node ID
        let failure_with_node = failure.with_node_id("node1");
        assert_eq!(failure_with_node.node_id, Some("node1".to_string()));

        // Test state changes
        failure.increment_retry();
        assert_eq!(failure.retry_attempts, 1);

        failure.mark_recovered();
        assert_eq!(failure.recovered, true);
    }

    // Test basic execution with fault tolerance
    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_basic_fault_tolerance() -> Result<()> {
        let context = create_test_context()?;

        // Create fault handler
        let handler = FaultToleranceHandler::new(
            RetryPolicy::Fixed {
                max_retries: 2,
                delay_ms: 10, // Short delay for tests
            },
            RecoveryStrategy::RetryQuery,
        );

        // Create fault-tolerant context
        let ft_context = FaultTolerantDataFusionContext::new(context, handler);

        // Execute a simple plan
        let plan = ExecutionPlan::new(
            Operation::Filter {
                predicate: "id > 10".to_string(),
            },
            ["test_data".to_string()],
            "filtered_result".to_string(),
        );

        let result = ft_context.execute(&plan)?;

        // Verify result
        let df = result.collect_to_local()?;
        assert_eq!(df.shape()?.0, 10); // Should have 10 rows (ids 11-20)

        // Test SQL execution
        let sql_result = ft_context.sql("SELECT * FROM test_data WHERE id <= 5")?;
        let sql_df = sql_result.collect_to_local()?;
        assert_eq!(sql_df.shape()?.0, 5); // Should have 5 rows (ids 1-5)

        Ok(())
    }

    // Test the recovery from failures
    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_recovery_from_failure() -> Result<()> {
        let context = create_test_context()?;

        // Create handler with fast retries for testing
        let handler = FaultToleranceHandler::new(
            RetryPolicy::Fixed {
                max_retries: 3,
                delay_ms: 1,
            },
            RecoveryStrategy::RetryQuery,
        );

        // We'll set up a test that fails a few times then succeeds
        struct TestWithFailures {
            attempts: usize,
        }

        impl TestWithFailures {
            fn new() -> Self {
                Self { attempts: 0 }
            }

            #[allow(clippy::result_large_err)]
            #[allow(clippy::result_large_err)]
            fn execute(&mut self) -> Result<()> {
                self.attempts += 1;

                if self.attempts <= 2 {
                    // First two attempts fail
                    Err(Error::IoError(format!(
                        "Simulated failure, attempt {}",
                        self.attempts
                    )))
                } else {
                    // Third attempt succeeds
                    Ok(())
                }
            }
        }

        // Create test
        let mut test = TestWithFailures::new();

        // Execute with retry logic
        let result = handler.execute_with_retry(|| test.execute());

        // Should succeed on third attempt
        assert!(result.is_ok());
        assert_eq!(test.attempts, 3);

        // Verify recorded failures
        let failures = handler.recent_failures()?;
        assert_eq!(failures.len(), 2); // Should have recorded 2 failures

        // Clear failures
        handler.clear_failures()?;
        let failures_after_clear = handler.recent_failures()?;
        assert_eq!(failures_after_clear.len(), 0);

        Ok(())
    }

    // Test node health tracking
    #[test]
    #[allow(clippy::result_large_err)]
    #[allow(clippy::result_large_err)]
    fn test_node_health_tracking() -> Result<()> {
        let handler = FaultToleranceHandler::default();

        // Initially no health info
        let health = handler.node_health()?;
        assert_eq!(health.len(), 0);

        // Update health
        handler.update_node_health("node1", true)?;
        handler.update_node_health("node2", false)?;

        // Check health
        let updated_health = handler.node_health()?;
        assert_eq!(updated_health.len(), 2);
        assert_eq!(updated_health.get("node1"), Some(&true));
        assert_eq!(updated_health.get("node2"), Some(&false));

        Ok(())
    }
}
