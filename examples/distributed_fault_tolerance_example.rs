#[cfg(feature = "distributed")]
use pandrs::distributed::config::DistributedConfig;
#[cfg(feature = "distributed")]
use pandrs::distributed::datafusion::{DataFusionContext, FaultTolerantDataFusionContext};
#[cfg(feature = "distributed")]
use pandrs::distributed::execution::{ExecutionPlan, Operation};
#[cfg(feature = "distributed")]
use pandrs::distributed::fault_tolerance::{FaultToleranceHandler, RecoveryStrategy, RetryPolicy};
#[cfg(feature = "distributed")]
use pandrs::error::Result;

/// Example showing fault tolerance in action with retries
#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Configure the distributed processing
    let config = DistributedConfig::new()
        .with_concurrency(4)
        .with_memory_limit(Some(1024 * 1024 * 1024)) // 1GB
        .with_optimization(true);

    // Create a basic context
    let mut context = DataFusionContext::new(config);

    // Register test data
    register_test_data(&mut context)?;

    // Create a fault tolerance handler with exponential backoff
    let fault_handler = FaultToleranceHandler::new(
        RetryPolicy::Exponential {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 1000,
            backoff_factor: 2.0,
        },
        RecoveryStrategy::RetryFailedPartitions,
    );

    // Create a fault-tolerant context
    let ft_context = FaultTolerantDataFusionContext::new(context, fault_handler);

    // Create a plan with a simple filter
    let plan = ExecutionPlan::new(
        Operation::Filter {
            predicate: "id > 10".to_string(),
        },
        vec!["test_data".to_string()],
        "filtered_result".to_string(),
    );

    // Execute the plan with fault tolerance
    let result = ft_context.execute(&plan)?;

    // Get the execution metrics
    println!("Execution Result Metrics:\n{}", result.metrics().format());

    // Display results
    let df = result.collect_to_local()?;
    println!("\nResults:\n{:?}", df);

    // Run a fault-tolerant SQL query
    println!("\nRunning SQL Query with Fault Tolerance:");
    let sql_result = ft_context.sql("SELECT * FROM test_data WHERE value > 50")?;

    println!("SQL Execution Metrics:\n{}", sql_result.metrics().format());

    let sql_df = sql_result.collect_to_local()?;
    println!("\nSQL Results:\n{:?}", sql_df);

    // Demonstrate fault recovery by simulating failures
    println!("\nDemonstrating fault recovery with simulated failures:");
    let mut failure_simulation = FailureSimulation::new(ft_context);
    failure_simulation.run_with_simulated_failures()?;

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_fault_tolerance_example --features distributed");
}

/// Register test data for the example
#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn register_test_data(context: &mut DataFusionContext) -> Result<()> {
    // Create a simple dataframe with test data
    let mut df = pandrs::dataframe::DataFrame::new();

    // Add some columns
    df.add_column(
        "id".to_string(),
        pandrs::series::Series::new((1..=100).collect::<Vec<i64>>(), Some("id".to_string()))?,
    )?;

    df.add_column(
        "value".to_string(),
        pandrs::series::Series::new(
            (1..=100).map(|x| x as f64 * 1.5).collect::<Vec<f64>>(),
            Some("value".to_string()),
        )?,
    )?;

    // Convert to partitions
    #[cfg(feature = "distributed")]
    {
        use pandrs::distributed::datafusion::conversion::dataframe_to_record_batches;
        use pandrs::distributed::partition::{Partition, PartitionSet};

        let batches = dataframe_to_record_batches(&df, 25)?;
        let mut partitions = Vec::new();

        for (i, batch) in batches.iter().enumerate() {
            let partition = Partition::new(i, batch.clone());
            partitions.push(Arc::new(partition));
        }

        let partition_set = PartitionSet::new(partitions, batches[0].schema());

        // Register the data
        context.register_dataset("test_data", partition_set)?;
    }

    Ok(())
}

/// Simulation of failures for demonstration purposes
#[cfg(feature = "distributed")]
struct FailureSimulation {
    context: FaultTolerantDataFusionContext,
    failure_count: usize,
}

#[cfg(feature = "distributed")]
impl FailureSimulation {
    /// Create a new failure simulation
    fn new(context: FaultTolerantDataFusionContext) -> Self {
        Self {
            context,
            failure_count: 0,
        }
    }

    /// Run an example with simulated failures
    #[allow(clippy::result_large_err)]
    fn run_with_simulated_failures(&mut self) -> Result<()> {
        // Set up handler with visible reporting
        let new_handler = FaultToleranceHandler::new(
            RetryPolicy::Fixed {
                max_retries: 2,
                delay_ms: 500,
            },
            RecoveryStrategy::RetryQuery,
        );

        let ft_context =
            FaultTolerantDataFusionContext::new(self.context.inner().clone(), new_handler);

        // First try - will fail with our simulated error
        println!("Attempt 1: Expect failure and retry");
        let result = match self.simulate_query_with_failures(
            &ft_context,
            "SELECT * FROM test_data WHERE id > 50 AND id < 70",
        ) {
            Ok(r) => r,
            Err(e) => {
                println!("Query failed after all retries: {}", e);
                return Ok(());
            }
        };

        println!("Query eventually succeeded after failures!");
        println!("Results: {} rows", result.collect_to_local()?.shape()?.0);

        // Show the failures that were recorded
        let failures = ft_context.fault_handler().recent_failures()?;
        println!("\nRecorded Failures:");
        for (i, failure) in failures.iter().enumerate() {
            println!(
                "Failure {}: Type: {:?}, Message: {}, Recovered: {}, Retries: {}",
                i + 1,
                failure.failure_type,
                failure.error_message,
                failure.recovered,
                failure.retry_attempts
            );
        }

        Ok(())
    }

    /// Simulate a query with failures injected
    fn simulate_query_with_failures(
        &mut self,
        context: &FaultTolerantDataFusionContext,
        query: &str,
    ) -> Result<pandrs::distributed::execution::ExecutionResult> {
        // Track simulated failures
        self.failure_count += 1;

        // Inject failure for the first two attempts
        if self.failure_count <= 2 {
            println!("Simulating failure #{}", self.failure_count);

            // Return different errors based on the failure count
            if self.failure_count == 1 {
                return Err(pandrs::error::Error::Timeout(
                    "Simulated timeout error for testing fault tolerance".to_string(),
                ));
            } else {
                return Err(pandrs::error::Error::IoError(
                    "Simulated network error for testing fault tolerance".to_string(),
                ));
            }
        }

        // After simulated failures, return success
        println!("Execution succeeding on attempt #{}", self.failure_count);
        context.sql(query)
    }
}
