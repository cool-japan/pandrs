//! # Fault Recovery Mechanisms
//!
//! This module provides mechanisms for recovering from failures in distributed processing.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use super::core::{FailureInfo, FailureType, RecoveryStrategy, RetryPolicy};
use crate::distributed::execution::{ExecutionPlan, ExecutionResult};
use crate::error::{Error, Result};

/// Action to take for recovery
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the entire query
    RetryQuery {
        /// Execution plan to retry
        plan: ExecutionPlan,
        /// Delay before retry
        delay: Duration,
    },
    /// Retry only the failed partitions
    RetryFailedPartitions {
        /// Execution plan to retry
        plan: ExecutionPlan,
        /// Partition IDs that failed
        partition_ids: Vec<usize>,
        /// Delay before retry
        delay: Duration,
    },
    /// Reroute the query to different nodes
    Reroute {
        /// Execution plan to reroute
        plan: ExecutionPlan,
        /// Nodes to exclude from execution
        excluded_nodes: Vec<String>,
    },
    /// Fallback to a local execution engine
    LocalFallback {
        /// Execution plan to execute locally
        plan: ExecutionPlan,
    },
}

/// Handler for fault detection and recovery
pub struct FaultToleranceHandler {
    /// Retry policy
    retry_policy: RetryPolicy,
    /// Recovery strategy
    recovery_strategy: RecoveryStrategy,
    /// Recent failures
    recent_failures: Arc<RwLock<Vec<FailureInfo>>>,
    /// Node health tracking
    node_health: Arc<RwLock<HashMap<String, bool>>>,
    /// Partition recovery tracking
    partition_recovery: Arc<RwLock<HashMap<usize, Vec<FailureInfo>>>>,
}

impl FaultToleranceHandler {
    /// Creates a new fault tolerance handler
    pub fn new(retry_policy: RetryPolicy, recovery_strategy: RecoveryStrategy) -> Self {
        Self {
            retry_policy,
            recovery_strategy,
            recent_failures: Arc::new(RwLock::new(Vec::new())),
            node_health: Arc::new(RwLock::new(HashMap::new())),
            partition_recovery: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates a new fault tolerance handler with default settings
    pub fn default() -> Self {
        Self::new(
            RetryPolicy::default_exponential(),
            RecoveryStrategy::RetryFailedPartitions,
        )
    }

    /// Executes an operation with fault tolerance
    pub fn execute_with_retry<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        let mut attempt = 0;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let failure_type = FailureType::from_error(&error);

                    // Record the failure
                    let mut failure = FailureInfo::new(failure_type.clone(), error.to_string());

                    if let Ok(mut failures) = self.recent_failures.write() {
                        failures.push(failure.clone());
                    }

                    // Check if we should retry
                    if !failure_type.is_retriable() || attempt >= self.retry_policy.max_retries() {
                        return Err(error);
                    }

                    // Increment attempt and wait before retrying
                    attempt += 1;
                    failure.increment_retry();

                    // Sleep for the appropriate delay
                    std::thread::sleep(self.retry_policy.delay_for_attempt(attempt));
                }
            }
        }
    }

    /// Handles a failure during execution
    pub fn handle_failure(
        &self,
        plan: &ExecutionPlan,
        error: &Error,
    ) -> Result<Option<RecoveryAction>> {
        let failure_type = FailureType::from_error(error);

        if !failure_type.is_retriable() {
            return Err(Error::DistributedProcessing(format!(
                "Non-retriable error: {}",
                error
            )));
        }

        // Record the failure
        let mut failure = FailureInfo::new(failure_type, error.to_string());

        if let Ok(mut failures) = self.recent_failures.write() {
            failures.push(failure.clone());
        }

        // Determine recovery action based on strategy
        let action = match self.recovery_strategy {
            RecoveryStrategy::RetryQuery => Some(RecoveryAction::RetryQuery {
                plan: plan.clone(),
                delay: self.retry_policy.delay_for_attempt(0),
            }),
            RecoveryStrategy::RetryFailedPartitions => {
                Some(RecoveryAction::RetryFailedPartitions {
                    plan: plan.clone(),
                    partition_ids: vec![], // Would come from actual failure details
                    delay: self.retry_policy.delay_for_attempt(0),
                })
            }
            RecoveryStrategy::Reroute => {
                Some(RecoveryAction::Reroute {
                    plan: plan.clone(),
                    excluded_nodes: vec![], // Would come from node health tracking
                })
            }
            RecoveryStrategy::LocalFallback => {
                Some(RecoveryAction::LocalFallback { plan: plan.clone() })
            }
        };

        Ok(action)
    }

    /// Gets recent failures
    pub fn recent_failures(&self) -> Result<Vec<FailureInfo>> {
        match self.recent_failures.read() {
            Ok(failures) => Ok(failures.clone()),
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to read recent failures".to_string(),
            )),
        }
    }

    /// Gets node health
    pub fn node_health(&self) -> Result<HashMap<String, bool>> {
        match self.node_health.read() {
            Ok(health) => Ok(health.clone()),
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to read node health".to_string(),
            )),
        }
    }

    /// Updates the health status of a node
    pub fn update_node_health(&self, node_id: impl Into<String>, healthy: bool) -> Result<()> {
        match self.node_health.write() {
            Ok(mut health) => {
                health.insert(node_id.into(), healthy);
                Ok(())
            }
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to update node health".to_string(),
            )),
        }
    }

    /// Clears all recorded failures
    pub fn clear_failures(&self) -> Result<()> {
        match self.recent_failures.write() {
            Ok(mut failures) => {
                failures.clear();
                Ok(())
            }
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to clear failures".to_string(),
            )),
        }
    }
}

/// Executor that provides fault tolerance
pub struct FaultTolerantExecutor<E> {
    /// Inner executor
    inner: E,
    /// Fault tolerance handler
    fault_handler: Arc<FaultToleranceHandler>,
}

impl<E> FaultTolerantExecutor<E> {
    /// Creates a new fault tolerant executor
    pub fn new(inner: E, fault_handler: FaultToleranceHandler) -> Self {
        Self {
            inner,
            fault_handler: Arc::new(fault_handler),
        }
    }

    /// Gets the fault tolerance handler
    pub fn fault_handler(&self) -> Arc<FaultToleranceHandler> {
        self.fault_handler.clone()
    }
}

/// Extension trait for execution contexts to add fault tolerance
pub trait FaultTolerantContext {
    /// Executes a plan with fault tolerance
    fn execute_with_fault_tolerance(
        &self,
        plan: &ExecutionPlan,
        fault_handler: &FaultToleranceHandler,
    ) -> Result<ExecutionResult>;

    /// Recovers from a failure
    fn recover_from_failure(&self, action: RecoveryAction) -> Result<ExecutionResult>;
}
