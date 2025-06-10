//! # Checkpointing for Fault Recovery
//!
//! This module provides checkpointing mechanisms for fault recovery in distributed processing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::distributed::execution::ExecutionPlan;
use crate::distributed::partition::PartitionSet;
use crate::error::{Error, Result};

/// Checkpoint for fault recovery
#[derive(Debug, Clone)]
pub struct ExecutionCheckpoint {
    /// Execution plan
    pub plan: ExecutionPlan,
    /// Partial result (if available)
    pub partial_result: Option<PartitionSet>,
    /// Failed partitions
    pub failed_partitions: Vec<usize>,
    /// Checkpoint time
    pub checkpoint_time: Instant,
}

impl ExecutionCheckpoint {
    /// Creates a new execution checkpoint
    pub fn new(plan: ExecutionPlan) -> Self {
        Self {
            plan,
            partial_result: None,
            failed_partitions: Vec::new(),
            checkpoint_time: Instant::now(),
        }
    }

    /// Sets the partial result
    pub fn with_partial_result(mut self, partial_result: PartitionSet) -> Self {
        self.partial_result = Some(partial_result);
        self
    }

    /// Adds a failed partition
    pub fn add_failed_partition(&mut self, partition_id: usize) {
        self.failed_partitions.push(partition_id);
    }
}

/// Manager for execution checkpoints
pub struct CheckpointManager {
    /// Checkpoints indexed by execution ID
    checkpoints: Arc<Mutex<HashMap<String, ExecutionCheckpoint>>>,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager
    pub fn new() -> Self {
        Self {
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Creates a checkpoint for an execution
    pub fn create_checkpoint(
        &self,
        execution_id: impl Into<String>,
        plan: ExecutionPlan,
    ) -> Result<()> {
        let checkpoint = ExecutionCheckpoint::new(plan);

        match self.checkpoints.lock() {
            Ok(mut checkpoints) => {
                checkpoints.insert(execution_id.into(), checkpoint);
                Ok(())
            }
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to create checkpoint".to_string(),
            )),
        }
    }

    /// Gets a checkpoint for an execution
    pub fn get_checkpoint(&self, execution_id: &str) -> Result<Option<ExecutionCheckpoint>> {
        match self.checkpoints.lock() {
            Ok(checkpoints) => Ok(checkpoints.get(execution_id).cloned()),
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to get checkpoint".to_string(),
            )),
        }
    }

    /// Updates a checkpoint for an execution
    pub fn update_checkpoint(
        &self,
        execution_id: &str,
        partial_result: Option<PartitionSet>,
        failed_partitions: Option<Vec<usize>>,
    ) -> Result<()> {
        match self.checkpoints.lock() {
            Ok(mut checkpoints) => {
                if let Some(checkpoint) = checkpoints.get_mut(execution_id) {
                    if let Some(result) = partial_result {
                        checkpoint.partial_result = Some(result);
                    }

                    if let Some(partitions) = failed_partitions {
                        checkpoint.failed_partitions = partitions;
                    }

                    Ok(())
                } else {
                    Err(Error::DistributedProcessing(format!(
                        "Checkpoint not found for execution ID: {}",
                        execution_id
                    )))
                }
            }
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to update checkpoint".to_string(),
            )),
        }
    }

    /// Removes a checkpoint for an execution
    pub fn remove_checkpoint(&self, execution_id: &str) -> Result<()> {
        match self.checkpoints.lock() {
            Ok(mut checkpoints) => {
                checkpoints.remove(execution_id);
                Ok(())
            }
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to remove checkpoint".to_string(),
            )),
        }
    }
}
