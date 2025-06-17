//! Model Deployment Module
//!
//! This module provides model deployment capabilities including deployment configuration,
//! resource management, scaling, and health monitoring.

use crate::core::error::{Error, Result};
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, DeploymentConfig, HealthCheckConfig,
    HealthStatus, ModelInfo, ModelMetadata, ModelServing, MonitoringConfig, PredictionRequest,
    PredictionResponse, ResourceConfig, ScalingConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Deployment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment is starting up
    Starting,
    /// Deployment is running and healthy
    Running,
    /// Deployment is unhealthy but still running
    Degraded,
    /// Deployment is stopping
    Stopping,
    /// Deployment has stopped
    Stopped,
    /// Deployment failed
    Failed,
}

/// Deployment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentMetrics {
    /// Current status
    pub status: DeploymentStatus,
    /// Number of active instances
    pub active_instances: usize,
    /// Current CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Current memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Current request rate (requests per second)
    pub request_rate: f64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Total number of requests served
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Last health check timestamp
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    /// Deployment start time
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Deployed model wrapper
pub struct DeployedModel {
    /// Underlying model
    model: Box<dyn ModelServing>,
    /// Deployment configuration
    config: DeploymentConfig,
    /// Deployment metrics
    metrics: Arc<Mutex<DeploymentMetrics>>,
    /// Request statistics
    stats: Arc<Mutex<RequestStats>>,
    /// Health check status
    health_status: Arc<Mutex<HealthStatus>>,
}

/// Request statistics
#[derive(Debug, Clone)]
struct RequestStats {
    /// Request timestamps for rate calculation
    request_times: Vec<Instant>,
    /// Response times in milliseconds
    response_times: Vec<u64>,
    /// Error count
    error_count: u64,
    /// Success count
    success_count: u64,
    /// Last cleanup time
    last_cleanup: Instant,
}

impl RequestStats {
    fn new() -> Self {
        Self {
            request_times: Vec::new(),
            response_times: Vec::new(),
            error_count: 0,
            success_count: 0,
            last_cleanup: Instant::now(),
        }
    }

    /// Record a successful request
    fn record_success(&mut self, response_time_ms: u64) {
        let now = Instant::now();
        self.request_times.push(now);
        self.response_times.push(response_time_ms);
        self.success_count += 1;

        // Cleanup old entries every 60 seconds
        if now.duration_since(self.last_cleanup) > Duration::from_secs(60) {
            self.cleanup_old_entries();
            self.last_cleanup = now;
        }
    }

    /// Record a failed request
    fn record_error(&mut self) {
        let now = Instant::now();
        self.request_times.push(now);
        self.error_count += 1;

        if now.duration_since(self.last_cleanup) > Duration::from_secs(60) {
            self.cleanup_old_entries();
            self.last_cleanup = now;
        }
    }

    /// Remove entries older than 5 minutes
    fn cleanup_old_entries(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(300);

        let mut valid_indices = Vec::new();
        for (i, &time) in self.request_times.iter().enumerate() {
            if time > cutoff {
                valid_indices.push(i);
            }
        }

        // Keep only recent entries
        let new_request_times: Vec<_> = valid_indices
            .iter()
            .map(|&i| self.request_times[i])
            .collect();

        let new_response_times: Vec<_> = valid_indices
            .iter()
            .filter_map(|&i| self.response_times.get(i).copied())
            .collect();

        self.request_times = new_request_times;
        self.response_times = new_response_times;
    }

    /// Calculate current request rate (requests per second)
    fn calculate_request_rate(&self) -> f64 {
        let cutoff = Instant::now() - Duration::from_secs(60);
        let recent_requests = self
            .request_times
            .iter()
            .filter(|&&time| time > cutoff)
            .count();

        recent_requests as f64 / 60.0
    }

    /// Calculate average response time
    fn calculate_avg_response_time(&self) -> f64 {
        if self.response_times.is_empty() {
            0.0
        } else {
            let sum: u64 = self.response_times.iter().sum();
            sum as f64 / self.response_times.len() as f64
        }
    }

    /// Calculate error rate
    fn calculate_error_rate(&self) -> f64 {
        let total = self.success_count + self.error_count;
        if total == 0 {
            0.0
        } else {
            self.error_count as f64 / total as f64
        }
    }
}

impl DeployedModel {
    /// Create a new deployed model
    pub fn new(model: Box<dyn ModelServing>, config: DeploymentConfig) -> Result<Self> {
        let metrics = Arc::new(Mutex::new(DeploymentMetrics {
            status: DeploymentStatus::Starting,
            active_instances: 1,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            request_rate: 0.0,
            avg_response_time_ms: 0.0,
            error_rate: 0.0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            last_health_check: chrono::Utc::now(),
            started_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }));

        let stats = Arc::new(Mutex::new(RequestStats::new()));

        let health_status = Arc::new(Mutex::new(HealthStatus {
            status: "starting".to_string(),
            details: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }));

        let deployed_model = Self {
            model,
            config,
            metrics,
            stats,
            health_status,
        };

        // Perform initial health check
        deployed_model.update_health_status()?;

        // Mark as running if health check passes
        {
            let mut metrics = deployed_model.metrics.lock().unwrap();
            metrics.status = DeploymentStatus::Running;
            metrics.updated_at = chrono::Utc::now();
        }

        Ok(deployed_model)
    }

    /// Get deployment configuration
    pub fn get_config(&self) -> &DeploymentConfig {
        &self.config
    }

    /// Get deployment metrics
    pub fn get_metrics(&self) -> DeploymentMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Update deployment metrics
    fn update_metrics(&self) -> Result<()> {
        let stats = self.stats.lock().unwrap();
        let mut metrics = self.metrics.lock().unwrap();

        metrics.request_rate = stats.calculate_request_rate();
        metrics.avg_response_time_ms = stats.calculate_avg_response_time();
        metrics.error_rate = stats.calculate_error_rate();
        metrics.total_requests = stats.success_count + stats.error_count;
        metrics.successful_requests = stats.success_count;
        metrics.failed_requests = stats.error_count;
        metrics.updated_at = chrono::Utc::now();

        // Simulate resource utilization (in a real implementation, this would come from system metrics)
        metrics.cpu_utilization = (metrics.request_rate / 100.0).min(1.0);
        metrics.memory_utilization = (metrics.total_requests as f64 / 10000.0).min(1.0);

        Ok(())
    }

    /// Update health status
    fn update_health_status(&self) -> Result<()> {
        let health_result = self.model.health_check();

        let mut health_status = self.health_status.lock().unwrap();
        match health_result {
            Ok(status) => {
                *health_status = status;
            }
            Err(e) => {
                health_status.status = "unhealthy".to_string();
                health_status.details.clear();
                health_status
                    .details
                    .insert("error".to_string(), e.to_string());
                health_status.timestamp = chrono::Utc::now();
            }
        }

        // Update deployment status based on health
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.last_health_check = chrono::Utc::now();

            if health_status.status == "healthy" {
                if metrics.status == DeploymentStatus::Degraded {
                    metrics.status = DeploymentStatus::Running;
                }
            } else if health_status.status == "unhealthy" {
                metrics.status = DeploymentStatus::Degraded;
            }
        }

        Ok(())
    }

    /// Check if scaling is needed
    pub fn should_scale_up(&self) -> bool {
        let metrics = self.metrics.lock().unwrap();
        let config = &self.config.scaling;

        metrics.cpu_utilization > config.scale_up_threshold
            || metrics.memory_utilization > config.scale_up_threshold
    }

    /// Check if scaling down is needed
    pub fn should_scale_down(&self) -> bool {
        let metrics = self.metrics.lock().unwrap();
        let config = &self.config.scaling;

        metrics.active_instances > config.min_instances
            && metrics.cpu_utilization < config.scale_down_threshold
            && metrics.memory_utilization < config.scale_down_threshold
    }

    /// Scale up deployment
    pub fn scale_up(&self) -> Result<()> {
        let mut metrics = self.metrics.lock().unwrap();
        let config = &self.config.scaling;

        if metrics.active_instances < config.max_instances {
            metrics.active_instances += 1;
            metrics.updated_at = chrono::Utc::now();
            log::info!(
                "Scaled up deployment to {} instances",
                metrics.active_instances
            );
        }

        Ok(())
    }

    /// Scale down deployment
    pub fn scale_down(&self) -> Result<()> {
        let mut metrics = self.metrics.lock().unwrap();
        let config = &self.config.scaling;

        if metrics.active_instances > config.min_instances {
            metrics.active_instances -= 1;
            metrics.updated_at = chrono::Utc::now();
            log::info!(
                "Scaled down deployment to {} instances",
                metrics.active_instances
            );
        }

        Ok(())
    }

    /// Stop deployment
    pub fn stop(&self) -> Result<()> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.status = DeploymentStatus::Stopping;
        metrics.updated_at = chrono::Utc::now();

        // In a real implementation, this would stop the actual instances

        metrics.status = DeploymentStatus::Stopped;
        metrics.active_instances = 0;

        Ok(())
    }

    /// Restart deployment
    pub fn restart(&self) -> Result<()> {
        self.stop()?;

        let mut metrics = self.metrics.lock().unwrap();
        metrics.status = DeploymentStatus::Starting;
        metrics.active_instances = self.config.scaling.min_instances;
        metrics.updated_at = chrono::Utc::now();

        // Perform health check
        drop(metrics);
        self.update_health_status()?;

        let mut metrics = self.metrics.lock().unwrap();
        metrics.status = DeploymentStatus::Running;

        Ok(())
    }
}

impl ModelServing for DeployedModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PredictionResponse> {
        let start_time = Instant::now();

        // Check if deployment is healthy
        {
            let metrics = self.metrics.lock().unwrap();
            if metrics.status != DeploymentStatus::Running {
                return Err(Error::InvalidOperation(format!(
                    "Deployment is not running (status: {:?})",
                    metrics.status
                )));
            }
        }

        // Perform prediction
        let result = self.model.predict(request);

        // Record statistics
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = self.stats.lock().unwrap();
            match &result {
                Ok(_) => stats.record_success(processing_time),
                Err(_) => stats.record_error(),
            }
        }

        // Update metrics periodically
        self.update_metrics()?;

        result
    }

    fn predict_batch(&self, request: &BatchPredictionRequest) -> Result<BatchPredictionResponse> {
        let start_time = Instant::now();

        // Check if deployment is healthy
        {
            let metrics = self.metrics.lock().unwrap();
            if metrics.status != DeploymentStatus::Running {
                return Err(Error::InvalidOperation(format!(
                    "Deployment is not running (status: {:?})",
                    metrics.status
                )));
            }
        }

        // Perform batch prediction
        let result = self.model.predict_batch(request);

        // Record statistics
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = self.stats.lock().unwrap();
            match &result {
                Ok(response) => {
                    let avg_time = processing_time / request.data.len().max(1) as u64;
                    for _ in 0..response.summary.successful_predictions {
                        stats.record_success(avg_time);
                    }
                    for _ in 0..response.summary.failed_predictions {
                        stats.record_error();
                    }
                }
                Err(_) => stats.record_error(),
            }
        }

        // Update metrics
        self.update_metrics()?;

        result
    }

    fn get_metadata(&self) -> &ModelMetadata {
        self.model.get_metadata()
    }

    fn health_check(&self) -> Result<HealthStatus> {
        self.update_health_status()?;
        Ok(self.health_status.lock().unwrap().clone())
    }

    fn info(&self) -> ModelInfo {
        let mut info = self.model.info();

        // Add deployment information
        info.configuration.insert(
            "deployment_config".to_string(),
            serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null),
        );

        info.configuration.insert(
            "deployment_metrics".to_string(),
            serde_json::to_value(&self.get_metrics()).unwrap_or(serde_json::Value::Null),
        );

        info
    }
}

/// Deployment manager for managing multiple deployments
pub struct DeploymentManager {
    /// Active deployments
    deployments: HashMap<String, DeployedModel>,
    /// Deployment configurations
    configs: HashMap<String, DeploymentConfig>,
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new() -> Self {
        Self {
            deployments: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Deploy a model
    pub fn deploy(
        &mut self,
        deployment_name: String,
        model: Box<dyn ModelServing>,
        config: DeploymentConfig,
    ) -> Result<()> {
        if self.deployments.contains_key(&deployment_name) {
            return Err(Error::InvalidOperation(format!(
                "Deployment '{}' already exists",
                deployment_name
            )));
        }

        let deployed_model = DeployedModel::new(model, config.clone())?;

        self.deployments
            .insert(deployment_name.clone(), deployed_model);
        self.configs.insert(deployment_name, config);

        Ok(())
    }

    /// Undeploy a model
    pub fn undeploy(&mut self, deployment_name: &str) -> Result<()> {
        if let Some(deployment) = self.deployments.get(deployment_name) {
            deployment.stop()?;
        }

        self.deployments.remove(deployment_name);
        self.configs.remove(deployment_name);

        Ok(())
    }

    /// Get deployment
    pub fn get_deployment(&self, deployment_name: &str) -> Option<&DeployedModel> {
        self.deployments.get(deployment_name)
    }

    /// List all deployments
    pub fn list_deployments(&self) -> Vec<String> {
        self.deployments.keys().cloned().collect()
    }

    /// Get deployment metrics
    pub fn get_deployment_metrics(&self, deployment_name: &str) -> Option<DeploymentMetrics> {
        self.deployments
            .get(deployment_name)
            .map(|deployment| deployment.get_metrics())
    }

    /// Scale deployment
    pub fn scale_deployment(&self, deployment_name: &str, target_instances: usize) -> Result<()> {
        let deployment = self.deployments.get(deployment_name).ok_or_else(|| {
            Error::KeyNotFound(format!("Deployment '{}' not found", deployment_name))
        })?;

        let current_instances = deployment.get_metrics().active_instances;

        if target_instances > current_instances {
            for _ in current_instances..target_instances {
                deployment.scale_up()?;
            }
        } else if target_instances < current_instances {
            for _ in target_instances..current_instances {
                deployment.scale_down()?;
            }
        }

        Ok(())
    }

    /// Auto-scale all deployments based on metrics
    pub fn auto_scale_all(&self) -> Result<()> {
        for deployment in self.deployments.values() {
            if deployment.should_scale_up() {
                deployment.scale_up()?;
            } else if deployment.should_scale_down() {
                deployment.scale_down()?;
            }
        }
        Ok(())
    }

    /// Health check all deployments
    pub fn health_check_all(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();

        for (name, deployment) in &self.deployments {
            match deployment.health_check() {
                Ok(status) => {
                    results.insert(name.clone(), status);
                }
                Err(e) => {
                    results.insert(
                        name.clone(),
                        HealthStatus {
                            status: "error".to_string(),
                            details: {
                                let mut details = HashMap::new();
                                details.insert("error".to_string(), e.to_string());
                                details
                            },
                            timestamp: chrono::Utc::now(),
                        },
                    );
                }
            }
        }

        results
    }
}

impl Default for DeploymentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::serving::serialization::GenericServingModel;
    use crate::ml::serving::ModelMetadata;

    fn create_test_config() -> DeploymentConfig {
        DeploymentConfig {
            model_name: "test_model".to_string(),
            model_version: "1.0.0".to_string(),
            environment: "test".to_string(),
            resources: ResourceConfig {
                cpu_cores: 1.0,
                memory_mb: 1024,
                gpu_memory_mb: None,
                max_concurrent_requests: 10,
            },
            scaling: ScalingConfig {
                min_instances: 1,
                max_instances: 5,
                target_cpu_utilization: 0.7,
                target_memory_utilization: 0.8,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
            },
            health_check: HealthCheckConfig {
                path: "/health".to_string(),
                interval_seconds: 30,
                timeout_seconds: 5,
                failure_threshold: 3,
                success_threshold: 2,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_logging: true,
                enable_tracing: false,
                metrics_interval_seconds: 60,
                log_level: "info".to_string(),
            },
        }
    }

    #[test]
    fn test_deployment_status() {
        let status = DeploymentStatus::Running;
        assert_eq!(status, DeploymentStatus::Running);
    }

    #[test]
    fn test_request_stats() {
        let mut stats = RequestStats::new();

        // Record some requests
        stats.record_success(100);
        stats.record_success(150);
        stats.record_error();

        assert!(stats.calculate_avg_response_time() > 0.0);
        assert!(stats.calculate_error_rate() > 0.0);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.error_count, 1);
    }

    #[test]
    fn test_deployment_manager() {
        let manager = DeploymentManager::new();

        // Test that manager starts empty
        assert!(manager.list_deployments().is_empty());

        // Test deployment configuration
        let config = create_test_config();
        assert_eq!(config.model_name, "test_model");
        assert_eq!(config.scaling.min_instances, 1);
        assert_eq!(config.scaling.max_instances, 5);
    }
}
