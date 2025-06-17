//! Tests for resilience patterns and fault tolerance
//!
//! This test module validates retry mechanisms, circuit breakers, and
//! overall resilience management for PandRS connectors.

#[cfg(feature = "resilience")]
mod resilience_tests {
    use pandrs::config::resilience::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Test retry mechanism with different backoff strategies
    #[tokio::test]
    async fn test_retry_mechanism_exponential_backoff() {
        let config = RetryConfig {
            max_attempts: 4,
            base_delay_ms: 10,
            max_delay_ms: 1000,
            backoff_strategy: BackoffStrategy::Exponential,
            backoff_multiplier: 2.0,
            jitter: false,
            retryable_errors: vec!["TestError".to_string()],
        };

        let retry = RetryMechanism::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let start_time = Instant::now();
        let result = retry
            .execute(|| {
                let attempts = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                if attempts < 3 {
                    Err("TestError: Temporary failure")
                } else {
                    Ok("Success")
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 4);

        // Should have some delay due to retries
        assert!(start_time.elapsed().as_millis() >= 30); // 10 + 20 + 40 ms minimum
    }

    #[tokio::test]
    async fn test_retry_mechanism_fixed_backoff() {
        let config = RetryConfig {
            max_attempts: 3,
            base_delay_ms: 5,
            backoff_strategy: BackoffStrategy::Fixed,
            jitter: false,
            retryable_errors: vec!["RetryableError".to_string()],
            ..Default::default()
        };

        let retry = RetryMechanism::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = retry
            .execute(|| {
                let attempts = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                if attempts < 2 {
                    Err("RetryableError")
                } else {
                    Ok(42)
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_mechanism_non_retryable_error() {
        let config = RetryConfig {
            max_attempts: 3,
            base_delay_ms: 1,
            retryable_errors: vec!["RetryableError".to_string()],
            ..Default::default()
        };

        let retry = RetryMechanism::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<&str, _> = retry
            .execute(|| {
                attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                Err("NonRetryableError")
            })
            .await;

        assert!(result.is_err());
        // Should fail immediately without retries
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }

    /// Test circuit breaker basic functionality
    #[test]
    fn test_circuit_breaker_state_transitions() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            minimum_calls: 5,
            timeout_seconds: 1,
            ..Default::default()
        };

        let cb = CircuitBreaker::new(config);

        // Initially closed
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_execute());

        // Record successful calls
        for _ in 0..3 {
            cb.record_success();
        }
        assert_eq!(cb.state(), CircuitState::Closed);

        // Record failures (not enough to trip yet due to minimum_calls)
        for _ in 0..2 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Closed);

        // One more failure should trip the circuit (5 total calls, 3 failures >= threshold)
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
    }

    #[tokio::test]
    async fn test_circuit_breaker_timeout_and_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            minimum_calls: 2,
            timeout_seconds: 1,
            half_open_max_calls: 2,
            success_threshold_percentage: 50.0,
            ..Default::default()
        };

        let cb = CircuitBreaker::new(config);

        // Trip the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should transition to half-open
        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record success in half-open state
        cb.record_success();
        cb.record_success();

        // Should close circuit after successful calls
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            minimum_calls: 2,
            timeout_seconds: 1,
            ..Default::default()
        };

        let cb = CircuitBreaker::new(config);

        // Trip the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should be half-open
        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Failure in half-open should reopen circuit
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    /// Test resilience manager integration
    #[tokio::test]
    async fn test_resilience_manager_integration() {
        let manager = ResilienceManager::new();

        // Set custom retry config
        let retry_config = RetryConfig {
            max_attempts: 2,
            base_delay_ms: 5,
            retryable_errors: vec!["TestError".to_string()],
            ..Default::default()
        };
        manager.set_retry_config("test_service", retry_config);

        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = manager
            .execute_with_resilience("test_service", || {
                let attempts = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                if attempts == 0 {
                    Err("TestError")
                } else {
                    Ok("Success")
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_resilience_manager_circuit_breaker_integration() {
        let manager = ResilienceManager::new();

        // Execute operations that will trip the circuit breaker
        for _ in 0..10 {
            let _ = manager
                .execute_with_resilience("failing_service", || Err::<(), _>("PersistentError"))
                .await;
        }

        // Circuit should be open now
        let result = manager
            .execute_with_resilience("failing_service", || Ok::<&str, &str>("Success"))
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Circuit breaker is open"));
    }

    #[test]
    fn test_resilience_manager_health_status() {
        let manager = ResilienceManager::new();

        // Get circuit breaker to initialize it
        let cb = manager.get_circuit_breaker("test_service");
        cb.record_success();
        cb.record_failure();

        let health_status = manager.get_health_status();

        // Should have health information for the service
        assert!(health_status.contains_key("test_service"));

        let service_health = &health_status["test_service"];
        assert_eq!(service_health.service_name, "test_service");
        assert_eq!(service_health.total_calls, 2);
        assert_eq!(service_health.successful_calls, 1);
        assert_eq!(service_health.failed_calls, 1);
        assert_eq!(service_health.success_rate, 50.0);
    }

    /// Test custom backoff strategies
    #[tokio::test]
    async fn test_custom_backoff_strategy() {
        let config = RetryConfig {
            max_attempts: 4,
            backoff_strategy: BackoffStrategy::Custom(vec![5, 10, 20, 50]),
            jitter: false,
            retryable_errors: vec!["TestError".to_string()],
            ..Default::default()
        };

        let retry = RetryMechanism::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let start_time = Instant::now();
        let result = retry
            .execute(|| {
                let attempts = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                if attempts < 3 {
                    Err("TestError")
                } else {
                    Ok("Success")
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 4);

        // Should have used custom delays: 5 + 10 + 20 = 35ms minimum
        assert!(start_time.elapsed().as_millis() >= 35);
    }

    /// Test linear backoff strategy
    #[tokio::test]
    async fn test_linear_backoff_strategy() {
        let config = RetryConfig {
            max_attempts: 3,
            base_delay_ms: 10,
            backoff_strategy: BackoffStrategy::Linear,
            jitter: false,
            retryable_errors: vec!["TestError".to_string()],
            ..Default::default()
        };

        let retry = RetryMechanism::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let start_time = Instant::now();
        let result = retry
            .execute(|| {
                let attempts = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                if attempts < 2 {
                    Err("TestError")
                } else {
                    Ok("Success")
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);

        // Linear backoff: 10ms (attempt 1) + 20ms (attempt 2) = 30ms minimum
        assert!(start_time.elapsed().as_millis() >= 30);
    }

    /// Test circuit breaker statistics tracking
    #[test]
    fn test_circuit_breaker_statistics() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        // Record various operations
        cb.record_success();
        cb.record_success();
        cb.record_failure();
        cb.record_rejection();

        let stats = cb.stats();
        assert_eq!(stats.total_calls, 3); // Success and failure calls
        assert_eq!(stats.successful_calls, 2);
        assert_eq!(stats.failed_calls, 1);
        assert_eq!(stats.rejected_calls, 1);
        assert!(stats.last_failure_time.is_some());
    }

    /// Test max delay enforcement in retry mechanism
    #[tokio::test]
    async fn test_retry_max_delay_enforcement() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay_ms: 100,
            max_delay_ms: 200, // Cap at 200ms
            backoff_strategy: BackoffStrategy::Exponential,
            backoff_multiplier: 10.0, // Would normally create very large delays
            jitter: false,
            retryable_errors: vec!["TestError".to_string()],
        };

        let retry = RetryMechanism::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let start_time = Instant::now();
        let result = retry
            .execute(|| {
                let attempts = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
                if attempts < 3 {
                    Err("TestError")
                } else {
                    Ok("Success")
                }
            })
            .await;

        assert!(result.is_ok());
        let elapsed = start_time.elapsed().as_millis();

        // Even with exponential backoff, delays should be capped
        // Expected: 100ms + 200ms + 200ms = 500ms maximum (plus some tolerance)
        assert!(
            elapsed < 800,
            "Elapsed time {} ms should be less than 800ms",
            elapsed
        );
    }
}
