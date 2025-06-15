//! Comprehensive testing framework for trait compliance
//!
//! This module provides comprehensive tests to ensure all traits in the PandRS
//! system work correctly together and maintain their contracts.

#![allow(clippy::result_large_err)]

use pandrs::core::data_value::DataValue;
use pandrs::core::error::{Error, Result};
use std::any::Any;

/// Trait compliance test suite
pub struct TraitComplianceTestSuite;

impl TraitComplianceTestSuite {
    /// Run all trait compliance tests
    pub fn run_all_tests() -> Result<TestResults> {
        let mut results = TestResults::new();

        // Test core error handling system
        results.add_result("Error Handling", Self::test_error_handling());

        // Test data value system
        results.add_result("Data Value System", Self::test_data_value_system());

        // Test basic trait patterns
        results.add_result("Basic Trait Patterns", Self::test_basic_trait_patterns());

        // Test memory safety and thread safety
        results.add_result(
            "Memory and Thread Safety",
            Self::test_memory_thread_safety(),
        );

        // Test trait object compatibility
        results.add_result(
            "Trait Object Compatibility",
            Self::test_trait_object_compatibility(),
        );

        Ok(results)
    }

    /// Test error handling system compliance
    fn test_error_handling() -> TestResult {
        let mut test = TestResult::new("Error Handling");

        // Test error creation and propagation
        let parse_error = Error::InvalidInput("test parse error".to_string());
        test.assert(
            "parse error creation",
            matches!(parse_error, Error::InvalidInput(_)),
            "Should create parse error correctly",
        );

        let io_error = Error::IoError("test io error".to_string());
        test.assert(
            "io error creation",
            matches!(io_error, Error::IoError(_)),
            "Should create IO error correctly",
        );

        let invalid_op = Error::InvalidOperation("test invalid operation".to_string());
        test.assert(
            "invalid operation error",
            matches!(invalid_op, Error::InvalidOperation(_)),
            "Should create invalid operation error correctly",
        );

        // Test error display
        test.assert(
            "error display",
            !parse_error.to_string().is_empty(),
            "Error should have meaningful display",
        );

        // Test error conversion and chaining
        let chained_result: Result<i32> = Err(parse_error);
        test.assert(
            "error chaining",
            chained_result.is_err(),
            "Should properly chain errors",
        );

        test
    }

    /// Test data value system compliance
    fn test_data_value_system() -> TestResult {
        let mut test = TestResult::new("Data Value System");

        // Create test data values
        let int_value = TestDataValue::new_int(42);
        let float_value = TestDataValue::new_float(std::f64::consts::PI);
        let string_value = TestDataValue::new_string("hello".to_string());
        let null_value = TestDataValue::new_null();

        // Test basic properties
        test.assert(
            "int data type",
            int_value.data_type() == "i64",
            "Integer should have correct data type",
        );

        test.assert(
            "float data type",
            float_value.data_type() == "f64",
            "Float should have correct data type",
        );

        test.assert(
            "string data type",
            string_value.data_type() == "string",
            "String should have correct data type",
        );

        // Test null handling
        test.assert(
            "null detection",
            null_value.is_null(),
            "Null value should be detected as null",
        );

        test.assert(
            "non-null detection",
            !int_value.is_null(),
            "Non-null value should not be detected as null",
        );

        // Test string conversion
        test.assert(
            "int to string",
            int_value.to_string() == "42",
            "Integer should convert to string correctly",
        );

        test.assert(
            "float to string",
            float_value.to_string().starts_with("3.14"),
            "Float should convert to string correctly",
        );

        // Test cloning
        let cloned_int = int_value.clone_box();
        test.assert(
            "cloning preserves type",
            cloned_int.type_name() == int_value.type_name(),
            "Cloned value should preserve data type",
        );

        test.assert(
            "cloning preserves value",
            cloned_int.to_string() == int_value.to_string(),
            "Cloned value should preserve string representation",
        );

        test
    }

    /// Test basic trait patterns and generic programming
    fn test_basic_trait_patterns() -> TestResult {
        let mut test = TestResult::new("Basic Trait Patterns");

        // Test trait objects
        let values: Vec<Box<dyn DataValue>> = vec![
            Box::new(TestDataValue::new_int(1)),
            Box::new(TestDataValue::new_float(2.0)),
            Box::new(TestDataValue::new_string("three".to_string())),
        ];

        test.assert(
            "trait object collection",
            values.len() == 3,
            "Should be able to store different types in trait object collection",
        );

        // Test polymorphism
        let polymorphic_result: Vec<String> =
            values.iter().map(|v| v.type_name().to_string()).collect();

        test.assert(
            "polymorphic method calls",
            polymorphic_result.len() == 3,
            "Should be able to call methods polymorphically",
        );

        test.assert(
            "polymorphic type preservation",
            polymorphic_result.contains(&"i64".to_string())
                && polymorphic_result.contains(&"f64".to_string())
                && polymorphic_result.contains(&"string".to_string()),
            "Polymorphic calls should preserve individual types",
        );

        test
    }

    /// Test memory safety and thread safety patterns
    fn test_memory_thread_safety() -> TestResult {
        let mut test = TestResult::new("Memory and Thread Safety");

        // Test Send/Sync compliance for basic types
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<Error>();
        assert_sync::<Error>();

        test.assert(
            "error types are Send",
            true,
            "Error types should implement Send",
        );

        test.assert(
            "error types are Sync",
            true,
            "Error types should implement Sync",
        );

        // Test clone safety
        let original = TestDataValue::new_int(42);
        let cloned = original.clone();

        test.assert(
            "clone independence",
            original.to_string() == cloned.to_string(),
            "Cloned values should be independent but equal",
        );

        // Test that we can move values safely
        let value = TestDataValue::new_string("moveable".to_string());
        let moved_value = value;

        test.assert(
            "move semantics",
            moved_value.data_type() == "string",
            "Values should be moveable",
        );

        test
    }

    /// Test trait object compatibility and dynamic dispatch
    fn test_trait_object_compatibility() -> TestResult {
        let mut test = TestResult::new("Trait Object Compatibility");

        // Test Any compatibility
        let value = TestDataValue::new_int(42);
        let any_ref = value.as_any();

        test.assert(
            "Any trait compatibility",
            any_ref.downcast_ref::<TestDataValue>().is_some(),
            "DataValue should be compatible with Any trait",
        );

        // Test dynamic dispatch with different concrete types
        let values: Vec<Box<dyn DataValue>> = vec![
            Box::new(TestDataValue::new_int(1)),
            Box::new(TestDataValue::new_float(2.5)),
            Box::new(TestDataValue::new_string("test".to_string())),
            Box::new(TestDataValue::new_null()),
        ];

        let null_count = values
            .iter()
            .filter(|v| {
                if let Some(test_val) = v.as_any().downcast_ref::<TestDataValue>() {
                    test_val.is_null()
                } else {
                    false
                }
            })
            .count();
        test.assert(
            "dynamic null detection",
            null_count == 1,
            "Should be able to detect nulls dynamically",
        );

        let type_count = values
            .iter()
            .map(|v| v.type_name())
            .collect::<std::collections::HashSet<_>>()
            .len();

        test.assert(
            "dynamic type diversity",
            type_count == 4, // i64, f64, string, null
            "Should preserve type diversity in dynamic dispatch",
        );

        test
    }
}

/// Test results aggregator
#[derive(Debug)]
pub struct TestResults {
    results: Vec<(String, TestResult)>,
}

impl TestResults {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, name: impl Into<String>, result: TestResult) {
        self.results.push((name.into(), result));
    }

    pub fn total_tests(&self) -> usize {
        self.results.iter().map(|(_, r)| r.assertions.len()).sum()
    }

    pub fn passed_tests(&self) -> usize {
        self.results
            .iter()
            .map(|(_, r)| r.assertions.iter().filter(|a| a.passed).count())
            .sum()
    }

    pub fn failed_tests(&self) -> usize {
        self.total_tests() - self.passed_tests()
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_tests() == 0 {
            1.0
        } else {
            self.passed_tests() as f64 / self.total_tests() as f64
        }
    }

    pub fn print_summary(&self) {
        println!("=== Trait Compliance Test Results ===");
        println!("Total tests: {}", self.total_tests());
        println!("Passed: {}", self.passed_tests());
        println!("Failed: {}", self.failed_tests());
        println!("Success rate: {:.2}%", self.success_rate() * 100.0);
        println!();

        for (name, result) in &self.results {
            println!("=== {} ===", name);
            result.print_details();
            println!();
        }
    }
}

/// Individual test result
#[derive(Debug)]
pub struct TestResult {
    #[allow(dead_code)]
    name: String,
    assertions: Vec<TestAssertion>,
}

impl TestResult {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            assertions: Vec::new(),
        }
    }

    pub fn assert(
        &mut self,
        test_name: impl Into<String>,
        condition: bool,
        message: impl Into<String>,
    ) {
        self.assertions.push(TestAssertion {
            name: test_name.into(),
            passed: condition,
            message: message.into(),
        });
    }

    pub fn passed(&self) -> bool {
        self.assertions.iter().all(|a| a.passed)
    }

    pub fn print_details(&self) {
        for assertion in &self.assertions {
            let status = if assertion.passed { "✓" } else { "✗" };
            println!("  {} {}: {}", status, assertion.name, assertion.message);
        }
    }
}

/// Individual test assertion
#[derive(Debug)]
pub struct TestAssertion {
    name: String,
    passed: bool,
    message: String,
}

/// Test DataValue implementation for testing
#[derive(Debug, Clone)]
pub struct TestDataValue {
    data: TestData,
}

#[derive(Debug, Clone)]
enum TestData {
    Int(i64),
    Float(f64),
    String(String),
    Null,
}

impl TestDataValue {
    pub fn new_int(value: i64) -> Self {
        Self {
            data: TestData::Int(value),
        }
    }

    pub fn new_float(value: f64) -> Self {
        Self {
            data: TestData::Float(value),
        }
    }

    pub fn new_string(value: String) -> Self {
        Self {
            data: TestData::String(value),
        }
    }

    pub fn new_null() -> Self {
        Self {
            data: TestData::Null,
        }
    }
}

impl DataValue for TestDataValue {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        match &self.data {
            TestData::Int(_) => "i64",
            TestData::Float(_) => "f64",
            TestData::String(_) => "string",
            TestData::Null => "null",
        }
    }

    fn to_string(&self) -> String {
        match &self.data {
            TestData::Int(i) => ToString::to_string(i),
            TestData::Float(f) => ToString::to_string(f),
            TestData::String(s) => s.clone(),
            TestData::Null => "null".to_string(),
        }
    }

    fn clone_boxed(&self) -> Box<dyn DataValue> {
        Box::new(self.clone())
    }

    fn equals(&self, other: &dyn DataValue) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<TestDataValue>() {
            match (&self.data, &other.data) {
                (TestData::Int(a), TestData::Int(b)) => a == b,
                (TestData::Float(a), TestData::Float(b)) => (a - b).abs() < f64::EPSILON,
                (TestData::String(a), TestData::String(b)) => a == b,
                (TestData::Null, TestData::Null) => true,
                _ => false,
            }
        } else {
            false
        }
    }
}

impl TestDataValue {
    /// Check if the value is null
    pub fn is_null(&self) -> bool {
        matches!(self.data, TestData::Null)
    }

    /// Get the data type as a string (convenience method)
    pub fn data_type(&self) -> &'static str {
        self.type_name()
    }

    /// Clone as boxed DataValue (convenience method)
    pub fn clone_box(&self) -> Box<dyn DataValue> {
        self.clone_boxed()
    }
}

/// Trait to test generic programming patterns
pub trait TestOperation<T> {
    type Output;

    fn perform(&self, input: T) -> Self::Output;
}

/// Generic test struct
pub struct GenericTestStruct<T> {
    value: T,
}

impl<T: Clone> TestOperation<T> for GenericTestStruct<T> {
    type Output = T;

    fn perform(&self, _input: T) -> Self::Output {
        self.value.clone()
    }
}

impl<T> GenericTestStruct<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_compliance_suite() {
        let results = TraitComplianceTestSuite::run_all_tests().unwrap();
        results.print_summary();

        // Ensure we have a reasonable success rate
        assert!(
            results.success_rate() > 0.95,
            "Success rate should be above 95%"
        );
        assert!(results.total_tests() > 15, "Should have at least 15 tests");
    }

    #[test]
    fn test_error_handling_compliance() {
        let result = TraitComplianceTestSuite::test_error_handling();
        assert!(result.passed(), "Error handling should pass all tests");
    }

    #[test]
    fn test_data_value_compliance() {
        let result = TraitComplianceTestSuite::test_data_value_system();
        assert!(result.passed(), "Data value system should pass all tests");
    }

    #[test]
    fn test_trait_patterns() {
        let result = TraitComplianceTestSuite::test_basic_trait_patterns();
        assert!(
            result.passed(),
            "Basic trait patterns should pass all tests"
        );
    }

    #[test]
    fn test_memory_safety() {
        let result = TraitComplianceTestSuite::test_memory_thread_safety();
        assert!(
            result.passed(),
            "Memory and thread safety should pass all tests"
        );
    }

    #[test]
    fn test_trait_objects() {
        let result = TraitComplianceTestSuite::test_trait_object_compatibility();
        assert!(
            result.passed(),
            "Trait object compatibility should pass all tests"
        );
    }

    #[test]
    fn test_generic_programming() {
        // Test generic trait implementation
        let test_struct = GenericTestStruct::new(42i32);
        let result = test_struct.perform(100i32);
        assert_eq!(result, 42i32);

        // Test with different types
        let string_struct = GenericTestStruct::new("hello".to_string());
        let string_result = string_struct.perform("world".to_string());
        assert_eq!(string_result, "hello");
    }

    #[test]
    fn test_error_propagation() {
        fn fallible_operation() -> Result<i32> {
            Err(Error::InvalidOperation("test error".to_string()))
        }

        fn chain_operations() -> Result<i32> {
            let _value = fallible_operation()?;
            Ok(42)
        }

        let result = chain_operations();
        assert!(result.is_err());

        if let Err(err) = result {
            assert!(err.to_string().contains("test error"));
        }
    }
}
