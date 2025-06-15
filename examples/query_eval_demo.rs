#![allow(clippy::result_large_err)]

use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::query::{LiteralValue, QueryContext, QueryEngine, QueryExt};
use pandrs::error::Result;
use pandrs::series::base::Series;

fn main() -> Result<()> {
    println!("=== Alpha 4 Query and Eval Expression Engine Example ===\n");

    // Create sample employee data
    println!("1. Creating Sample Employee Data:");
    let mut df = DataFrame::new();

    let names = vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    ];
    let ages = vec!["25", "30", "35", "28", "32", "45", "29", "38"];
    let salaries = vec![
        "50000", "65000", "75000", "58000", "62000", "85000", "56000", "72000",
    ];
    let departments = vec![
        "IT",
        "HR",
        "IT",
        "Marketing",
        "IT",
        "Finance",
        "HR",
        "Finance",
    ];
    let experience = vec!["2", "5", "8", "3", "6", "15", "4", "10"];

    let name_series = Series::new(
        names.into_iter().map(|s| s.to_string()).collect(),
        Some("Name".to_string()),
    )?;
    let age_series = Series::new(
        ages.into_iter().map(|s| s.to_string()).collect(),
        Some("Age".to_string()),
    )?;
    let salary_series = Series::new(
        salaries.into_iter().map(|s| s.to_string()).collect(),
        Some("Salary".to_string()),
    )?;
    let dept_series = Series::new(
        departments.into_iter().map(|s| s.to_string()).collect(),
        Some("Department".to_string()),
    )?;
    let exp_series = Series::new(
        experience.into_iter().map(|s| s.to_string()).collect(),
        Some("Experience".to_string()),
    )?;

    df.add_column("Name".to_string(), name_series)?;
    df.add_column("Age".to_string(), age_series)?;
    df.add_column("Salary".to_string(), salary_series)?;
    df.add_column("Department".to_string(), dept_series)?;
    df.add_column("Experience".to_string(), exp_series)?;

    println!("Original Employee Data:");
    println!("{:?}", df);

    println!("\n=== Basic Query Operations ===\n");

    // 2. Basic comparison queries
    println!("2. Basic Comparison Queries:");

    // Age greater than 30
    let age_filter = df.query("Age > 30")?;
    println!("Employees older than 30:");
    println!("{:?}", age_filter);

    // High salary filter
    let salary_filter = df.query("Salary >= 65000")?;
    println!("\nEmployees with salary >= 65000:");
    println!("{:?}", salary_filter);

    // Department filter
    let it_filter = df.query("Department == 'IT'")?;
    println!("\nIT Department employees:");
    println!("{:?}", it_filter);

    println!("\n=== Logical Operations ===\n");

    // 3. Logical operations (AND, OR, NOT)
    println!("3. Logical Operations:");

    // Multiple conditions with AND
    let senior_it = df.query("Department == 'IT' && Age > 30")?;
    println!("Senior IT employees (Age > 30):");
    println!("{:?}", senior_it);

    // Multiple conditions with OR
    let hr_or_finance = df.query("Department == 'HR' || Department == 'Finance'")?;
    println!("\nHR or Finance employees:");
    println!("{:?}", hr_or_finance);

    // NOT operation
    let not_it = df.query("!(Department == 'IT')")?;
    println!("\nNon-IT employees:");
    println!("{:?}", not_it);

    println!("\n=== Arithmetic Operations in Queries ===\n");

    // 4. Arithmetic operations
    println!("4. Arithmetic Operations:");

    // Salary per year of experience
    let efficient_employees = df.query("Salary / Experience > 8000")?;
    println!("Employees with salary/experience ratio > 8000:");
    println!("{:?}", efficient_employees);

    // Age and experience combination
    let young_experienced = df.query("Age - Experience < 25")?;
    println!("\nEmployees who started young (Age - Experience < 25):");
    println!("{:?}", young_experienced);

    println!("\n=== Complex Query Expressions ===\n");

    // 5. Complex expressions
    println!("5. Complex Query Expressions:");

    // Complex logical expression
    let complex_query =
        df.query("(Age > 30 && Salary > 60000) || (Experience > 10 && Department != 'HR')")?;
    println!("Complex filter - Senior high earners OR experienced non-HR:");
    println!("{:?}", complex_query);

    // Mathematical expression with parentheses
    let bonus_eligible = df.query("(Age + Experience) * 1000 > Salary / 10")?;
    println!("\nBonus eligible employees (complex formula):");
    println!("{:?}", bonus_eligible);

    println!("\n=== Expression Evaluation with .eval() ===\n");

    // 6. Expression evaluation to create new columns
    println!("6. Expression Evaluation (.eval()):");

    // Calculate age when started working
    let with_start_age = df.eval("Age - Experience", "StartAge")?;
    println!("DataFrame with calculated start age:");
    println!("{:?}", with_start_age);

    // Calculate salary per year of experience
    let with_efficiency = df.eval("Salary / Experience", "SalaryPerExp")?;
    println!("\nDataFrame with salary efficiency ratio:");
    println!("{:?}", with_efficiency);

    // Complex calculation
    let with_score = df.eval(
        "(Salary / 1000) + (Experience * 2) - (Age * 0.5)",
        "EmployeeScore",
    )?;
    println!("\nDataFrame with employee score:");
    println!("{:?}", with_score);

    println!("\n=== Custom Context and Functions ===\n");

    // 7. Custom context with variables and functions
    println!("7. Custom Context and Functions:");

    let mut context = QueryContext::new();

    // Add variables
    context.set_variable("min_salary".to_string(), LiteralValue::Number(60000.0));
    context.set_variable("target_age".to_string(), LiteralValue::Number(35.0));

    // Add custom function
    context.add_function("salary_grade".to_string(), |args| {
        if args.is_empty() {
            return 0.0;
        }
        let salary = args[0];
        if salary >= 80000.0 {
            5.0
        } else if salary >= 70000.0 {
            4.0
        } else if salary >= 60000.0 {
            3.0
        } else if salary >= 50000.0 {
            2.0
        } else {
            1.0
        }
    });

    // Use custom context
    let engine = QueryEngine::with_context(context);
    let custom_query = engine.query(&df, "Salary > min_salary && Age < target_age")?;
    println!("Query with custom variables:");
    println!("{:?}", custom_query);

    println!("\n=== Mathematical Functions ===\n");

    // 8. Built-in mathematical functions
    println!("8. Built-in Mathematical Functions:");

    // Using built-in functions in queries
    let sqrt_age = df.eval("sqrt(Age)", "SqrtAge")?;
    println!("DataFrame with square root of age:");
    println!("{:?}", sqrt_age);

    // Logarithmic salary analysis
    let log_salary = df.eval("log(Salary)", "LogSalary")?;
    println!("\nDataFrame with logarithmic salary:");
    println!("{:?}", log_salary);

    // Power calculations
    let power_calc = df.eval("Experience ** 2", "ExpSquared")?;
    println!("\nDataFrame with experience squared:");
    println!("{:?}", power_calc);

    println!("\n=== String Operations ===\n");

    // 9. String operations and comparisons
    println!("9. String Operations:");

    // String equality
    let marketing_employees = df.query("Department == 'Marketing'")?;
    println!("Marketing department employees:");
    println!("{:?}", marketing_employees);

    // String concatenation in eval
    let name_dept = df.eval("Name + '_' + Department", "NameDept")?;
    println!("\nDataFrame with name and department combined:");
    println!("{:?}", name_dept);

    println!("\n=== Performance Demonstration ===\n");

    // 10. Performance with larger dataset
    println!("10. Performance Demonstration:");
    demonstrate_performance()?;

    println!("\n=== Error Handling ===\n");

    // 11. Error handling examples
    println!("11. Error Handling:");
    demonstrate_error_handling(&df)?;

    println!("\n=== Alpha 4 Query and Eval Engine Complete ===");
    println!("\nNew query and evaluation capabilities implemented:");
    println!("✓ String-based query expressions (.query() method)");
    println!("✓ Expression evaluation (.eval() method)");
    println!("✓ Boolean indexing with complex conditions");
    println!("✓ Mathematical operations and comparisons");
    println!("✓ Logical operations (AND, OR, NOT)");
    println!("✓ Arithmetic operations (+, -, *, /, %, **)");
    println!("✓ Built-in mathematical functions");
    println!("✓ Custom variables and functions");
    println!("✓ String operations and comparisons");
    println!("✓ Parentheses for operation precedence");
    println!("✓ Robust error handling and validation");

    Ok(())
}

/// Demonstrate performance with larger dataset
fn demonstrate_performance() -> Result<()> {
    println!("--- Performance with Large Dataset ---");

    // Create a larger dataset
    let mut large_df = DataFrame::new();
    let size = 1000;

    let ids: Vec<String> = (1..=size).map(|i| i.to_string()).collect();
    let values: Vec<String> = (1..=size).map(|i| (i as f64 * 1.5).to_string()).collect();
    let categories: Vec<String> = (1..=size)
        .map(|i| match i % 4 {
            0 => "A".to_string(),
            1 => "B".to_string(),
            2 => "C".to_string(),
            _ => "D".to_string(),
        })
        .collect();

    let id_series = Series::new(ids, Some("ID".to_string()))?;
    let value_series = Series::new(values, Some("Value".to_string()))?;
    let cat_series = Series::new(categories, Some("Category".to_string()))?;

    large_df.add_column("ID".to_string(), id_series)?;
    large_df.add_column("Value".to_string(), value_series)?;
    large_df.add_column("Category".to_string(), cat_series)?;

    println!("Created dataset with {} rows", size);

    // Test query performance
    let start = std::time::Instant::now();
    let filtered = large_df.query("Value > 500 && Category == 'A'")?;
    let duration = start.elapsed();

    println!(
        "Query 'Value > 500 && Category == \"A\"' took: {:?}",
        duration
    );
    println!("Filtered {} rows to {} rows", size, filtered.row_count());

    // Test eval performance
    let start = std::time::Instant::now();
    let _with_calc = large_df.eval("Value * 2 + sqrt(Value)", "Calculated")?;
    let duration = start.elapsed();

    println!("Eval 'Value * 2 + sqrt(Value)' took: {:?}", duration);

    Ok(())
}

/// Demonstrate error handling
fn demonstrate_error_handling(df: &DataFrame) -> Result<()> {
    println!("--- Error Handling Examples ---");

    // Invalid column name
    match df.query("NonExistentColumn > 10") {
        Ok(_) => println!("Unexpected success with invalid column"),
        Err(e) => println!("Expected error for invalid column: {:?}", e),
    }

    // Invalid syntax
    match df.query("Age >") {
        Ok(_) => println!("Unexpected success with invalid syntax"),
        Err(e) => println!("Expected error for invalid syntax: {:?}", e),
    }

    // Division by zero
    match df.eval("Salary / 0", "Invalid") {
        Ok(_) => println!("Unexpected success with division by zero"),
        Err(e) => println!("Expected error for division by zero: {:?}", e),
    }

    // Invalid function
    match df.eval("unknown_function(Age)", "Invalid") {
        Ok(_) => println!("Unexpected success with unknown function"),
        Err(e) => println!("Expected error for unknown function: {:?}", e),
    }

    // Type mismatch
    match df.query("Age && Salary") {
        Ok(_) => println!("Unexpected success with type mismatch"),
        Err(e) => println!("Expected error for type mismatch: {:?}", e),
    }

    Ok(())
}
