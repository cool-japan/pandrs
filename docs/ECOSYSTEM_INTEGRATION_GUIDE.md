# PandRS Ecosystem Integration Guide

## 🌐 Overview

PandRS provides comprehensive integration with the modern data ecosystem, offering seamless connectivity to databases, cloud storage, Python environments, and Apache Arrow. This guide covers all ecosystem integration features with practical examples and best practices.

## 📊 Quick Start

```rust
use pandrs::prelude::*;
use pandrs::connectors::*;

// Create a DataFrame
let mut df = DataFrame::new();
df.add_column("id".to_string(), Series::new((1..=1000).map(|i| i.to_string()).collect(), Some("id".to_string()))?)?;

// Write to cloud storage
let s3_connector = CloudConnectorFactory::s3();
s3_connector.write_dataframe(&df, "my-bucket", "data/output.parquet", FileFormat::Parquet).await?;

// Query from database
let db_connector = DatabaseConnectorFactory::postgresql();
let result = db_connector.query("SELECT * FROM users WHERE active = true").await?;
```

---

## 🗄️ Database Integration

### Supported Databases

- **PostgreSQL** - Full async support with connection pooling
- **SQLite** - In-memory and file-based databases
- **MySQL** - (Planned for v0.2.0)

### Configuration

```rust
use pandrs::connectors::database::*;

// PostgreSQL Configuration
let pg_config = DatabaseConfig::new("postgresql://user:password@localhost:5432/mydb")
    .with_pool_size(20)
    .with_timeout(30)
    .with_ssl()
    .with_parameter("sslmode", "require");

// SQLite Configuration  
let sqlite_config = DatabaseConfig::new("sqlite:///path/to/database.db")
    .with_pool_size(5);

// In-memory SQLite
let memory_config = DatabaseConfig::new("sqlite::memory:");
```

### Basic Operations

#### Connect and Query
```rust
use pandrs::connectors::database::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create and connect
    let mut connector = DatabaseConnectorFactory::postgresql();
    connector.connect(&pg_config).await?;
    
    // Execute query
    let df = connector.query("SELECT id, name, created_at FROM users WHERE active = true").await?;
    println!("Retrieved {} rows", df.row_count());
    
    // Parameterized query
    let filtered_df = connector.query_with_params(
        "SELECT * FROM orders WHERE amount > $1 AND date >= $2",
        &[&1000, &"2024-01-01"]
    ).await?;
    
    Ok(())
}
```

#### Write DataFrame to Database
```rust
// Write modes: Replace, Append, Fail
connector.write_table(&df, "customers", WriteMode::Replace).await?;
connector.write_table(&new_data, "customers", WriteMode::Append).await?;
```

#### Database Metadata
```rust
// List all tables
let tables = connector.list_tables().await?;
println!("Available tables: {:?}", tables);

// Get table schema
let table_info = connector.get_table_info("customers").await?;
println!("Table schema: {:?}", table_info);
```

### Advanced Features

#### Transaction Management
```rust
// Begin transaction (returns transaction ID for tracking)
let tx_id = connector.begin_transaction().await?;
println!("Started transaction: {}", tx_id);

// Execute multiple operations
connector.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1").await?;
connector.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2").await?;

// Commit is handled automatically or explicitly through SQL
connector.execute("COMMIT").await?;
```

#### Connection Health
```rust
// Health check
match connector.execute("SELECT 1").await {
    Ok(_) => println!("Database connection healthy"),
    Err(e) => println!("Database connection failed: {}", e),
}
```

---

## ☁️ Cloud Storage Integration

### Supported Providers

- **AWS S3** - Complete S3 API support
- **Google Cloud Storage** - Full GCS integration
- **Azure Blob Storage** - Comprehensive Azure support
- **MinIO** - S3-compatible object storage

### Authentication

#### AWS S3
```rust
use pandrs::connectors::cloud::*;

// Environment-based credentials (recommended)
let s3_config = CloudConfig::new(
    CloudProvider::AWS,
    CloudCredentials::Environment
).with_region("us-west-2");

// Explicit credentials
let s3_config = CloudConfig::new(
    CloudProvider::AWS,
    CloudCredentials::AWS {
        access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
        secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        session_token: None,
    }
).with_region("us-west-2");
```

#### Google Cloud Storage
```rust
let gcs_config = CloudConfig::new(
    CloudProvider::GCS,
    CloudCredentials::GCS {
        project_id: "my-project-id".to_string(),
        service_account_key: "/path/to/service-account.json".to_string(),
    }
);
```

#### Azure Blob Storage
```rust
let azure_config = CloudConfig::new(
    CloudProvider::Azure,
    CloudCredentials::Azure {
        account_name: "mystorageaccount".to_string(),
        account_key: "base64-encoded-key".to_string(),
    }
);
```

### File Operations

#### Read DataFrames
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let mut s3_connector = CloudConnectorFactory::s3();
    s3_connector.connect(&s3_config).await?;
    
    // Read CSV file
    let df = s3_connector.read_dataframe(
        "my-bucket",
        "data/customers.csv",
        FileFormat::CSV { delimiter: ',', has_header: true }
    ).await?;
    
    // Read Parquet file (automatic compression detection)
    let df = s3_connector.read_dataframe(
        "my-bucket", 
        "data/large_dataset.parquet",
        FileFormat::Parquet
    ).await?;
    
    // Read JSON Lines
    let df = s3_connector.read_dataframe(
        "my-bucket",
        "logs/events.jsonl", 
        FileFormat::JSONL
    ).await?;
    
    Ok(())
}
```

#### Write DataFrames
```rust
// Write as Parquet (recommended for large datasets)
s3_connector.write_dataframe(
    &df,
    "analytics-bucket",
    "processed/sales_2024.parquet",
    FileFormat::Parquet
).await?;

// Write as CSV
s3_connector.write_dataframe(
    &df,
    "exports-bucket", 
    "reports/monthly_summary.csv",
    FileFormat::CSV { delimiter: ',', has_header: true }
).await?;
```

#### Object Management
```rust
// List objects with prefix
let objects = s3_connector.list_objects("my-bucket", Some("data/")).await?;
for obj in objects {
    println!("Found: {} ({} bytes)", obj.key, obj.size);
}

// Check if object exists
let exists = s3_connector.object_exists("my-bucket", "data/file.csv").await?;

// Get object metadata
let metadata = s3_connector.get_object_metadata("my-bucket", "data/file.parquet").await?;
println!("File size: {} bytes, modified: {:?}", metadata.size, metadata.last_modified);

// Download to local file
s3_connector.download_object("my-bucket", "data/source.csv", "/tmp/local_copy.csv").await?;

// Upload from local file
s3_connector.upload_object("/tmp/processed.parquet", "my-bucket", "results/output.parquet").await?;
```

### Format Detection

PandRS automatically detects file formats based on extensions:

```rust
use pandrs::connectors::cloud::FileFormat;

// Automatic detection
let format = FileFormat::from_extension("data.csv");        // CSV with comma delimiter
let format = FileFormat::from_extension("data.parquet");    // Parquet
let format = FileFormat::from_extension("data.json");       // JSON
let format = FileFormat::from_extension("logs.jsonl");      // JSON Lines
```

---

## 🏹 Apache Arrow Integration

### Zero-Copy Operations

```rust
use pandrs::arrow_integration::*;

// Convert DataFrame to Arrow RecordBatch
let record_batch = df.to_arrow()?;
println!("Arrow schema: {}", record_batch.schema());

// Convert back from Arrow
let df2 = DataFrame::from_arrow(&record_batch)?;

// Batch processing for large datasets
let batches = ArrowConverter::dataframes_to_record_batches(
    &[df1, df2, df3], 
    Some(1000) // batch size
)?;
```

### Arrow Compute Operations

```rust
// Use Arrow compute kernels for performance
let result = df.compute_arrow(ArrowOperation::Sum("sales_amount".to_string()))?;
let result = df.compute_arrow(ArrowOperation::Mean("temperature".to_string()))?;
let result = df.compute_arrow(ArrowOperation::Filter("age > 18".to_string()))?;
```

### Memory Efficiency

```rust
// Arrow integration provides:
// - Columnar memory layout for cache efficiency
// - SIMD vectorized operations
// - Zero-copy data sharing with other Arrow-compatible libraries
// - Efficient serialization/deserialization

// Example: Processing large datasets
let large_df = DataFrame::from_csv("large_dataset.csv", true)?;
let arrow_batch = large_df.to_arrow()?; // Zero-copy conversion
let processed = arrow_batch.compute_aggregate()?; // SIMD-optimized
let result_df = DataFrame::from_arrow(&processed)?; // Zero-copy back
```

---

## 🐍 Python Integration

### PyO3 Bindings

PandRS provides pandas-compatible Python bindings for seamless integration with existing Python workflows.

#### Installation

```bash
pip install pandrs
```

#### Basic Usage

```python
import pandrs as pr
import pandas as pd

# Create PandRS DataFrame
df = pr.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Tokyo']
})

# Pandas-compatible methods
print(df.head())
print(df.info())
print(df.describe())

# Indexing
print(df.iloc[0])  # First row
print(df.loc[df['age'] > 25])  # Conditional selection

# Convert to/from pandas
pandas_df = df.to_pandas()
pandrs_df = pr.DataFrame.from_pandas(pandas_df)
```

#### Advanced Operations

```python
# GroupBy operations
grouped = df.groupby('city').agg({
    'age': ['mean', 'std'],
    'name': 'count'
})

# Window operations
df['rolling_mean'] = df['age'].rolling(window=3).mean()
df['expanding_sum'] = df['age'].expanding().sum()
df['ewm_mean'] = df['age'].ewm(span=3).mean()

# String operations
df['name_upper'] = df['name'].str.upper()
df['name_length'] = df['name'].str.len()
```

#### Jupyter Integration

```python
# Rich HTML display in Jupyter
df  # Automatically renders as HTML table

# Progress bars for long operations
with pr.progress_bar():
    result = df.groupby('category').apply(complex_function)

# Memory usage display
df.memory_usage(deep=True)
```

---

## 🔗 Unified Data Access

### Connection String Patterns

PandRS supports unified data access through connection strings:

```rust
// Database connections
DataFrame::read_from("sqlite:///data.db", "SELECT * FROM users")?;
DataFrame::read_from("postgresql://localhost/mydb", "SELECT * FROM orders")?;

// Cloud storage  
DataFrame::read_from("s3://bucket/data.parquet", "")?;
DataFrame::read_from("gs://bucket/dataset.csv", "")?;
DataFrame::read_from("azure://container/file.json", "")?;

// Local files
DataFrame::read_from("file:///path/to/data.csv", "")?;
```

### Automatic Connector Selection

```rust
use pandrs::connectors::*;

// The system automatically selects the appropriate connector
async fn load_data(source: &str, query: &str) -> Result<DataFrame> {
    match source {
        s if s.starts_with("postgresql://") => {
            let connector = DatabaseConnectorFactory::postgresql();
            connector.query(query).await
        },
        s if s.starts_with("s3://") => {
            let connector = CloudConnectorFactory::s3();
            // Parse bucket and key from URL
            connector.read_dataframe(bucket, key, FileFormat::from_extension(key).unwrap()).await
        },
        _ => Err(Error::UnsupportedDataSource(source.to_string()))
    }
}
```

---

## 🚀 Performance Optimization

### Best Practices

#### Database Connections

```rust
// Use connection pooling
let config = DatabaseConfig::new(connection_string)
    .with_pool_size(20)  // Optimize based on workload
    .with_timeout(30);   // Prevent hanging connections

// Batch operations
let mut batch = Vec::new();
for record in large_dataset {
    batch.push(record);
    if batch.len() >= 1000 {
        connector.write_batch(&batch).await?;
        batch.clear();
    }
}
```

#### Cloud Storage

```rust
// Use appropriate formats
// Parquet: Best for analytical workloads, columnar compression
// CSV: Human-readable, but slower for large datasets  
// JSON Lines: Good for log data and streaming

// Optimize batch sizes
let batches = ArrowConverter::dataframes_to_record_batches(
    &dataframes,
    Some(5000)  // Tune based on memory and network
)?;

// Parallel uploads for large datasets
use tokio::task;
let handles: Vec<_> = chunks.into_iter().map(|chunk| {
    task::spawn(async move {
        connector.write_dataframe(&chunk, bucket, &key, format).await
    })
}).collect();
```

#### Memory Management

```rust
// Use streaming for large datasets
let stream = connector.read_streaming("large_table").await?;
while let Some(batch) = stream.next().await {
    let df = DataFrame::from_arrow(&batch?)?;
    process_chunk(&df)?;
    // DataFrame is dropped here, freeing memory
}

// Prefer Arrow operations for large data
let result = df.to_arrow()?
    .compute_aggregate()?  // SIMD-optimized
    .filter(predicate)?    // Pushdown filtering
    .to_dataframe()?;
```

---

## 🛠️ Configuration Management

### Environment Variables

```bash
# Database
export PANDRS_DB_URL="postgresql://localhost/mydb"
export PANDRS_DB_POOL_SIZE=20

# AWS
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-west-2"

# GCP
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="my-project"

# Azure
export AZURE_STORAGE_ACCOUNT="mystorageaccount"
export AZURE_STORAGE_KEY="base64-key"
```

### Configuration Files

#### YAML Configuration
```yaml
# pandrs.yml
database:
  url: "postgresql://localhost/mydb"
  pool_size: 20
  timeout: 30
  ssl: true

cloud:
  aws:
    region: "us-west-2"
    credentials: "environment"
  gcp:
    project_id: "my-project"
    service_account: "/path/to/key.json"

performance:
  batch_size: 5000
  parallel_workers: 4
  memory_limit: "2GB"
```

#### Loading Configuration
```rust
use pandrs::config::*;

// Load from file
let config = PandRSConfig::from_file("pandrs.yml")?;

// Load from environment
let config = PandRSConfig::from_env()?;

// Use configuration
let db_connector = DatabaseConnectorFactory::from_config(&config.database)?;
```

---

## 🔒 Security Best Practices

### Credential Management

```rust
// ✅ Good: Use environment variables
let config = CloudConfig::new(
    CloudProvider::AWS,
    CloudCredentials::Environment
);

// ❌ Bad: Hardcode credentials
let config = CloudConfig::new(
    CloudProvider::AWS,
    CloudCredentials::AWS {
        access_key_id: "AKIA...".to_string(),  // Never do this!
        secret_access_key: "secret".to_string(),
        session_token: None,
    }
);

// ✅ Good: Use credential files with restricted permissions
// chmod 600 ~/.aws/credentials
let config = CloudConfig::from_file("~/.aws/credentials")?;
```

### Network Security

```rust
// Enable SSL for database connections
let config = DatabaseConfig::new(connection_string)
    .with_ssl()
    .with_parameter("sslmode", "require");

// Use VPC endpoints for cloud storage
let config = CloudConfig::new(provider, credentials)
    .with_endpoint("https://s3.vpc-endpoint.amazonaws.com");
```

---

## 🧪 Testing and Development

### Mock Services

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pandrs::testing::*;
    
    #[tokio::test]
    async fn test_database_operations() {
        // Use in-memory SQLite for testing
        let connector = DatabaseConnectorFactory::sqlite();
        let config = DatabaseConfig::new("sqlite::memory:");
        connector.connect(&config).await.unwrap();
        
        // Test operations
        connector.execute("CREATE TABLE test (id INTEGER, name TEXT)").await.unwrap();
        // ... test code
    }
    
    #[tokio::test]
    async fn test_cloud_operations() {
        // Use mock cloud connector
        let mock_connector = MockCloudConnector::new();
        mock_connector.expect_read_dataframe()
            .returning(|_, _, _| Ok(DataFrame::new()));
        
        // Test operations
        let df = mock_connector.read_dataframe("bucket", "key", FileFormat::CSV).await.unwrap();
        // ... test code
    }
}
```

### Integration Testing

```bash
# Start test services
docker run -d --name test-postgres -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:14
docker run -d --name test-minio -p 9000:9000 minio/minio server /data

# Run integration tests
cargo test --features integration-tests

# Cleanup
docker rm -f test-postgres test-minio
```

---

## 📊 Monitoring and Observability

### Performance Metrics

```rust
use pandrs::metrics::*;

// Enable metrics collection
let config = PandRSConfig::new()
    .with_metrics(MetricsConfig::enabled());

// Collect custom metrics
let timer = metrics::start_timer("dataframe_operation");
let result = df.complex_operation()?;
timer.observe_duration();

// Built-in metrics
// - Connection pool usage
// - Query execution times  
// - Memory usage
// - Error rates
// - Throughput statistics
```

### Health Checks

```rust
use pandrs::health::*;

// Check all connector health
let health_report = HealthChecker::check_all().await;
for (connector, status) in health_report {
    match status {
        HealthStatus::Healthy => println!("{} is healthy", connector),
        HealthStatus::Degraded(msg) => println!("{} is degraded: {}", connector, msg),
        HealthStatus::Unhealthy(err) => println!("{} is down: {}", connector, err),
    }
}
```

---

## 🚨 Error Handling and Troubleshooting

### Common Error Patterns

```rust
use pandrs::core::error::*;

match result {
    Err(Error::ConnectionError(msg)) => {
        // Retry with backoff
        tokio::time::sleep(Duration::from_secs(1)).await;
        retry_operation()?;
    },
    Err(Error::AuthenticationError(msg)) => {
        // Check credentials
        verify_credentials().await?;
    },
    Err(Error::TimeoutError(msg)) => {
        // Increase timeout or optimize query
        let config = config.with_timeout(60);
    },
    _ => return Err(error),
}
```

### Diagnostic Tools

```rust
use pandrs::diagnostics::*;

// Connection diagnostics
let diag = ConnectionDiagnostics::run(&connector).await;
println!("Latency: {}ms", diag.latency_ms);
println!("Throughput: {} MB/s", diag.throughput_mbps);

// Query performance analysis
let analysis = QueryAnalysis::profile(&query).await;
println!("Execution plan: {}", analysis.plan);
println!("Bottlenecks: {:?}", analysis.bottlenecks);
```

---

## 📈 Performance Benchmarks

### Typical Performance Characteristics

#### Database Operations
- **Query Latency**: 10-50ms for simple queries
- **Throughput**: 1000-5000 rows/second for inserts
- **Connection Pool**: 90%+ utilization efficiency

#### Cloud Storage
- **S3 Read Throughput**: 100-500 MB/s (depending on instance type)
- **Parquet vs CSV**: 3-5x faster reading Parquet for analytical workloads
- **Compression Ratio**: 70-90% size reduction with Parquet+ZSTD

#### Arrow Integration  
- **Conversion Overhead**: <1ms for datasets under 1M rows
- **Memory Usage**: 60-80% reduction vs row-based formats
- **SIMD Speedup**: 2-10x performance improvement for numerical operations

### Benchmarking Your Workload

```rust
use pandrs::benchmark::*;

// Benchmark database operations
let benchmark = DatabaseBenchmark::new(&connector);
let results = benchmark.run_suite().await;
println!("Query latency p95: {}ms", results.query_latency_p95);

// Benchmark I/O operations
let io_benchmark = IOBenchmark::new();
let results = io_benchmark.compare_formats(&df).await;
println!("CSV read: {}ms, Parquet read: {}ms", results.csv_ms, results.parquet_ms);
```

---

## 🤝 Contributing and Community

### Getting Involved

- **GitHub Repository**: https://github.com/cool-japan/pandrs
- **Documentation**: https://pandrs.dev/docs
- **Discord/Slack**: [Community chat link]
- **Issue Tracker**: Report bugs and request features

### Development Setup

```bash
# Clone repository
git clone https://github.com/cool-japan/pandrs.git
cd pandrs

# Install dependencies
cargo build --all-features

# Run tests
cargo test --all-features

# Run examples
cargo run --example ecosystem_integration_demo --features distributed
```

### Contributing Guidelines

1. **Fork and Branch**: Create feature branches from `main`
2. **Tests Required**: All new features must include tests
3. **Documentation**: Update docs for new APIs
4. **Performance**: Benchmark performance-critical changes
5. **Code Review**: All changes require review

---

## 📚 Additional Resources

### API Documentation
- [Full API Reference](https://docs.rs/pandrs)
- [Connector API](https://docs.rs/pandrs/latest/pandrs/connectors/)
- [Error Handling Guide](https://docs.rs/pandrs/latest/pandrs/core/error/)

### Examples and Tutorials
- [Getting Started Tutorial](../examples/getting_started.rs)
- [Database Integration Example](../examples/database_integration_example.rs)
- [Cloud Storage Demo](../examples/ecosystem_integration_demo.rs)
- [Python Integration Tutorial](../py_bindings/examples/pandrs_tutorial.ipynb)

### Performance Guides
- [Optimization Best Practices](PERFORMANCE_OPTIMIZATION.md)
- [Memory Management Guide](MEMORY_GUIDE.md)
- [Benchmarking Methodology](BENCHMARKING.md)

---

*This guide covers the major ecosystem integration features in PandRS. For the latest updates and additional examples, visit the [official documentation](https://pandrs.dev/docs).*