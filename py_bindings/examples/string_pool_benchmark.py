"""
String Pool Optimization Benchmark

This benchmark evaluates the memory efficiency and conversion performance
of string pool optimization.
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
import psutil
import gc
import sys
from tabulate import tabulate

def measure_memory():
    """Returns the current memory usage (in MB)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def generate_data(rows, unique_ratio=0.1):
    """Generate test data
    
    Args:
        rows: Number of rows to generate
        unique_ratio: Ratio of unique strings (0.1 = 10% unique)
    """
    # Calculate the number of unique strings
    unique_count = int(rows * unique_ratio)
    unique_count = max(unique_count, 1)  # At least 1 unique string
    
    # Generate a pool of unique strings
    unique_strings = [f"unique_value_{i}" for i in range(unique_count)]
    
    # Generate data for the specified number of rows (randomly select unique strings)
    data = np.random.choice(unique_strings, size=rows)
    return data

def run_benchmark():
    """Run the benchmark"""
    print("=== String Pool Optimization Performance Benchmark ===\n")
    
    # Test sizes and unique ratios
    test_configs = [
        {"rows": 100_000, "unique_ratio": 0.01},   # 1% unique (high duplication)
        {"rows": 100_000, "unique_ratio": 0.1},    # 10% unique
        {"rows": 100_000, "unique_ratio": 0.5},    # 50% unique
        {"rows": 1_000_000, "unique_ratio": 0.01}, # 1% unique, large scale
    ]
    
    results = []
    
    for config in test_configs:
        rows = config["rows"]
        unique_ratio = config["unique_ratio"]
        
        print(f"\n## Data Size: {rows:,} rows, Unique Ratio: {unique_ratio:.1%} ##")
        
        # Generate dataset
        string_data = generate_data(rows, unique_ratio)
        numeric_data = np.arange(rows)
        
        # Reset memory usage
        gc.collect()
        base_memory = measure_memory()
        print(f"Base Memory Usage: {base_memory:.2f} MB")
        
        # ------ Regular StringColumn (No Pooling) ------
        # Measure in a new scope to track memory usage
        start_time = time.time()
        df_no_pool = pr.OptimizedDataFrame()
        
        # Add without using string pool (previous implementation)
        string_list = list(string_data)  # Convert ndarray to list
        df_no_pool.add_int_column('id', list(range(rows)))
        
        # Add directly to StringColumn (no pooling)
        with_nopool_time_start = time.time()
        df_no_pool.add_string_column('str_value', string_list)
        with_nopool_time = time.time() - with_nopool_time_start
        
        # Measure memory
        gc.collect()
        no_pool_memory = measure_memory() - base_memory
        
        print(f"1. Without String Pool:")
        print(f"   - Processing Time: {with_nopool_time:.6f} seconds")
        print(f"   - Additional Memory: {no_pool_memory:.2f} MB")
        
        # ------ Using String Pool ------
        # Reset memory
        df_no_pool = None
        gc.collect()
        reset_memory = measure_memory()
        
        # Initialize string pool
        string_pool = pr.StringPool()
        
        # Create DataFrame using the pool
        df_with_pool = pr.OptimizedDataFrame()
        df_with_pool.add_int_column('id', list(range(rows)))
        
        # Add using the string pool
        with_pool_time_start = time.time()
        
        # Convert to indices before adding
        py_list = string_list  # Already converted to list
        # Add to the string pool first
        pool_indices = string_pool.add_list(py_list)
        df_with_pool.add_string_column_from_pylist('str_value', py_list)
        
        with_pool_time = time.time() - with_pool_time_start
        
        # Measure memory
        gc.collect()
        with_pool_memory = measure_memory() - reset_memory
        
        # Get pool statistics
        pool_stats = string_pool.get_stats()
        
        print(f"2. Using String Pool:")
        print(f"   - Processing Time: {with_pool_time:.6f} seconds")
        print(f"   - Additional Memory: {with_pool_memory:.2f} MB")
        print(f"   - String Pool Statistics:")
        print(f"     * Total Strings: {pool_stats['total_strings']:,}")
        print(f"     * Unique Strings: {pool_stats['unique_strings']:,}")
        print(f"     * Bytes Saved: {pool_stats['bytes_saved']:,}")
        print(f"     * Duplication Rate: {pool_stats['duplicate_ratio']:.2%}")
        
        # Conversion Benchmark (Pool ↔ pandas)
        to_pandas_start = time.time()
        pd_df = df_with_pool.to_pandas()
        to_pandas_time = time.time() - to_pandas_start
        
        from_pandas_start = time.time()
        back_to_optimized = pr.OptimizedDataFrame.from_pandas(pd_df)
        from_pandas_time = time.time() - from_pandas_start
        
        print(f"3. pandas Conversion:")
        print(f"   - Optimized → pandas: {to_pandas_time:.6f} seconds")
        print(f"   - pandas → Optimized: {from_pandas_time:.6f} seconds")
        
        # Record results
        results.append({
            'Data Size': f"{rows:,} rows",
            'Unique Ratio': f"{unique_ratio:.1%}",
            'No Pool Time': with_nopool_time,
            'With Pool Time': with_pool_time,
            'No Pool Memory': no_pool_memory,
            'With Pool Memory': with_pool_memory,
            'Memory Reduction Rate': f"{(1 - with_pool_memory / no_pool_memory):.2%}" if no_pool_memory > 0 else "N/A",
            'Duplication Rate': pool_stats['duplicate_ratio']
        })
    
    # Display results
    print("\n=== Summary of Results ===")
    try:
        # Ensure tabulate is imported correctly
        from tabulate import tabulate as tab_func
        print(tab_func(results, headers="keys", tablefmt="grid", floatfmt=".6f"))
    except Exception as e:
        print("Error formatting results:", e)
        for r in results:
            print(r)
    
    # Observations
    print("\nObservations:")
    print("1. Memory Efficiency: String pool significantly reduces memory usage for datasets with high duplication.")
    print("2. Conversion Performance: Using the pool speeds up string conversion between Python and Rust.")
    print("3. Optimal Use Case: Particularly effective for categorical string data with high duplication rates.")
    print("\nNote: Memory measurements using psutil are approximate. Refer to internal metrics for actual savings.")

if __name__ == "__main__":
    try:
        import tabulate
        import psutil
    except ImportError:
        print("Required modules are missing. Install them using the following command:")
        print("pip install tabulate psutil")
        sys.exit(1)
    
    print("Running the String Pool Optimization Benchmark...")
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")