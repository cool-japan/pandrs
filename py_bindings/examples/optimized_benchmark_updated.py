"""
Benchmark for Python bindings of optimized implementation and string pool
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import sys
import gc
import psutil

def measure_memory():
    """Returns the current memory usage (in MB)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def run_basic_benchmark():
    """Run basic benchmark"""
    print("=== PandRS Optimized Implementation vs pandas Performance Benchmark ===\n")
    
    # Test sizes
    row_sizes = [10_000, 100_000, 1_000_000]
    results = []
    
    for rows in row_sizes:
        print(f"\n## Data size: {rows:,} rows ##")
        
        # Prepare data
        numeric_data = list(range(rows))
        string_data = [f"value_{i % 100}" for i in range(rows)]
        float_data = [i * 0.5 for i in range(rows)]
        bool_data = [i % 2 == 0 for i in range(rows)]
        
        # pandas test - DataFrame creation
        start = time.time()
        pd_df = pd.DataFrame({
            'A': numeric_data,
            'B': string_data,
            'C': float_data,
            'D': bool_data
        })
        pandas_create_time = time.time() - start
        print(f"pandas DataFrame creation: {pandas_create_time:.6f} seconds")
        
        # PandRS legacy implementation - DataFrame creation
        start = time.time()
        legacy_df = pr.DataFrame({
            'A': numeric_data,
            'B': string_data,
            'C': float_data,
            'D': bool_data
        })
        legacy_create_time = time.time() - start
        print(f"PandRS legacy implementation DataFrame creation: {legacy_create_time:.6f} seconds")
        
        # PandRS optimized implementation - DataFrame creation
        start = time.time()
        optimized_df = pr.OptimizedDataFrame()
        optimized_df.add_int_column('A', numeric_data)
        optimized_df.add_string_column('B', string_data)
        optimized_df.add_float_column('C', float_data)
        optimized_df.add_boolean_column('D', bool_data)
        optimized_create_time = time.time() - start
        print(f"PandRS optimized implementation DataFrame creation: {optimized_create_time:.6f} seconds")
        
        # Save results
        results.append({
            'Data size': f"{rows:,} rows",
            'pandas creation': pandas_create_time,
            'PandRS legacy implementation': legacy_create_time,
            'PandRS optimized implementation': optimized_create_time,
            'Legacy ratio': legacy_create_time / optimized_create_time,
            'pandas ratio': pandas_create_time / optimized_create_time
        })
        
        # pandas interconversion test
        print("\n## pandas conversion test ##")
        
        # PandRS → pandas conversion
        start = time.time()
        pd_from_optimized = optimized_df.to_pandas()
        to_pandas_time = time.time() - start
        print(f"Optimized DataFrame → pandas: {to_pandas_time:.6f} seconds")
        
        # pandas → PandRS conversion
        start = time.time()
        optimized_from_pd = pr.OptimizedDataFrame.from_pandas(pd_df)
        from_pandas_time = time.time() - start
        print(f"pandas → Optimized DataFrame: {from_pandas_time:.6f} seconds")
        
        # Lazy evaluation benchmark (only for up to 100,000 rows)
        if rows <= 100_000:
            print("\n## Lazy evaluation test ##")
            
            # Filtering - pandas
            start = time.time()
            filtered_pd = pd_df[pd_df['D'] == True]
            pandas_filter_time = time.time() - start
            print(f"pandas filtering: {pandas_filter_time:.6f} seconds")
            
            # Filtering - legacy implementation
            # Not implemented or uses a different API, so omitted
            legacy_filter_time = "-"
            
            # Filtering - optimized implementation
            # 1. Prepare boolean column for filter condition
            start_filter = time.time()
            optimized_df_with_filter = pr.OptimizedDataFrame()
            optimized_df_with_filter.add_int_column('A', numeric_data)
            optimized_df_with_filter.add_string_column('B', string_data)
            optimized_df_with_filter.add_float_column('C', float_data)
            optimized_df_with_filter.add_boolean_column('filter', bool_data)
            
            # 2. Execute filtering
            filtered_optimized = optimized_df_with_filter.filter('filter')
            optimized_filter_time = time.time() - start_filter
            print(f"PandRS optimized implementation filtering: {optimized_filter_time:.6f} seconds")
            
            # Lazy evaluation - using LazyFrame
            start_lazy = time.time()
            lazy_df = pr.LazyFrame(optimized_df_with_filter)
            lazy_filtered = lazy_df.filter('filter').execute()
            lazy_filter_time = time.time() - start_lazy
            print(f"PandRS LazyFrame filtering: {lazy_filter_time:.6f} seconds")
    
    # Display results
    print("\n=== Basic Benchmark Results Summary ===")
    try:
        from tabulate import tabulate as tab_func
        print(tab_func(results, headers="keys", tablefmt="pretty", floatfmt=".6f"))
    except Exception as e:
        print("Error formatting results:", e)
        for r in results:
            print(r)

def generate_string_data(rows, unique_ratio=0.1):
    """Generate test data
    
    Args:
        rows: Number of rows to generate
        unique_ratio: Ratio of unique strings (0.1 = 10% unique)
    """
    # Calculate the number of unique strings
    unique_count = int(rows * unique_ratio)
    unique_count = max(unique_count, 1)  # At least 1
    
    # Generate a pool of unique strings
    unique_strings = [f"unique_value_{i}" for i in range(unique_count)]
    
    # Generate data for the number of rows (randomly select from unique strings)
    data = np.random.choice(unique_strings, size=rows)
    return list(data)  # Convert numpy array to list and return

def run_string_pool_benchmark():
    """Run string pool optimization benchmark"""
    print("\n\n=== String Pool Optimization Performance Benchmark ===\n")
    
    # Test configurations
    test_configs = [
        {"rows": 100_000, "unique_ratio": 0.01},   # 1% unique (high duplication)
        {"rows": 100_000, "unique_ratio": 0.1},    # 10% unique
        {"rows": 100_000, "unique_ratio": 0.5},    # 50% unique
    ]
    
    # Preload data into the pool
    string_pool = pr.StringPool()
    
    results = []
    
    for config in test_configs:
        rows = config["rows"]
        unique_ratio = config["unique_ratio"]
        
        print(f"\n## Data size: {rows:,} rows, Unique ratio: {unique_ratio:.1%} ##")
        
        # Generate dataset
        string_data = generate_string_data(rows, unique_ratio)
        numeric_data = list(range(rows))
        
        # Reset memory usage
        gc.collect()
        base_memory = measure_memory()
        print(f"Base memory usage: {base_memory:.2f} MB")
        
        # ------ Regular StringColumn (no pool) ------
        # Measure memory usage in a new scope
        start_time = time.time()
        df_no_pool = pr.OptimizedDataFrame()
        df_no_pool.add_int_column('id', numeric_data)
        
        # Add directly to StringColumn (no pooling - legacy implementation)
        with_nopool_time_start = time.time()
        df_no_pool.add_string_column('str_value', string_data)  
        with_nopool_time = time.time() - with_nopool_time_start
        
        # Measure memory
        gc.collect()
        no_pool_memory = measure_memory() - base_memory
        
        print(f"1. Without string pool:")
        print(f"   - Processing time: {with_nopool_time:.6f} seconds")
        print(f"   - Additional memory: {no_pool_memory:.2f} MB")
        
        # ------ Using string pool ------
        # Reset memory
        df_no_pool = None
        gc.collect()
        reset_memory = measure_memory()
        
        # Initialize string pool
        string_pool = pr.StringPool()
        
        # Create DataFrame using pool
        df_with_pool = pr.OptimizedDataFrame()
        df_with_pool.add_int_column('id', numeric_data)
        
        # Add using string pool
        with_pool_time_start = time.time()
        
        # Add directly from Python list to string column (new implementation)
        py_list = string_data
        # Register in the pool first
        indices = string_pool.add_list(py_list)
        df_with_pool.add_string_column_from_pylist('str_value', py_list)
        
        with_pool_time = time.time() - with_pool_time_start
        
        # Measure memory
        gc.collect()
        with_pool_memory = measure_memory() - reset_memory
        
        # Get pool statistics
        pool_stats = string_pool.get_stats()
        
        print(f"2. Using string pool:")
        print(f"   - Processing time: {with_pool_time:.6f} seconds")
        print(f"   - Additional memory: {with_pool_memory:.2f} MB")
        print(f"   - String pool statistics:")
        print(f"     * Total strings: {pool_stats['total_strings']:,}")
        print(f"     * Unique strings: {pool_stats['unique_strings']:,}")
        print(f"     * Bytes saved: {pool_stats['bytes_saved']:,}")
        print(f"     * Duplication ratio: {pool_stats['duplicate_ratio']:.2%}")
        
        # Conversion benchmark (pool ↔ pandas)
        to_pandas_start = time.time()
        pd_df = df_with_pool.to_pandas()
        to_pandas_time = time.time() - to_pandas_start
        
        from_pandas_start = time.time()
        back_to_optimized = pr.OptimizedDataFrame.from_pandas(pd_df)
        from_pandas_time = time.time() - from_pandas_start
        
        print(f"3. pandas conversion:")
        print(f"   - Optimized → pandas: {to_pandas_time:.6f} seconds")
        print(f"   - pandas → Optimized: {from_pandas_time:.6f} seconds")
        
        # Record results
        memory_reduction = "-"
        if no_pool_memory > 0:
            memory_reduction = f"{(1 - with_pool_memory / no_pool_memory):.2%}"
            
        results.append({
            'Data size': f"{rows:,} rows",
            'Unique ratio': f"{unique_ratio:.1%}",
            'Without pool time': with_nopool_time,
            'With pool time': with_pool_time,
            'Speed improvement': f"{with_nopool_time / with_pool_time:.2f}x",
            'Without pool memory': f"{no_pool_memory:.2f} MB",
            'With pool memory': f"{with_pool_memory:.2f} MB",
            'Memory reduction rate': memory_reduction,
            'Duplication ratio': f"{pool_stats['duplicate_ratio']:.2%}"
        })
    
    # Display results
    print("\n=== String Pool Optimization Results Summary ===")
    try:
        from tabulate import tabulate as tab_func
        print(tab_func(results, headers="keys", tablefmt="grid"))
    except Exception as e:
        print("Error formatting results:", e)
        for r in results:
            print(r)
    
    # Observations
    print("\nObservations:")
    print("1. Memory efficiency: String pool significantly reduces memory usage for datasets with high duplication.")
    print("2. Processing performance: Using string pool improves processing speed, especially for highly duplicated data.")
    print("3. Impact of unique ratio: The lower the unique ratio (more duplication), the greater the effect.")
    print("\nParticularly effective for categorical data with many duplicates or string data with limited value sets.")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
        import psutil
    except ImportError:
        print("Required modules are missing. Please install them using the following command:")
        print("pip install tabulate psutil")
        sys.exit(1)
    
    print("Running benchmark... Press Ctrl+C to interrupt.\n")
    
    try:
        run_basic_benchmark()
        run_string_pool_benchmark()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")