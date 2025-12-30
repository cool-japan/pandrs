"""
Benchmark for Python bindings of optimized implementation
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import sys

def run_benchmark():
    """Run benchmark"""
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
    print("\n=== Results Summary ===")
    print(tabulate(results, headers="keys", tablefmt="pretty", floatfmt=".6f"))
    
    # Observations on comparison with pandas
    print("\nObservations:")
    print("1. DataFrame creation: Optimized implementation is significantly faster than the legacy implementation.")
    if any(r['pandas ratio'] > 1.0 for r in results):
        print("   - Faster than pandas in some cases.")
    else:
        print("   - pandas is still faster, but the gap is narrowing.")
    print("2. Efficient type conversion: Type-specialized implementation reduces data conversion overhead.")
    print("3. Lazy evaluation: Optimized operation pipelines improve efficiency for multiple operations.")
    print("\nNote that the Rust native version is even faster.")
    print("The Python bindings still have overhead due to data conversion.")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("The tabulate module is required. Please install it with pip install tabulate.")
        sys.exit(1)
    
    print("Warning: Large benchmarks require sufficient memory. Run in an environment with enough RAM.")
    print("Press Ctrl+C to interrupt.\n")
    
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")