"""
PandRS vs pandas 1 Million Rows Benchmark
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
import sys

def run_benchmark():
    """Benchmark for creating a DataFrame with 1 million rows"""
    print("=== PandRS vs pandas 1 Million Rows Benchmark ===\n")
    
    # Prepare benchmark data
    rows = 1_000_000
    print(f"Preparing data: {rows:,} rows x 3 columns...")
    
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # Measure pandas performance
    print("\n--- Creating pandas DataFrame ---")
    start = time.time()
    pd_df = pd.DataFrame(data)
    pandas_time = time.time() - start
    print(f"pandas DataFrame creation time: {pandas_time:.6f} seconds")
    
    # Check memory usage (approximate)
    pd_memory = pd_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"pandas DataFrame approximate memory usage: {pd_memory:.2f} MB")
    
    # Measure PandRS performance
    print("\n--- Creating PandRS DataFrame ---")
    start = time.time()
    pr_df = pr.DataFrame(data)
    pandrs_time = time.time() - start
    print(f"PandRS DataFrame creation time: {pandrs_time:.6f} seconds")
    
    # Calculate ratio
    ratio = pandas_time / pandrs_time if pandrs_time > 0 else float('inf')
    if ratio > 1:
        print(f"PandRS is {ratio:.2f} times faster than pandas")
    else:
        print(f"PandRS is {1/ratio:.2f} times slower than pandas")
    
    # Summary report
    print("\n=== Benchmark Summary ===")
    print(f"Data size: {rows:,} rows x 3 columns")
    print(f"pandas DataFrame creation time: {pandas_time:.6f} seconds")
    print(f"PandRS DataFrame creation time: {pandrs_time:.6f} seconds")
    print(f"pandas/PandRS ratio: {ratio:.2f}x")
    
    print("\nNote: The native Rust version of PandRS is even faster, completing the same operation in a few hundred milliseconds.")
    print("The main performance difference is due to the overhead of the Python bindings.")
    
if __name__ == "__main__":
    print("Warning: This benchmark uses a large amount of memory. Ensure sufficient RAM is available.")
    
    try:
        run_benchmark()
    except MemoryError:
        print("Error: Execution failed due to insufficient memory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)