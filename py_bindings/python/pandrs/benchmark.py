"""
PandRS Performance Benchmark Module
"""
import time
import sys

try:
    import pandrs as pr
except ImportError:
    print("Cannot find pandrs module. Please check if it is installed correctly.")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Cannot find pandas module. Please install with pip install pandas.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Cannot find numpy module. Please install with pip install numpy.")
    sys.exit(1)

def run_benchmark(name, rows, pandas_func, pandrs_func):
    """
    Run benchmark for specified functions and return results.
    
    Args:
        name: Benchmark name
        rows: Number of data rows
        pandas_func: Pandas benchmark function
        pandrs_func: PandRS benchmark function
        
    Returns:
        Dictionary containing results
    """
    print(f"\nRunning: {name} ({rows:,} rows)...")
    
    # Pandas measurement
    start = time.time()
    pandas_result = pandas_func()
    pandas_time = time.time() - start
    print(f"  pandas: {pandas_time:.6f} seconds")
    
    # PandRS measurement
    start = time.time()
    pandrs_result = pandrs_func()
    pandrs_time = time.time() - start
    print(f"  pandrs: {pandrs_time:.6f} seconds")
    
    # Ratio
    if pandrs_time > 0:
        ratio = pandas_time / pandrs_time
        relative = "faster" if ratio > 1 else "slower"
        print(f"  Ratio: pandas/pandrs = {ratio:.2f}x ({abs(ratio-1):.2f}x {relative} than pandas)")
    else:
        ratio = float('inf')
        print("  Ratio: cannot calculate (pandrs execution time is 0)")
    
    return {
        'name': name,
        'rows': rows,
        'pandas_time': pandas_time,
        'pandrs_time': pandrs_time,
        'ratio': ratio
    }

def dataframe_creation_benchmark(rows=10000):
    """
    DataFrame creation benchmark
    """
    # Test data
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # Pandas function
    def pandas_create():
        return pd.DataFrame(data)
    
    # PandRS function
    def pandrs_create():
        return pr.DataFrame(data)
    
    return run_benchmark("DataFrame Creation", rows, pandas_create, pandrs_create)

def column_access_benchmark(rows=10000):
    """
    Column access benchmark
    """
    # Test data
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # Pre-creation
    pd_df = pd.DataFrame(data)
    pr_df = pr.DataFrame(data)
    
    # Pandas function
    def pandas_access():
        return pd_df['A']
    
    # PandRS function
    def pandrs_access():
        return pr_df['A']
    
    return run_benchmark("Column Access", rows, pandas_access, pandrs_access)

def conversion_benchmark(rows=10000):
    """
    Conversion functionality benchmark
    """
    # Test data
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # Pre-creation
    pd_df = pd.DataFrame(data)
    pr_df = pr.DataFrame(data)
    
    # Pandas function
    def pandas_to_dict():
        return pd_df.to_dict()
    
    # PandRS function
    def pandrs_to_dict():
        return pr_df.to_dict()
    
    return run_benchmark("to_dict Conversion", rows, pandas_to_dict, pandrs_to_dict)

def interop_benchmark(rows=10000):
    """
    Interoperability benchmark
    """
    # Test data
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # Pre-creation
    pd_df = pd.DataFrame(data)
    pr_df = pr.DataFrame(data)
    
    # Pandas→PandRS function
    def pandas_to_pandrs():
        return pr.DataFrame.from_pandas(pd_df)
    
    # Dummy function
    def dummy():
        pass
    
    result1 = run_benchmark("pandas→pandrs Conversion", rows, pandas_to_pandrs, dummy)
    
    # PandRS→Pandas function
    def pandrs_to_pandas():
        return pr_df.to_pandas()
    
    result2 = run_benchmark("pandrs→pandas Conversion", rows, dummy, pandrs_to_pandas)
    
    return [result1, result2]

def run_all_benchmarks():
    """
    Run all benchmarks
    """
    print("=== PandRS vs pandas Performance Benchmark ===")
    
    results = []
    
    # Test DataFrame creation with various sizes
    for rows in [10, 100, 1000, 10000, 100000]:
        results.append(dataframe_creation_benchmark(rows))
    
    # Other benchmarks
    results.append(column_access_benchmark(10000))
    results.append(conversion_benchmark(10000))
    interop_results = interop_benchmark(10000)
    results.extend(interop_results)
    
    # Display result summary
    print("\n=== Benchmark Results Summary ===")
    print(f"{'Test Name':<25} {'Rows':>10} {'pandas(sec)':>12} {'pandrs(sec)':>12} {'Ratio':>10}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['name']:<25} {r['rows']:>10,} {r['pandas_time']:>12.6f} {r['pandrs_time']:>12.6f} {r['ratio']:>10.2f}x")
    
    print("\nNote: Ratio is pandas/pandrs. Values greater than 1.0 mean pandrs is faster.")
    print("Performance is reduced compared to pure Rust implementation due to Python-Rust data conversion overhead.")
    print("Pure Rust implementation of pandrs can complete DataFrame creation with 100k rows in about 50ms.")

if __name__ == "__main__":
    run_all_benchmarks()