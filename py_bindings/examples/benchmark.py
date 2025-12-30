"""
PandRS vs pandas Performance Benchmark
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import sys

# Disable displaying binary or other outputs in the standard output
plt.ioff()

def run_benchmark(test_name, rows, pandas_func, pandrs_func):
    """Run benchmark"""
    # Timing for pandas
    start = time.time()
    pandas_result = pandas_func(rows)
    pandas_time = time.time() - start
    
    # Timing for pandrs
    start = time.time()
    pandrs_result = pandrs_func(rows)
    pandrs_time = time.time() - start
    
    # Calculate ratio
    ratio = pandas_time / pandrs_time if pandrs_time > 0 else float('inf')
    
    return {
        'Test': test_name,
        'Rows': rows,
        'pandas (seconds)': pandas_time,
        'pandrs (seconds)': pandrs_time,
        'Ratio (pandas/pandrs)': ratio
    }

def main():
    print("=== PandRS vs pandas Performance Benchmark ===\n")
    
    # List of row sizes
    row_sizes = [10, 100, 1000, 10000, 100000]
    results = []
    
    # DataFrame creation benchmark
    for rows in row_sizes:
        # Prepare benchmark data
        numeric_data = list(range(rows))
        string_data = [f"value_{i}" for i in range(rows)]
        float_data = [i * 1.1 for i in range(rows)]
        
        # pandas DataFrame creation
        def pandas_create(n):
            data = {
                'A': numeric_data,
                'B': string_data,
                'C': float_data
            }
            return pd.DataFrame(data)
        
        # pandrs DataFrame creation
        def pandrs_create(n):
            data = {
                'A': numeric_data,
                'B': string_data,
                'C': float_data
            }
            return pr.DataFrame(data)
        
        result = run_benchmark(f"DataFrame Creation", rows, pandas_create, pandrs_create)
        results.append(result)
    
    # Column access benchmark (for 100,000 rows)
    rows = 100000
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    
    # Preprocessed DataFrames
    pd_df = pd.DataFrame({
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    })
    
    pr_df = pr.DataFrame({
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    })
    
    # Column access benchmark
    def pandas_column_access(n):
        col = pd_df['A']
        return col
    
    def pandrs_column_access(n):
        col = pr_df['A']
        return col
    
    result = run_benchmark(f"Column Access", rows, pandas_column_access, pandrs_column_access)
    results.append(result)
    
    # Data conversion benchmark
    def pandas_to_dict(n):
        dict_data = pd_df.to_dict()
        return dict_data
    
    def pandrs_to_dict(n):
        dict_data = pr_df.to_dict()
        return dict_data
    
    result = run_benchmark(f"to_dict Conversion", rows, pandas_to_dict, pandrs_to_dict)
    results.append(result)
    
    # Interconversion benchmark
    def pandas_to_pandrs(n):
        return pr.DataFrame.from_pandas(pd_df)
    
    def pandrs_to_pandas(n):
        return pr_df.to_pandas()
    
    results.append(run_benchmark(f"pandas → pandrs", rows, pandas_to_pandrs, lambda x: None))
    results.append(run_benchmark(f"pandrs → pandas", rows, lambda x: None, pandrs_to_pandas))
    
    # Display results
    print(tabulate(results, headers='keys', tablefmt='pretty', floatfmt='.6f'))
    
    # Visualize results
    data_creation_results = [r for r in results if r['Test'] == 'DataFrame Creation']
    
    pandas_times = [r['pandas (seconds)'] for r in data_creation_results]
    pandrs_times = [r['pandrs (seconds)'] for r in data_creation_results]
    row_labels = [str(r['Rows']) for r in data_creation_results]
    
    x = np.arange(len(row_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pandas_times, width, label='pandas')
    rects2 = ax.bar(x + width/2, pandrs_times, width, label='pandrs')
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title('DataFrame Creation - pandas vs pandrs')
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels)
    ax.set_xlabel('Rows')
    ax.legend()
    
    # Change to log scale (due to large differences in size)
    ax.set_yscale('log')
    
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{pandas_times[i]:.4f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', rotation=90)
    
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{pandrs_times[i]:.4f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', rotation=90)
    
    fig.tight_layout()
    plt.savefig('benchmark_results.png')
    
    print("\nSaved result graph to 'benchmark_results.png'")
    print("\nComparison between Rust native version and Python bindings")
    print("Time to create a 100,000-row DataFrame in Rust native version: approximately 0.05 seconds")
    print("Time in Python bindings: see above")
    print("The difference is mainly due to the overhead of data conversion between Python and Rust.")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("The tabulate module is not installed. Please install it with pip install tabulate.")
        sys.exit(1)
        
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Please install it with pip install matplotlib.")
        sys.exit(1)
        
    main()