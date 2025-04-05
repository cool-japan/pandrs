"""
Basic usage example for the PandRS Python bindings
"""

import pandrs as pr
import numpy as np
import pandas as pd
import time

def main():
    print(f"PandRS version: {getattr(pr, '__version__', '0.1.0')}")
    print("\n=== Creating DataFrame ===")
    
    # Create a DataFrame
    df = pr.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n=== Column Access ===")
    series_a = df['A']
    print(f"Series A: {series_a}")
    
    print("\n=== NumPy Conversion ===")
    np_array = series_a.to_numpy()
    print(f"NumPy array: {np_array}")
    print(f"Type: {type(np_array)}")
    
    print("\n=== pandas Interoperability ===")
    # Convert to pandas DataFrame
    pd_df = df.to_pandas()
    print("Pandas DataFrame:")
    print(pd_df)
    
    # Convert back to PandRS
    pr_df = pr.DataFrame.from_pandas(pd_df)
    print("\nBack to PandRS DataFrame:")
    print(pr_df)
    
    print("\n=== CSV I/O ===")
    # Save to CSV
    df.to_csv("sample_data.csv")
    print("Saved to sample_data.csv")
    
    # Read from CSV
    df_from_csv = pr.DataFrame.read_csv("sample_data.csv")
    print("\nLoaded from CSV:")
    print(df_from_csv)
    
    print("\n=== JSON I/O ===")
    # Convert to JSON
    json_str = df.to_json()
    print(f"JSON: {json_str[:100]}...")
    
    # Read from JSON
    df_from_json = pr.DataFrame.read_json(json_str)
    print("\nLoaded from JSON:")
    print(df_from_json)
    
    print("\n=== Performance Comparison ===")
    # Create a larger DataFrame for performance testing
    n_rows = 100000
    data = {
        'A': list(range(n_rows)),
        'B': [f"value_{i}" for i in range(n_rows)],
        'C': [i * 1.1 for i in range(n_rows)]
    }
    
    # Time pandas DataFrame creation
    start = time.time()
    pd_df = pd.DataFrame(data)
    pd_time = time.time() - start
    print(f"pandas DataFrame creation time: {pd_time:.4f} seconds")
    
    # Time PandRS DataFrame creation
    start = time.time()
    pr_df = pr.DataFrame(data)
    pr_time = time.time() - start
    print(f"PandRS DataFrame creation time: {pr_time:.4f} seconds")
    
    print(f"Speed ratio: {pd_time / pr_time:.2f}x")
    
if __name__ == "__main__":
    main()