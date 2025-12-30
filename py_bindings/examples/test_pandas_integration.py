#!/usr/bin/env python3
"""
Basic tests for Python bindings
"""

import os
import sys
import numpy as np
import pandas as pd

# Add path to make imports possible
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import pandrs as pr
    print(f"PandRS version: {pr.__version__}")
except ImportError:
    print("Cannot import the PandRS module. Please run 'cd py_bindings && pip install -e .' first.")
    sys.exit(1)

def test_dataframe_creation():
    """Test for DataFrame creation"""
    print("\n=== DataFrame Creation Test ===")
    
    # Prepare data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    
    # Create DataFrame
    df = pr.DataFrame(data)
    print(f"Created DataFrame:\n{df}")
    print(f"Shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    # Compatibility with Pandas
    pd_df = df.to_pandas()
    print(f"\nConverted to Pandas DataFrame:\n{pd_df}")
    
    # Access column
    series_a = df['A']
    print(f"\nAccessed column A: {series_a}")
    
    return df

def test_io_operations(df):
    """Test for input/output operations"""
    print("\n=== Input/Output Operations Test ===")
    
    # Save to CSV
    csv_path = "test_dataframe.csv"
    df.to_csv(csv_path)
    print(f"Saved as CSV: {csv_path}")
    
    # Load from CSV
    df_loaded = pr.DataFrame.read_csv(csv_path)
    print(f"DataFrame loaded from CSV:\n{df_loaded}")
    
    # Convert to JSON
    json_str = df.to_json()
    print(f"JSON string: {json_str[:100]}...")
    
    # Load from JSON
    df_from_json = pr.DataFrame.read_json(json_str)
    print(f"DataFrame loaded from JSON:\n{df_from_json}")
    
    # Delete file
    os.remove(csv_path)
    
    return df_from_json

def test_series_operations():
    """Test for Series operations"""
    print("\n=== Series Operations Test ===")
    
    # Create Series
    series = pr.Series("test_series", ["a", "b", "c", "d", "e"])
    print(f"Created Series: {series}")
    
    # Get values
    values = series.values
    print(f"Values: {values}")
    
    # Get and set name
    print(f"Series name: {series.name}")
    series.name = "renamed_series"
    print(f"Renamed Series name: {series.name}")
    
    # Convert to NumPy array
    # For numeric Series
    num_series = pr.Series("numbers", ["1", "2", "3", "4", "5"])
    np_array = num_series.to_numpy()
    print(f"NumPy array: {np_array}")
    print(f"Array type: {type(np_array)}")
    
    return series

def test_na_series():
    """Test for NASeries operations"""
    print("\n=== NASeries Operations Test ===")
    
    # Create Series with NA
    data = [None, "b", None, "d", "e"]
    na_series = pr.NASeries("na_test", data)
    print(f"Series with NA: {na_series}")
    
    # Detect NA values
    is_na = na_series.isna()
    print(f"NA mask: {is_na}")
    
    # Drop NA values
    dropped = na_series.dropna()
    print(f"After dropping NA: {dropped}")
    
    # Fill NA values
    filled = na_series.fillna("FILLED")
    print(f"After filling NA: {filled}")
    
    return na_series

def main():
    """Main function"""
    print("PandRS Python Bindings Test")
    
    df = test_dataframe_creation()
    df_json = test_io_operations(df)
    series = test_series_operations()
    na_series = test_na_series()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()