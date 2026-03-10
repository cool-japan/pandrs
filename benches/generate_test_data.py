#!/usr/bin/env python3
"""
Generate consistent test datasets for pandas/polars/PandRS benchmarks.
This ensures all three libraries are tested with identical data.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def generate_test_data(size: int, output_dir: str = "/tmp/benchmark_data"):
    """Generate test datasets of various sizes and formats."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    data = {
        'id': np.arange(size),
        'category': np.random.choice(categories, size=size),
        'category2': np.random.choice(categories[:4], size=size),
        'value': np.random.normal(100, 15, size),
        'value2': np.random.exponential(50, size),
        'integer': np.random.randint(0, 1000, size),
        'string_data': [f"string_{i % 1000}" for i in range(size)],
        'timestamp': pd.date_range('2023-01-01', periods=size, freq='1min')
    }

    df = pd.DataFrame(data)

    # Save in multiple formats
    csv_path = f"{output_dir}/data_{size}.csv"
    json_path = f"{output_dir}/data_{size}.json"
    parquet_path = f"{output_dir}/data_{size}.parquet"

    print(f"Generating test data for size {size:,}...")

    # CSV
    df.to_csv(csv_path, index=False)
    print(f"  - CSV: {csv_path} ({os.path.getsize(csv_path) / 1024 / 1024:.2f} MB)")

    # JSON
    df.to_json(json_path, orient='records', lines=True)
    print(f"  - JSON: {json_path} ({os.path.getsize(json_path) / 1024 / 1024:.2f} MB)")

    # Parquet
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"  - Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB)")

    return df

def generate_join_datasets(size_left: int, size_right: int, output_dir: str = "/tmp/benchmark_data"):
    """Generate datasets for join operations."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Left dataset
    left_data = {
        'key': np.arange(size_left),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=size_left),
        'value_left': np.random.normal(100, 15, size=size_left)
    }
    df_left = pd.DataFrame(left_data)

    # Right dataset (with some overlap in keys)
    right_data = {
        'key': np.random.randint(0, size_left, size=size_right),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=size_right),
        'value_right': np.random.exponential(50, size=size_right)
    }
    df_right = pd.DataFrame(right_data)

    left_path = f"{output_dir}/join_left_{size_left}.csv"
    right_path = f"{output_dir}/join_right_{size_right}.csv"

    df_left.to_csv(left_path, index=False)
    df_right.to_csv(right_path, index=False)

    print(f"Generated join datasets: {size_left:,} x {size_right:,}")
    print(f"  - Left: {left_path}")
    print(f"  - Right: {right_path}")

    return df_left, df_right

if __name__ == "__main__":
    # Generate datasets for different benchmark sizes
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    print("=" * 60)
    print("Generating Test Datasets for Benchmarks")
    print("=" * 60)

    for size in sizes:
        generate_test_data(size)
        print()

    # Generate join datasets
    print("=" * 60)
    print("Generating Join Datasets")
    print("=" * 60)

    generate_join_datasets(10_000, 10_000)
    generate_join_datasets(100_000, 100_000)
    generate_join_datasets(1_000_000, 1_000_000)

    print("\nTest data generation complete!")
