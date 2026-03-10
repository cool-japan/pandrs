#!/usr/bin/env python3
"""
Comprehensive polars benchmarks for comparison with PandRS.
"""

import polars as pl
import numpy as np
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class PolarsBenchmark:
    def __init__(self, data_dir: str = "/tmp/benchmark_data"):
        self.data_dir = Path(data_dir)
        self.results = {}

    def time_it(self, func, name: str, iterations: int = 3):
        """Run a function multiple times and record timing."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        median_time = np.median(times)
        self.results[name] = {
            'median_ms': median_time,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times)
        }
        return median_time, result

    # ===================
    # I/O BENCHMARKS
    # ===================

    def benchmark_csv_read(self, size: int):
        """Benchmark CSV reading."""
        csv_path = str(self.data_dir / f"data_{size}.csv")

        def read_csv():
            return pl.read_csv(csv_path)

        median_time, df = self.time_it(read_csv, f"csv_read_{size}")
        print(f"  CSV Read ({size:,} rows): {median_time:.2f} ms")
        return df

    def benchmark_csv_write(self, df: pl.DataFrame, size: int):
        """Benchmark CSV writing."""
        output_path = str(self.data_dir / f"polars_output_{size}.csv")

        def write_csv():
            df.write_csv(output_path)

        median_time, _ = self.time_it(write_csv, f"csv_write_{size}")
        print(f"  CSV Write ({size:,} rows): {median_time:.2f} ms")

    def benchmark_json_read(self, size: int):
        """Benchmark JSON reading."""
        json_path = str(self.data_dir / f"data_{size}.json")

        def read_json():
            return pl.read_ndjson(json_path)

        median_time, df = self.time_it(read_json, f"json_read_{size}")
        print(f"  JSON Read ({size:,} rows): {median_time:.2f} ms")
        return df

    def benchmark_parquet_read(self, size: int):
        """Benchmark Parquet reading."""
        parquet_path = str(self.data_dir / f"data_{size}.parquet")

        def read_parquet():
            return pl.read_parquet(parquet_path)

        median_time, df = self.time_it(read_parquet, f"parquet_read_{size}")
        print(f"  Parquet Read ({size:,} rows): {median_time:.2f} ms")
        return df

    def benchmark_parquet_write(self, df: pl.DataFrame, size: int):
        """Benchmark Parquet writing."""
        output_path = str(self.data_dir / f"polars_output_{size}.parquet")

        def write_parquet():
            df.write_parquet(output_path, compression='snappy')

        median_time, _ = self.time_it(write_parquet, f"parquet_write_{size}")
        print(f"  Parquet Write ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # DATAFRAME OPERATIONS
    # ===================

    def benchmark_creation_from_dict(self, size: int):
        """Benchmark DataFrame creation from dictionary."""
        categories = (['A', 'B', 'C', 'D'] * (size // 4 + 1))[:size]
        data = {
            'id': list(range(size)),
            'category': categories,
            'value': np.random.normal(100, 15, size).tolist()
        }

        def create_df():
            return pl.DataFrame(data)

        median_time, df = self.time_it(create_df, f"creation_from_dict_{size}")
        print(f"  Creation from Dict ({size:,} rows): {median_time:.2f} ms")
        return df

    def benchmark_column_selection(self, df: pl.DataFrame, size: int):
        """Benchmark column selection."""
        def select_columns():
            return df.select(['id', 'value'])

        median_time, _ = self.time_it(select_columns, f"column_selection_{size}")
        print(f"  Column Selection ({size:,} rows): {median_time:.2f} ms")

    def benchmark_row_filtering(self, df: pl.DataFrame, size: int):
        """Benchmark row filtering."""
        mean_value = df['value'].mean()

        def filter_rows():
            return df.filter(pl.col('value') > mean_value)

        median_time, filtered = self.time_it(filter_rows, f"row_filtering_{size}")
        print(f"  Row Filtering ({size:,} rows): {median_time:.2f} ms")

    def benchmark_sorting(self, df: pl.DataFrame, size: int):
        """Benchmark single column sorting."""
        def sort_data():
            return df.sort('value')

        median_time, _ = self.time_it(sort_data, f"sorting_single_{size}")
        print(f"  Sorting Single Column ({size:,} rows): {median_time:.2f} ms")

    def benchmark_multi_sort(self, df: pl.DataFrame, size: int):
        """Benchmark multi-column sorting."""
        def multi_sort():
            return df.sort(['category', 'value'])

        median_time, _ = self.time_it(multi_sort, f"sorting_multi_{size}")
        print(f"  Sorting Multi Column ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # AGGREGATIONS
    # ===================

    def benchmark_aggregations(self, df: pl.DataFrame, size: int):
        """Benchmark basic aggregations."""
        def agg_sum():
            return df['value'].sum()

        def agg_mean():
            return df['value'].mean()

        def agg_count():
            return df['value'].count()

        def agg_std():
            return df['value'].std()

        median_time, _ = self.time_it(agg_sum, f"agg_sum_{size}")
        print(f"  Aggregation Sum ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(agg_mean, f"agg_mean_{size}")
        print(f"  Aggregation Mean ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(agg_count, f"agg_count_{size}")
        print(f"  Aggregation Count ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(agg_std, f"agg_std_{size}")
        print(f"  Aggregation Std ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # GROUPBY OPERATIONS
    # ===================

    def benchmark_groupby_single(self, df: pl.DataFrame, size: int):
        """Benchmark single-key groupby with aggregation."""
        def groupby_agg():
            return df.group_by('category').agg(pl.col('value').sum())

        median_time, _ = self.time_it(groupby_agg, f"groupby_single_{size}")
        print(f"  GroupBy Single Key ({size:,} rows): {median_time:.2f} ms")

    def benchmark_groupby_multi(self, df: pl.DataFrame, size: int):
        """Benchmark multi-key groupby."""
        def groupby_multi():
            return df.group_by(['category', 'category2']).agg(pl.col('value').sum())

        median_time, _ = self.time_it(groupby_multi, f"groupby_multi_{size}")
        print(f"  GroupBy Multi Key ({size:,} rows): {median_time:.2f} ms")

    def benchmark_groupby_multiple_aggs(self, df: pl.DataFrame, size: int):
        """Benchmark groupby with multiple aggregations."""
        def groupby_aggs():
            return df.group_by('category').agg([
                pl.col('value').sum().alias('sum'),
                pl.col('value').mean().alias('mean'),
                pl.col('value').count().alias('count'),
                pl.col('value').std().alias('std')
            ])

        median_time, _ = self.time_it(groupby_aggs, f"groupby_multiple_aggs_{size}")
        print(f"  GroupBy Multiple Aggs ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # JOIN OPERATIONS
    # ===================

    def benchmark_join_inner(self, size: int):
        """Benchmark inner join."""
        left_path = str(self.data_dir / f"join_left_{size}.csv")
        right_path = str(self.data_dir / f"join_right_{size}.csv")

        left_df = pl.read_csv(left_path)
        right_df = pl.read_csv(right_path)

        def inner_join():
            return left_df.join(right_df, on='key', how='inner')

        median_time, result = self.time_it(inner_join, f"join_inner_{size}")
        print(f"  Inner Join ({size:,} x {size:,}): {median_time:.2f} ms")

    def benchmark_join_left(self, size: int):
        """Benchmark left join."""
        left_path = str(self.data_dir / f"join_left_{size}.csv")
        right_path = str(self.data_dir / f"join_right_{size}.csv")

        left_df = pl.read_csv(left_path)
        right_df = pl.read_csv(right_path)

        def left_join():
            return left_df.join(right_df, on='key', how='left')

        median_time, result = self.time_it(left_join, f"join_left_{size}")
        print(f"  Left Join ({size:,} x {size:,}): {median_time:.2f} ms")

    def benchmark_join_outer(self, size: int):
        """Benchmark outer join."""
        left_path = str(self.data_dir / f"join_left_{size}.csv")
        right_path = str(self.data_dir / f"join_right_{size}.csv")

        left_df = pl.read_csv(left_path)
        right_df = pl.read_csv(right_path)

        def outer_join():
            return left_df.join(right_df, on='key', how='full')

        median_time, result = self.time_it(outer_join, f"join_outer_{size}")
        print(f"  Outer Join ({size:,} x {size:,}): {median_time:.2f} ms")

    # ===================
    # STRING OPERATIONS
    # ===================

    def benchmark_string_ops(self, df: pl.DataFrame, size: int):
        """Benchmark string operations."""
        def str_upper():
            return df.select(pl.col('string_data').str.to_uppercase())

        def str_lower():
            return df.select(pl.col('string_data').str.to_lowercase())

        def str_contains():
            return df.select(pl.col('string_data').str.contains('string_5'))

        median_time, _ = self.time_it(str_upper, f"string_upper_{size}")
        print(f"  String Upper ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(str_lower, f"string_lower_{size}")
        print(f"  String Lower ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(str_contains, f"string_contains_{size}")
        print(f"  String Contains ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # WINDOW FUNCTIONS
    # ===================

    def benchmark_window_ops(self, df: pl.DataFrame, size: int):
        """Benchmark window operations."""
        def rolling_mean():
            return df.select(pl.col('value').rolling_mean(window_size=10))

        def rolling_sum():
            return df.select(pl.col('value').rolling_sum(window_size=10))

        # Polars doesn't have expanding() directly, use cumulative
        def expanding_mean():
            return df.select(pl.col('value').cum_sum() / (pl.arange(0, pl.len()) + 1))

        median_time, _ = self.time_it(rolling_mean, f"rolling_mean_{size}")
        print(f"  Rolling Mean ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(rolling_sum, f"rolling_sum_{size}")
        print(f"  Rolling Sum ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(expanding_mean, f"expanding_mean_{size}")
        print(f"  Expanding Mean ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # STATISTICAL OPERATIONS
    # ===================

    def benchmark_statistics(self, df: pl.DataFrame, size: int):
        """Benchmark statistical operations."""
        def describe():
            return df.select([
                pl.col('value').mean().alias('mean'),
                pl.col('value').std().alias('std'),
                pl.col('value').min().alias('min'),
                pl.col('value').max().alias('max')
            ])

        def correlation():
            # Polars correlation is done differently
            return df.select([
                pl.corr('value', 'value2').alias('corr_value_value2'),
                pl.corr('value', 'integer').alias('corr_value_integer')
            ])

        def value_counts():
            return df['category'].value_counts()

        median_time, _ = self.time_it(describe, f"describe_{size}")
        print(f"  Describe ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(correlation, f"correlation_{size}")
        print(f"  Correlation ({size:,} rows): {median_time:.2f} ms")

        median_time, _ = self.time_it(value_counts, f"value_counts_{size}")
        print(f"  Value Counts ({size:,} rows): {median_time:.2f} ms")

    # ===================
    # RUN ALL BENCHMARKS
    # ===================

    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("=" * 80)
        print("POLARS COMPREHENSIVE BENCHMARKS")
        print("=" * 80)

        # Test sizes
        io_sizes = [10_000, 100_000]
        operation_sizes = [10_000, 100_000, 1_000_000]
        join_sizes = [10_000, 100_000]

        # I/O Benchmarks
        print("\n--- I/O Operations ---")
        for size in io_sizes:
            print(f"\nSize: {size:,}")
            df = self.benchmark_csv_read(size)
            self.benchmark_csv_write(df, size)
            df = self.benchmark_json_read(size)
            df = self.benchmark_parquet_read(size)
            self.benchmark_parquet_write(df, size)

        # DataFrame Operations
        print("\n--- DataFrame Operations ---")
        for size in operation_sizes:
            print(f"\nSize: {size:,}")
            df = self.benchmark_creation_from_dict(size)

            # Need to load actual data for operations
            if size <= 100_000:
                df_full = pl.read_csv(str(self.data_dir / f"data_{size}.csv"))
                self.benchmark_column_selection(df_full, size)
                self.benchmark_row_filtering(df_full, size)
                self.benchmark_sorting(df_full, size)
                self.benchmark_multi_sort(df_full, size)

        # Aggregations
        print("\n--- Aggregation Operations ---")
        for size in operation_sizes:
            print(f"\nSize: {size:,}")
            df = pl.read_csv(str(self.data_dir / f"data_{size}.csv"))
            self.benchmark_aggregations(df, size)

        # GroupBy
        print("\n--- GroupBy Operations ---")
        for size in operation_sizes:
            print(f"\nSize: {size:,}")
            df = pl.read_csv(str(self.data_dir / f"data_{size}.csv"))
            self.benchmark_groupby_single(df, size)
            self.benchmark_groupby_multi(df, size)
            self.benchmark_groupby_multiple_aggs(df, size)

        # Joins
        print("\n--- Join Operations ---")
        for size in join_sizes:
            print(f"\nSize: {size:,} x {size:,}")
            self.benchmark_join_inner(size)
            self.benchmark_join_left(size)
            self.benchmark_join_outer(size)

        # String Operations
        print("\n--- String Operations ---")
        for size in [10_000, 100_000]:
            print(f"\nSize: {size:,}")
            df = pl.read_csv(str(self.data_dir / f"data_{size}.csv"))
            self.benchmark_string_ops(df, size)

        # Window Functions
        print("\n--- Window Functions ---")
        for size in [10_000, 100_000]:
            print(f"\nSize: {size:,}")
            df = pl.read_csv(str(self.data_dir / f"data_{size}.csv"))
            self.benchmark_window_ops(df, size)

        # Statistics
        print("\n--- Statistical Operations ---")
        for size in operation_sizes:
            print(f"\nSize: {size:,}")
            df = pl.read_csv(str(self.data_dir / f"data_{size}.csv"))
            self.benchmark_statistics(df, size)

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

        # Save results
        output_path = "/tmp/polars_benchmark_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return self.results


if __name__ == "__main__":
    benchmark = PolarsBenchmark()
    results = benchmark.run_all_benchmarks()

    # Print summary
    print("\n--- PERFORMANCE SUMMARY ---")
    for op_name, metrics in sorted(results.items()):
        print(f"{op_name}: {metrics['median_ms']:.2f} ms")
