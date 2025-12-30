#!/usr/bin/env python
"""
GPU Benchmarking Example for PandRS Python Bindings

This example demonstrates how to use the GPU benchmarking utilities from Python.
Run this example with:
    python gpu_benchmark.py
"""

import pandas as pd
import numpy as np
import time
import pandrs as pr

def main():
    print("PandRS GPU Benchmarking Example")
    print("===============================")

    # Initialize GPU
    try:
        status = pr.gpu.init_gpu()
        print(f"GPU available: {status.available}")
        
        if status.available:
            print(f"Device: {status.device_name}")
            print(f"CUDA version: {status.cuda_version}")
            print(f"Total memory: {status.total_memory / (1024**2):.1f} MB")
            print(f"Free memory: {status.free_memory / (1024**2):.1f} MB")
    except Exception as e:
        print(f"Error initializing GPU: {e}")
        print("Running with CPU only.")

    # Matrix multiplication benchmark
    print("\nMatrix Multiplication Benchmark")
    print("-------------------------------")
    
    sizes = [(500, 500, 500), (1000, 1000, 1000), (2000, 2000, 2000)]
    
    for m, n, k in sizes:
        # Create random matrices
        a = np.random.rand(m, k)
        b = np.random.rand(k, n)
        
        # CPU timing
        cpu_start = time.time()
        c_cpu = np.dot(a, b)
        cpu_time = time.time() - cpu_start
        print(f"CPU time ({m}x{k} * {k}x{n}): {cpu_time*1000:.2f} ms")
        
        # GPU timing (if available)
        try:
            gpu_a = pr.gpu.GpuMatrix(a)
            gpu_b = pr.gpu.GpuMatrix(b)
            
            gpu_start = time.time()
            c_gpu = gpu_a.dot(gpu_b)
            gpu_time = time.time() - gpu_start
            
            print(f"GPU time ({m}x{k} * {k}x{n}): {gpu_time*1000:.2f} ms")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Verify results
            c_gpu_np = c_gpu.to_numpy()
            max_diff = np.max(np.abs(c_cpu - c_gpu_np))
            print(f"Max difference: {max_diff:.6e}")
        except Exception as e:
            print(f"GPU operation failed: {e}")
    
    # DataFrame operations benchmark
    print("\nDataFrame Operations Benchmark")
    print("-----------------------------")
    
    # Create a test DataFrame
    n_rows = 10000
    n_cols = 10
    
    data = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        data[col_name] = np.random.rand(n_rows)
    
    # Create pandas DataFrame
    pd_df = pd.DataFrame(data)
    
    # Create PandRS DataFrame
    pr_df = pr.DataFrame.from_pandas(pd_df)
    
    # Create PandRS OptimizedDataFrame
    opt_df = pr.OptimizedDataFrame.from_pandas(pd_df)
    
    # Benchmark correlation matrix
    print("\nCorrelation Matrix Benchmark")
    
    # pandas timing
    pd_start = time.time()
    pd_corr = pd_df.corr()
    pd_time = time.time() - pd_start
    print(f"pandas time: {pd_time*1000:.2f} ms")
    
    # PandRS timing
    pr_start = time.time()
    pr_corr = pr_df.corr()
    pr_time = time.time() - pr_start
    print(f"PandRS time: {pr_time*1000:.2f} ms")
    
    # PandRS GPU timing
    try:
        cols = list(pd_df.columns)
        gpu_start = time.time()
        gpu_corr = opt_df.gpu_corr(cols)
        gpu_time = time.time() - gpu_start
        print(f"PandRS GPU time: {gpu_time*1000:.2f} ms")
        print(f"Speedup vs pandas: {pd_time/gpu_time:.2f}x")
        print(f"Speedup vs PandRS CPU: {pr_time/gpu_time:.2f}x")
    except Exception as e:
        print(f"GPU correlation failed: {e}")
    
    # Benchmark linear regression
    print("\nLinear Regression Benchmark")
    
    # Create regression data
    X = np.random.rand(n_rows, n_cols-1)
    y = np.random.rand(n_rows)
    
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_cols-1)])
    X_df["y"] = y
    
    # pandas timing
    pd_start = time.time()
    pd_model = pd.DataFrame(X).corrwith(pd.Series(y))
    pd_time = time.time() - pd_start
    print(f"pandas time: {pd_time*1000:.2f} ms")
    
    # PandRS timing
    pr_df = pr.OptimizedDataFrame.from_pandas(X_df)
    
    pr_start = time.time()
    x_cols = [f"x{i}" for i in range(n_cols-1)]
    pr_model = pr_df.linear_regression("y", x_cols)
    pr_time = time.time() - pr_start
    print(f"PandRS time: {pr_time*1000:.2f} ms")
    
    # PandRS GPU timing
    try:
        gpu_start = time.time()
        gpu_model = pr_df.gpu_linear_regression("y", x_cols)
        gpu_time = time.time() - gpu_start
        print(f"PandRS GPU time: {gpu_time*1000:.2f} ms")
        print(f"Speedup vs PandRS CPU: {pr_time/gpu_time:.2f}x")
    except Exception as e:
        print(f"GPU linear regression failed: {e}")
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()