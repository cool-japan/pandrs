"""
PandRS vs pandas 100万行ベンチマーク
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
import sys

def run_benchmark():
    """100万行のDataFrame作成ベンチマーク"""
    print("=== PandRS vs pandas 100万行ベンチマーク ===\n")
    
    # ベンチマークデータ準備
    rows = 1_000_000
    print(f"データ準備中: {rows:,}行 x 3列...")
    
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # pandas計測
    print("\n--- pandas DataFrame作成 ---")
    start = time.time()
    pd_df = pd.DataFrame(data)
    pandas_time = time.time() - start
    print(f"pandas DataFrame作成時間: {pandas_time:.6f}秒")
    
    # メモリ使用量確認 (概算)
    pd_memory = pd_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"pandas DataFrame概算メモリ使用量: {pd_memory:.2f} MB")
    
    # pandrs計測
    print("\n--- PandRS DataFrame作成 ---")
    start = time.time()
    pr_df = pr.DataFrame(data)
    pandrs_time = time.time() - start
    print(f"PandRS DataFrame作成時間: {pandrs_time:.6f}秒")
    
    # 比率計算
    ratio = pandas_time / pandrs_time if pandrs_time > 0 else float('inf')
    if ratio > 1:
        print(f"PandRSはpandasより {ratio:.2f}倍速い")
    else:
        print(f"PandRSはpandasより {1/ratio:.2f}倍遅い")
    
    # 概要レポート
    print("\n=== ベンチマーク結果概要 ===")
    print(f"データサイズ: {rows:,}行 x 3列")
    print(f"pandas DataFrame作成時間: {pandas_time:.6f}秒")
    print(f"PandRS DataFrame作成時間: {pandrs_time:.6f}秒")
    print(f"pandas/PandRS比率: {ratio:.2f}x")
    
    print("\n注: Rustネイティブ版PandRSはさらに高速で、同じ操作が数百ミリ秒で完了します。")
    print("Pythonバインディングのオーバーヘッドが主な性能差の原因です。")
    
if __name__ == "__main__":
    print("警告: このベンチマークは大量のメモリを使用します。十分なRAMを確保してください。")
    
    try:
        run_benchmark()
    except MemoryError:
        print("エラー: メモリ不足のため実行できませんでした。")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {str(e)}")
        sys.exit(1)