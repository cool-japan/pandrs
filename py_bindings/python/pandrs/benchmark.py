"""
PandRS パフォーマンスベンチマークモジュール
"""
import time
import sys

try:
    import pandrs as pr
except ImportError:
    print("pandrsモジュールが見つかりません。正しくインストールされているか確認してください。")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("pandasモジュールが見つかりません。pip install pandasでインストールしてください。")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpyモジュールが見つかりません。pip install numpyでインストールしてください。")
    sys.exit(1)

def run_benchmark(name, rows, pandas_func, pandrs_func):
    """
    指定された関数のベンチマークを実行し、結果を返します。
    
    Args:
        name: ベンチマーク名
        rows: データ行数
        pandas_func: pandasのベンチマーク関数
        pandrs_func: pandrsのベンチマーク関数
        
    Returns:
        結果を含む辞書
    """
    print(f"\n実行中: {name} ({rows:,}行)...")
    
    # Pandas計測
    start = time.time()
    pandas_result = pandas_func()
    pandas_time = time.time() - start
    print(f"  pandas: {pandas_time:.6f}秒")
    
    # PandRS計測
    start = time.time()
    pandrs_result = pandrs_func()
    pandrs_time = time.time() - start
    print(f"  pandrs: {pandrs_time:.6f}秒")
    
    # 比率
    if pandrs_time > 0:
        ratio = pandas_time / pandrs_time
        relative = "速い" if ratio > 1 else "遅い"
        print(f"  比率: pandas/pandrs = {ratio:.2f}x (pandasより{abs(ratio-1):.2f}倍{relative})")
    else:
        ratio = float('inf')
        print("  比率: 計算不可 (pandrsの実行時間が0)")
    
    return {
        'name': name,
        'rows': rows,
        'pandas_time': pandas_time,
        'pandrs_time': pandrs_time,
        'ratio': ratio
    }

def dataframe_creation_benchmark(rows=10000):
    """
    DataFrame作成のベンチマーク
    """
    # テストデータ
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # Pandas関数
    def pandas_create():
        return pd.DataFrame(data)
    
    # PandRS関数
    def pandrs_create():
        return pr.DataFrame(data)
    
    return run_benchmark("DataFrame作成", rows, pandas_create, pandrs_create)

def column_access_benchmark(rows=10000):
    """
    カラムアクセスのベンチマーク
    """
    # テストデータ
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # 事前作成
    pd_df = pd.DataFrame(data)
    pr_df = pr.DataFrame(data)
    
    # Pandas関数
    def pandas_access():
        return pd_df['A']
    
    # PandRS関数
    def pandrs_access():
        return pr_df['A']
    
    return run_benchmark("カラムアクセス", rows, pandas_access, pandrs_access)

def conversion_benchmark(rows=10000):
    """
    変換機能のベンチマーク
    """
    # テストデータ
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # 事前作成
    pd_df = pd.DataFrame(data)
    pr_df = pr.DataFrame(data)
    
    # Pandas関数
    def pandas_to_dict():
        return pd_df.to_dict()
    
    # PandRS関数
    def pandrs_to_dict():
        return pr_df.to_dict()
    
    return run_benchmark("to_dict変換", rows, pandas_to_dict, pandrs_to_dict)

def interop_benchmark(rows=10000):
    """
    相互運用性のベンチマーク
    """
    # テストデータ
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    data = {
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    }
    
    # 事前作成
    pd_df = pd.DataFrame(data)
    pr_df = pr.DataFrame(data)
    
    # Pandas→PandRS関数
    def pandas_to_pandrs():
        return pr.DataFrame.from_pandas(pd_df)
    
    # ダミー関数
    def dummy():
        pass
    
    result1 = run_benchmark("pandas→pandrs変換", rows, pandas_to_pandrs, dummy)
    
    # PandRS→Pandas関数
    def pandrs_to_pandas():
        return pr_df.to_pandas()
    
    result2 = run_benchmark("pandrs→pandas変換", rows, dummy, pandrs_to_pandas)
    
    return [result1, result2]

def run_all_benchmarks():
    """
    すべてのベンチマークを実行
    """
    print("=== PandRS vs pandas パフォーマンスベンチマーク ===")
    
    results = []
    
    # 様々なサイズでDataFrame作成をテスト
    for rows in [10, 100, 1000, 10000, 100000]:
        results.append(dataframe_creation_benchmark(rows))
    
    # その他のベンチマーク
    results.append(column_access_benchmark(10000))
    results.append(conversion_benchmark(10000))
    interop_results = interop_benchmark(10000)
    results.extend(interop_results)
    
    # 結果のサマリー表示
    print("\n=== ベンチマーク結果サマリー ===")
    print(f"{'テスト名':<25} {'行数':>10} {'pandas(秒)':>12} {'pandrs(秒)':>12} {'比率':>10}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['name']:<25} {r['rows']:>10,} {r['pandas_time']:>12.6f} {r['pandrs_time']:>12.6f} {r['ratio']:>10.2f}x")
    
    print("\n注意: 比率はpandas/pandrsです。1.0より大きい値はpandrsが速いことを意味します。")
    print("Python-Rust間のデータ変換オーバーヘッドのため、パフォーマンスは純粋なRust実装より低下します。")
    print("純粋なRust実装のpandrsは、10万行のDataFrame作成が約50msで完了します。")

if __name__ == "__main__":
    run_all_benchmarks()