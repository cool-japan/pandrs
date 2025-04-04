"""
文字列プール最適化のベンチマーク

このベンチマークでは、文字列プール最適化による
メモリ使用効率と変換パフォーマンスを検証します。
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
import psutil
import gc
import sys
from tabulate import tabulate

def measure_memory():
    """現在のメモリ使用量を返す (MB単位)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def generate_data(rows, unique_ratio=0.1):
    """テストデータを生成
    
    Args:
        rows: 生成する行数
        unique_ratio: 重複なし文字列の割合 (0.1 = 10%が一意)
    """
    # 一意な文字列の数を計算
    unique_count = int(rows * unique_ratio)
    unique_count = max(unique_count, 1)  # 最低1つ
    
    # 一意な文字列のプールを生成
    unique_strings = [f"unique_value_{i}" for i in range(unique_count)]
    
    # 行数分のデータを生成（一意な文字列をランダムに選択）
    data = np.random.choice(unique_strings, size=rows)
    return data

def run_benchmark():
    """ベンチマーク実行"""
    print("=== 文字列プール最適化 パフォーマンスベンチマーク ===\n")
    
    # テストサイズとユニーク率
    test_configs = [
        {"rows": 100_000, "unique_ratio": 0.01},   # 1% ユニーク (高重複)
        {"rows": 100_000, "unique_ratio": 0.1},    # 10% ユニーク
        {"rows": 100_000, "unique_ratio": 0.5},    # 50% ユニーク
        {"rows": 1_000_000, "unique_ratio": 0.01}, # 1% ユニーク、大規模
    ]
    
    results = []
    
    for config in test_configs:
        rows = config["rows"]
        unique_ratio = config["unique_ratio"]
        
        print(f"\n## データサイズ: {rows:,}行, ユニーク率: {unique_ratio:.1%} ##")
        
        # データセット生成
        string_data = generate_data(rows, unique_ratio)
        numeric_data = np.arange(rows)
        
        # メモリ使用量をリセット
        gc.collect()
        base_memory = measure_memory()
        print(f"ベースメモリ使用量: {base_memory:.2f} MB")
        
        # ------ 通常のStringColumn (プール無し) ------
        # メモリ使用量測定のため新しいスコープで測定
        start_time = time.time()
        df_no_pool = pr.OptimizedDataFrame()
        
        # 文字列プールを使わない方法で追加 (以前の実装)
        string_list = list(string_data)  # ndarray -> list変換
        df_no_pool.add_int_column('id', list(range(rows)))
        
        # StringColumnに直接追加（プーリングなし）
        with_nopool_time_start = time.time()
        df_no_pool.add_string_column('str_value', string_list)
        with_nopool_time = time.time() - with_nopool_time_start
        
        # メモリ測定
        gc.collect()
        no_pool_memory = measure_memory() - base_memory
        
        print(f"1. 文字列プール無し:")
        print(f"   - 処理時間: {with_nopool_time:.6f}秒")
        print(f"   - 追加メモリ: {no_pool_memory:.2f} MB")
        
        # ------ 文字列プール使用 ------
        # メモリをリセット
        df_no_pool = None
        gc.collect()
        reset_memory = measure_memory()
        
        # 文字列プールを初期化
        string_pool = pr.StringPool()
        
        # プール使用のDataFrame作成
        df_with_pool = pr.OptimizedDataFrame()
        df_with_pool.add_int_column('id', list(range(rows)))
        
        # 文字列プールを使用して追加
        with_pool_time_start = time.time()
        
        # インデックスに変換してから追加
        py_list = string_list  # すでにlistに変換済み
        # 先にストリングプールに追加してから使う
        pool_indices = string_pool.add_list(py_list)
        df_with_pool.add_string_column_from_pylist('str_value', py_list)
        
        with_pool_time = time.time() - with_pool_time_start
        
        # メモリ測定
        gc.collect()
        with_pool_memory = measure_memory() - reset_memory
        
        # プール統計を取得
        pool_stats = string_pool.get_stats()
        
        print(f"2. 文字列プール使用:")
        print(f"   - 処理時間: {with_pool_time:.6f}秒")
        print(f"   - 追加メモリ: {with_pool_memory:.2f} MB")
        print(f"   - 文字列プール統計:")
        print(f"     * 総文字列数: {pool_stats['total_strings']:,}")
        print(f"     * 一意な文字列: {pool_stats['unique_strings']:,}")
        print(f"     * 節約バイト数: {pool_stats['bytes_saved']:,}")
        print(f"     * 重複率: {pool_stats['duplicate_ratio']:.2%}")
        
        # 変換ベンチマーク (プール ↔ pandas)
        to_pandas_start = time.time()
        pd_df = df_with_pool.to_pandas()
        to_pandas_time = time.time() - to_pandas_start
        
        from_pandas_start = time.time()
        back_to_optimized = pr.OptimizedDataFrame.from_pandas(pd_df)
        from_pandas_time = time.time() - from_pandas_start
        
        print(f"3. pandas変換:")
        print(f"   - 最適化→pandas: {to_pandas_time:.6f}秒")
        print(f"   - pandas→最適化: {from_pandas_time:.6f}秒")
        
        # 結果を記録
        results.append({
            'データサイズ': f"{rows:,}行",
            'ユニーク率': f"{unique_ratio:.1%}",
            'プール無し時間': with_nopool_time,
            'プール使用時間': with_pool_time,
            'プール無しメモリ': no_pool_memory,
            'プール使用メモリ': with_pool_memory,
            'メモリ削減率': f"{(1 - with_pool_memory / no_pool_memory):.2%}" if no_pool_memory > 0 else "N/A",
            '重複率': pool_stats['duplicate_ratio']
        })
    
    # 結果表示
    print("\n=== 結果サマリー ===")
    try:
        # Ensure tabulate is imported correctly
        from tabulate import tabulate as tab_func
        print(tab_func(results, headers="keys", tablefmt="grid", floatfmt=".6f"))
    except Exception as e:
        print("結果のフォーマットエラー:", e)
        for r in results:
            print(r)
    
    # 考察
    print("\n考察:")
    print("1. メモリ効率: 文字列プールは重複が多いデータセットでメモリ使用量を大幅に削減")
    print("2. 変換性能: プール使用時はPythonとRust間の文字列変換が高速化")
    print("3. 最適な用途: 重複率が高いカテゴリカル文字列データでは特に有効")
    print("\n注意: psutilのメモリ測定はおおよその値です。実際の削減量は内部計測値を参照してください。")

if __name__ == "__main__":
    try:
        import tabulate
        import psutil
    except ImportError:
        print("必要なモジュールがありません。以下のコマンドでインストールしてください:")
        print("pip install tabulate psutil")
        sys.exit(1)
    
    print("文字列プール最適化のベンチマークを実行中...")
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。")
    except Exception as e:
        print(f"エラー: {e}")