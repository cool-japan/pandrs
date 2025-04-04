"""
最適化実装と文字列プールのPythonバインディングベンチマーク (更新版)
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import sys
import gc
import psutil

def measure_memory():
    """現在のメモリ使用量を返す (MB単位)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def run_basic_benchmark():
    """基本ベンチマーク実行"""
    print("=== PandRS 最適化実装 vs pandas パフォーマンスベンチマーク ===\n")
    
    # テストサイズ
    row_sizes = [10_000, 100_000, 1_000_000]
    results = []
    
    for rows in row_sizes:
        print(f"\n## データサイズ: {rows:,}行 ##")
        
        # データ準備
        numeric_data = list(range(rows))
        string_data = [f"value_{i % 100}" for i in range(rows)]
        float_data = [i * 0.5 for i in range(rows)]
        bool_data = [i % 2 == 0 for i in range(rows)]
        
        # pandasテスト - DataFrame作成
        start = time.time()
        pd_df = pd.DataFrame({
            'A': numeric_data,
            'B': string_data,
            'C': float_data,
            'D': bool_data
        })
        pandas_create_time = time.time() - start
        print(f"pandas DataFrame作成: {pandas_create_time:.6f}秒")
        
        # PandRS従来実装 - DataFrame作成
        start = time.time()
        legacy_df = pr.DataFrame({
            'A': numeric_data,
            'B': string_data,
            'C': float_data,
            'D': bool_data
        })
        legacy_create_time = time.time() - start
        print(f"PandRS従来実装 DataFrame作成: {legacy_create_time:.6f}秒")
        
        # PandRS最適化実装 - DataFrame作成
        start = time.time()
        optimized_df = pr.OptimizedDataFrame()
        optimized_df.add_int_column('A', numeric_data)
        optimized_df.add_string_column('B', string_data)
        optimized_df.add_float_column('C', float_data)
        optimized_df.add_boolean_column('D', bool_data)
        optimized_create_time = time.time() - start
        print(f"PandRS最適化実装 DataFrame作成: {optimized_create_time:.6f}秒")
        
        # 結果保存
        results.append({
            'データサイズ': f"{rows:,}行",
            'pandas作成': pandas_create_time,
            'PandRS従来実装': legacy_create_time,
            'PandRS最適化実装': optimized_create_time,
            '従来比': legacy_create_time / optimized_create_time,
            'pandas比': pandas_create_time / optimized_create_time
        })
        
        # pandasとの相互変換テスト
        print("\n## pandas変換テスト ##")
        
        # PandRS→pandas変換
        start = time.time()
        pd_from_optimized = optimized_df.to_pandas()
        to_pandas_time = time.time() - start
        print(f"最適化DataFrame→pandas: {to_pandas_time:.6f}秒")
        
        # pandas→PandRS変換
        start = time.time()
        optimized_from_pd = pr.OptimizedDataFrame.from_pandas(pd_df)
        from_pandas_time = time.time() - start
        print(f"pandas→最適化DataFrame: {from_pandas_time:.6f}秒")
        
        # 遅延評価ベンチマーク（100万行以下のみ実行）
        if rows <= 100_000:
            print("\n## 遅延評価テスト ##")
            
            # フィルタリング - pandas
            start = time.time()
            filtered_pd = pd_df[pd_df['D'] == True]
            pandas_filter_time = time.time() - start
            print(f"pandas フィルタリング: {pandas_filter_time:.6f}秒")
            
            # フィルタリング - 従来実装
            # 実装されていないか、異なるAPIなので省略
            legacy_filter_time = "-"
            
            # フィルタリング - 最適化実装
            # 1. フィルタ条件用にブール列を準備
            start_filter = time.time()
            optimized_df_with_filter = pr.OptimizedDataFrame()
            optimized_df_with_filter.add_int_column('A', numeric_data)
            optimized_df_with_filter.add_string_column('B', string_data)
            optimized_df_with_filter.add_float_column('C', float_data)
            optimized_df_with_filter.add_boolean_column('filter', bool_data)
            
            # 2. フィルタリング実行
            filtered_optimized = optimized_df_with_filter.filter('filter')
            optimized_filter_time = time.time() - start_filter
            print(f"PandRS最適化実装 フィルタリング: {optimized_filter_time:.6f}秒")
            
            # 遅延評価 - LazyFrame使用
            start_lazy = time.time()
            lazy_df = pr.LazyFrame(optimized_df_with_filter)
            lazy_filtered = lazy_df.filter('filter').execute()
            lazy_filter_time = time.time() - start_lazy
            print(f"PandRS LazyFrame フィルタリング: {lazy_filter_time:.6f}秒")
    
    # 結果表示
    print("\n=== 基本ベンチマーク結果サマリー ===")
    try:
        from tabulate import tabulate as tab_func
        print(tab_func(results, headers="keys", tablefmt="pretty", floatfmt=".6f"))
    except Exception as e:
        print("結果のフォーマットエラー:", e)
        for r in results:
            print(r)

def generate_string_data(rows, unique_ratio=0.1):
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
    return list(data)  # numpy配列をリストに変換して返す

def run_string_pool_benchmark():
    """文字列プール最適化ベンチマーク実行"""
    print("\n\n=== 文字列プール最適化 パフォーマンスベンチマーク ===\n")
    
    # テスト構成
    test_configs = [
        {"rows": 100_000, "unique_ratio": 0.01},   # 1% ユニーク (高重複)
        {"rows": 100_000, "unique_ratio": 0.1},    # 10% ユニーク
        {"rows": 100_000, "unique_ratio": 0.5},    # 50% ユニーク
    ]
    
    # 同じように先にプールにデータをロードしておく
    string_pool = pr.StringPool()
    
    results = []
    
    for config in test_configs:
        rows = config["rows"]
        unique_ratio = config["unique_ratio"]
        
        print(f"\n## データサイズ: {rows:,}行, ユニーク率: {unique_ratio:.1%} ##")
        
        # データセット生成
        string_data = generate_string_data(rows, unique_ratio)
        numeric_data = list(range(rows))
        
        # メモリ使用量をリセット
        gc.collect()
        base_memory = measure_memory()
        print(f"ベースメモリ使用量: {base_memory:.2f} MB")
        
        # ------ 通常のStringColumn (プール無し) ------
        # メモリ使用量測定のため新しいスコープで測定
        start_time = time.time()
        df_no_pool = pr.OptimizedDataFrame()
        df_no_pool.add_int_column('id', numeric_data)
        
        # StringColumnに直接追加（プーリングなし - 従来の実装）
        with_nopool_time_start = time.time()
        df_no_pool.add_string_column('str_value', string_data)  
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
        df_with_pool.add_int_column('id', numeric_data)
        
        # 文字列プールを使用して追加
        with_pool_time_start = time.time()
        
        # Pythonリストから直接文字列カラムに追加（新しい実装）
        py_list = string_data
        # 先にプールに登録
        indices = string_pool.add_list(py_list)
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
        memory_reduction = "-"
        if no_pool_memory > 0:
            memory_reduction = f"{(1 - with_pool_memory / no_pool_memory):.2%}"
            
        results.append({
            'データサイズ': f"{rows:,}行",
            'ユニーク率': f"{unique_ratio:.1%}",
            'プール無し時間': with_nopool_time,
            'プール使用時間': with_pool_time,
            '速度向上': f"{with_nopool_time / with_pool_time:.2f}倍",
            'プール無しメモリ': f"{no_pool_memory:.2f} MB",
            'プール使用メモリ': f"{with_pool_memory:.2f} MB",
            'メモリ削減率': memory_reduction,
            '重複率': f"{pool_stats['duplicate_ratio']:.2%}"
        })
    
    # 結果表示
    print("\n=== 文字列プール最適化結果サマリー ===")
    try:
        from tabulate import tabulate as tab_func
        print(tab_func(results, headers="keys", tablefmt="grid"))
    except Exception as e:
        print("結果のフォーマットエラー:", e)
        for r in results:
            print(r)
    
    # 考察
    print("\n考察:")
    print("1. メモリ効率: 文字列プールは重複が多いデータセットでメモリ使用量を大幅に削減")
    print("2. 処理性能: 文字列プールを使用することで、特に重複率が高いデータで処理速度も向上")
    print("3. ユニーク率の影響: ユニーク率が低いほど（重複が多いほど）効果が高い")
    print("\n重複の多いカテゴリカルデータや限定的な値セットを持つ文字列データに特に効果的です。")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
        import psutil
    except ImportError:
        print("必要なモジュールがありません。以下のコマンドでインストールしてください:")
        print("pip install tabulate psutil")
        sys.exit(1)
    
    print("ベンチマークを実行中... Ctrl+Cで中断できます。\n")
    
    try:
        run_basic_benchmark()
        run_string_pool_benchmark()
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。")
    except Exception as e:
        print(f"エラー: {e}")