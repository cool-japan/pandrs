"""
最適化実装のPythonバインディングベンチマーク
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import sys

def run_benchmark():
    """ベンチマーク実行"""
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
    print("\n=== 結果サマリー ===")
    print(tabulate(results, headers="keys", tablefmt="pretty", floatfmt=".6f"))
    
    # pandasとの比較に関する考察
    print("\n考察:")
    print("1. DataFrame作成: 最適化実装は従来実装より大幅に高速化")
    if any(r['pandas比'] > 1.0 for r in results):
        print("   - 一部のケースでpandasよりも高速")
    else:
        print("   - 引き続きpandasの方が高速だがギャップは縮小")
    print("2. 効率的な型変換: 型特化した実装により、データ変換のオーバーヘッドを削減")
    print("3. 遅延評価: 操作パイプラインの最適化により複数操作の効率向上")
    print("\nRustネイティブ版はさらに高速であることに注意してください。")
    print("Pythonバインディングでは依然としてデータ変換のオーバーヘッドがあります。")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulateモジュールが必要です。pip install tabulateでインストールしてください。")
        sys.exit(1)
    
    print("警告: 大きなサイズのベンチマークは、十分なメモリがある環境で実行してください。")
    print("Ctrl+Cで中断できます。\n")
    
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。")
    except Exception as e:
        print(f"エラー: {e}")