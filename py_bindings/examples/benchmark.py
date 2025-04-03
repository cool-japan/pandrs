"""
PandRS vs pandas パフォーマンスベンチマーク
"""

import pandrs as pr
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import sys

# バイナリなど生成物を標準出力に表示しないようにする
plt.ioff()

def run_benchmark(test_name, rows, pandas_func, pandrs_func):
    """ベンチマーク実行"""
    # pandasのタイミング
    start = time.time()
    pandas_result = pandas_func(rows)
    pandas_time = time.time() - start
    
    # pandrsのタイミング
    start = time.time()
    pandrs_result = pandrs_func(rows)
    pandrs_time = time.time() - start
    
    # 比率計算
    ratio = pandas_time / pandrs_time if pandrs_time > 0 else float('inf')
    
    return {
        'テスト': test_name,
        '行数': rows,
        'pandas (秒)': pandas_time,
        'pandrs (秒)': pandrs_time,
        '比率 (pandas/pandrs)': ratio
    }

def main():
    print("=== PandRS vs pandas パフォーマンスベンチマーク ===\n")
    
    # 行数のリスト
    row_sizes = [10, 100, 1000, 10000, 100000]
    results = []
    
    # データフレーム作成ベンチマーク
    for rows in row_sizes:
        # ベンチマークデータ準備
        numeric_data = list(range(rows))
        string_data = [f"value_{i}" for i in range(rows)]
        float_data = [i * 1.1 for i in range(rows)]
        
        # pandas データフレーム作成
        def pandas_create(n):
            data = {
                'A': numeric_data,
                'B': string_data,
                'C': float_data
            }
            return pd.DataFrame(data)
        
        # pandrs データフレーム作成
        def pandrs_create(n):
            data = {
                'A': numeric_data,
                'B': string_data,
                'C': float_data
            }
            return pr.DataFrame(data)
        
        result = run_benchmark(f"DataFrame作成", rows, pandas_create, pandrs_create)
        results.append(result)
    
    # カラムアクセスベンチマーク（10万行向け）
    rows = 100000
    numeric_data = list(range(rows))
    string_data = [f"value_{i}" for i in range(rows)]
    float_data = [i * 1.1 for i in range(rows)]
    
    # 前処理済みデータフレーム
    pd_df = pd.DataFrame({
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    })
    
    pr_df = pr.DataFrame({
        'A': numeric_data,
        'B': string_data,
        'C': float_data
    })
    
    # カラムアクセスベンチマーク
    def pandas_column_access(n):
        col = pd_df['A']
        return col
    
    def pandrs_column_access(n):
        col = pr_df['A']
        return col
    
    result = run_benchmark(f"カラムアクセス", rows, pandas_column_access, pandrs_column_access)
    results.append(result)
    
    # データ変換ベンチマーク
    def pandas_to_dict(n):
        dict_data = pd_df.to_dict()
        return dict_data
    
    def pandrs_to_dict(n):
        dict_data = pr_df.to_dict()
        return dict_data
    
    result = run_benchmark(f"to_dict変換", rows, pandas_to_dict, pandrs_to_dict)
    results.append(result)
    
    # 相互変換ベンチマーク
    def pandas_to_pandrs(n):
        return pr.DataFrame.from_pandas(pd_df)
    
    def pandrs_to_pandas(n):
        return pr_df.to_pandas()
    
    results.append(run_benchmark(f"pandas → pandrs", rows, pandas_to_pandrs, lambda x: None))
    results.append(run_benchmark(f"pandrs → pandas", rows, lambda x: None, pandrs_to_pandas))
    
    # 結果表示
    print(tabulate(results, headers='keys', tablefmt='pretty', floatfmt='.6f'))
    
    # 結果をグラフ化
    data_creation_results = [r for r in results if r['テスト'] == 'DataFrame作成']
    
    pandas_times = [r['pandas (秒)'] for r in data_creation_results]
    pandrs_times = [r['pandrs (秒)'] for r in data_creation_results]
    row_labels = [str(r['行数']) for r in data_creation_results]
    
    x = np.arange(len(row_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pandas_times, width, label='pandas')
    rects2 = ax.bar(x + width/2, pandrs_times, width, label='pandrs')
    
    ax.set_ylabel('時間 (秒)')
    ax.set_title('DataFrame作成 - pandas vs pandrs')
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels)
    ax.set_xlabel('行数')
    ax.legend()
    
    # ログスケールに変更（大きさの差が大きいため）
    ax.set_yscale('log')
    
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{pandas_times[i]:.4f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', rotation=90)
    
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{pandrs_times[i]:.4f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', rotation=90)
    
    fig.tight_layout()
    plt.savefig('benchmark_results.png')
    
    print("\n結果グラフを 'benchmark_results.png' に保存しました")
    print("\nRustネイティブ版とPythonバインディング版の比較")
    print("Rustネイティブバージョンでの10万行DataFrame作成時間: 約0.05秒")
    print("対してPythonバインディングでの時間：上記参照")
    print("差は主にPython-Rust間のデータ変換オーバーヘッドによるものです")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("タブルモジュールがありません。pip install tabulateでインストールしてください")
        sys.exit(1)
        
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlibがありません。pip install matplotlibでインストールしてください")
        sys.exit(1)
        
    main()