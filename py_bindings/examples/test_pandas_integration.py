#!/usr/bin/env python3
"""
Pythonバインディングの基本テスト
"""

import os
import sys
import numpy as np
import pandas as pd

# パスを追加してimport可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import pandrs as pr
    print(f"PandRS version: {pr.__version__}")
except ImportError:
    print("PandRSモジュールをインポートできません。先に 'cd py_bindings && pip install -e .' を実行してください。")
    sys.exit(1)

def test_dataframe_creation():
    """DataFrame作成のテスト"""
    print("\n=== DataFrame作成テスト ===")
    
    # データ準備
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    
    # DataFrameの作成
    df = pr.DataFrame(data)
    print(f"作成されたDataFrame:\n{df}")
    print(f"形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # Pandasとの互換性
    pd_df = df.to_pandas()
    print(f"\nPandas DataFrameに変換:\n{pd_df}")
    
    # 列アクセス
    series_a = df['A']
    print(f"\n列Aの取得: {series_a}")
    
    return df

def test_io_operations(df):
    """入出力操作のテスト"""
    print("\n=== 入出力操作テスト ===")
    
    # CSVに保存
    csv_path = "test_dataframe.csv"
    df.to_csv(csv_path)
    print(f"CSVとして保存: {csv_path}")
    
    # CSVから読み込み
    df_loaded = pr.DataFrame.read_csv(csv_path)
    print(f"CSVから読み込まれたDataFrame:\n{df_loaded}")
    
    # JSONに変換
    json_str = df.to_json()
    print(f"JSON文字列: {json_str[:100]}...")
    
    # JSONから読み込み
    df_from_json = pr.DataFrame.read_json(json_str)
    print(f"JSONから読み込まれたDataFrame:\n{df_from_json}")
    
    # ファイル削除
    os.remove(csv_path)
    
    return df_from_json

def test_series_operations():
    """Series操作のテスト"""
    print("\n=== Series操作テスト ===")
    
    # シリーズの作成
    series = pr.Series("test_series", ["a", "b", "c", "d", "e"])
    print(f"作成されたSeries: {series}")
    
    # 値の取得
    values = series.values
    print(f"値: {values}")
    
    # 名前の取得と設定
    print(f"Series名: {series.name}")
    series.name = "renamed_series"
    print(f"名前変更後のSeries名: {series.name}")
    
    # NumPy配列への変換
    # 数値Seriesの場合
    num_series = pr.Series("numbers", ["1", "2", "3", "4", "5"])
    np_array = num_series.to_numpy()
    print(f"NumPy配列: {np_array}")
    print(f"配列型: {type(np_array)}")
    
    return series

def test_na_series():
    """NASeries操作のテスト"""
    print("\n=== NASeries操作テスト ===")
    
    # NAを含むシリーズの作成
    data = [None, "b", None, "d", "e"]
    na_series = pr.NASeries("na_test", data)
    print(f"NAを含むシリーズ: {na_series}")
    
    # NA値の検出
    is_na = na_series.isna()
    print(f"NA値のマスク: {is_na}")
    
    # NA値の削除
    dropped = na_series.dropna()
    print(f"NA削除後: {dropped}")
    
    # NA値の埋め合わせ
    filled = na_series.fillna("FILLED")
    print(f"NA埋め合わせ後: {filled}")
    
    return na_series

def main():
    """メイン関数"""
    print("PandRS Python Bindings Test")
    
    df = test_dataframe_creation()
    df_json = test_io_operations(df)
    series = test_series_operations()
    na_series = test_na_series()
    
    print("\n=== すべてのテスト完了 ===")

if __name__ == "__main__":
    main()