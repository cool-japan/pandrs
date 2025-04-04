"""
文字列プール最適化の簡易テスト
"""

import pandrs as pr
import sys

def test_string_pool():
    """文字列プールの基本機能テスト"""
    print("文字列プールの基本機能テスト...")
    
    # 文字列プールの作成
    string_pool = pr.StringPool()
    
    # 文字列の追加
    idx1 = string_pool.add("hello")
    idx2 = string_pool.add("world")
    idx3 = string_pool.add("hello")  # 重複した文字列
    
    # インデックスの確認
    print(f"インデックス1: {idx1}, インデックス2: {idx2}, インデックス3: {idx3}")
    if idx1 == idx3:
        print("✓ 重複文字列に同じインデックスが割り当てられました")
    else:
        print("✗ エラー: 重複文字列に異なるインデックスが割り当てられました")
    
    # 文字列の取得
    str1 = string_pool.get(idx1)
    str2 = string_pool.get(idx2)
    
    print(f"インデックス{idx1}の文字列: {str1}")
    print(f"インデックス{idx2}の文字列: {str2}")
    
    # 大量の文字列追加
    print("\nリスト追加テスト...")
    test_list = ["apple", "banana", "cherry", "apple", "banana", "date", "apple"]
    indices = string_pool.add_list(test_list)
    print(f"インデックスリスト: {indices}")
    
    # 一括取得
    retrieved = string_pool.get_list(indices)
    print(f"取得した文字列リスト: {retrieved}")
    if retrieved == test_list:
        print("✓ リストが正しく復元されました")
    else:
        print("✗ エラー: リストの復元に問題があります")
    
    # 統計情報の取得
    stats = string_pool.get_stats()
    print("\n文字列プール統計:")
    print(f"- 総文字列数: {stats['total_strings']}")
    print(f"- 一意な文字列数: {stats['unique_strings']}")
    print(f"- 重複文字列数: {stats['duplicated_strings']}")
    print(f"- 節約されたバイト数: {stats['bytes_saved']}")
    print(f"- 重複率: {stats['duplicate_ratio']:.2%}")

def test_optimized_dataframe():
    """最適化DataFrameでの文字列プール使用テスト"""
    print("\n最適化DataFrameでの文字列プール使用テスト...")
    
    # テストデータ
    ids = list(range(5))
    texts = ["apple", "banana", "apple", "cherry", "banana"]
    
    # 文字列プール使用のDataFrame作成
    df = pr.OptimizedDataFrame()
    df.add_int_column('id', ids)
    
    # 通常の文字列カラム追加
    df.add_string_column('text1', texts)
    
    # Python Listから直接追加
    df.add_string_column_from_pylist('text2', texts)
    
    # 結果表示
    print(f"最適化DataFrame: {df}")
    
    # pandasに変換して内容確認
    pd_df = df.to_pandas()
    print("\npandas DataFrame:")
    print(pd_df)
    
    # 再度変換して確認
    df2 = pr.OptimizedDataFrame.from_pandas(pd_df)
    print("\n再変換後の最適化DataFrame:")
    print(df2)

if __name__ == "__main__":
    try:
        # 基本的な文字列プール機能のテスト
        test_string_pool()
        
        # 最適化DataFrameでの文字列プール使用テスト
        test_optimized_dataframe()
        
        print("\nすべてのテストが完了しました!")
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        sys.exit(1)