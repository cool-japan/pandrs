"""
Simple test for string pool optimization
"""

import pandrs as pr
import sys

def test_string_pool():
    """Basic functionality test for the string pool"""
    print("Basic functionality test for the string pool...")
    
    # Create a string pool
    string_pool = pr.StringPool()
    
    # Add strings
    idx1 = string_pool.add("hello")
    idx2 = string_pool.add("world")
    idx3 = string_pool.add("hello")  # Duplicate string
    
    # Check indices
    print(f"Index 1: {idx1}, Index 2: {idx2}, Index 3: {idx3}")
    if idx1 == idx3:
        print("✓ Duplicate strings were assigned the same index")
    else:
        print("✗ Error: Duplicate strings were assigned different indices")
    
    # Retrieve strings
    str1 = string_pool.get(idx1)
    str2 = string_pool.get(idx2)
    
    print(f"String at index {idx1}: {str1}")
    print(f"String at index {idx2}: {str2}")
    
    # Add a large number of strings
    print("\nList addition test...")
    test_list = ["apple", "banana", "cherry", "apple", "banana", "date", "apple"]
    indices = string_pool.add_list(test_list)
    print(f"Index list: {indices}")
    
    # Bulk retrieval
    retrieved = string_pool.get_list(indices)
    print(f"Retrieved string list: {retrieved}")
    if retrieved == test_list:
        print("✓ The list was correctly restored")
    else:
        print("✗ Error: There was an issue restoring the list")
    
    # Retrieve statistics
    stats = string_pool.get_stats()
    print("\nString pool statistics:")
    print(f"- Total number of strings: {stats['total_strings']}")
    print(f"- Number of unique strings: {stats['unique_strings']}")
    print(f"- Number of duplicate strings: {stats['duplicated_strings']}")
    print(f"- Bytes saved: {stats['bytes_saved']}")
    print(f"- Duplication rate: {stats['duplicate_ratio']:.2%}")

def test_optimized_dataframe():
    """Test using the string pool with an optimized DataFrame"""
    print("\nTest using the string pool with an optimized DataFrame...")
    
    # Test data
    ids = list(range(5))
    texts = ["apple", "banana", "apple", "cherry", "banana"]
    
    # Create a DataFrame using the string pool
    df = pr.OptimizedDataFrame()
    df.add_int_column('id', ids)
    
    # Add a regular string column
    df.add_string_column('text1', texts)
    
    # Add directly from a Python list
    df.add_string_column_from_pylist('text2', texts)
    
    # Display results
    print(f"Optimized DataFrame: {df}")
    
    # Convert to pandas and check contents
    pd_df = df.to_pandas()
    print("\npandas DataFrame:")
    print(pd_df)
    
    # Convert back and check
    df2 = pr.OptimizedDataFrame.from_pandas(pd_df)
    print("\nOptimized DataFrame after reconversion:")
    print(df2)

if __name__ == "__main__":
    try:
        # Test basic string pool functionality
        test_string_pool()
        
        # Test using the string pool with an optimized DataFrame
        test_optimized_dataframe()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        sys.exit(1)