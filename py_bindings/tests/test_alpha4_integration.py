#!/usr/bin/env python3
"""
Integration tests for PandRS alpha.4 Python bindings.

This test suite validates that all major alpha.4 features work correctly
through the Python interface.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Import pandrs modules (assuming they're built and available)
try:
    import pandrs
    from pandrs import DataFrame, OptimizedDataFrame, Series, NASeries
    PANDRS_AVAILABLE = True
except ImportError:
    PANDRS_AVAILABLE = False
    

@pytest.mark.skipif(not PANDRS_AVAILABLE, reason="PandRS not available")
class TestAlpha4PythonIntegration:
    """Test alpha.4 features through Python bindings."""
    
    def test_dataframe_rename_columns(self):
        """Test DataFrame.rename_columns() method (alpha.4 feature)."""
        # Create test DataFrame
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Carol'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Test rename_columns
        rename_map = {
            'name': 'employee_name',
            'age': 'employee_age'
        }
        df.rename_columns(rename_map)
        
        # Verify rename worked
        columns = df.columns
        assert 'employee_name' in columns
        assert 'employee_age' in columns
        assert 'salary' in columns  # Unchanged
        assert 'name' not in columns
        assert 'age' not in columns
        
        # Verify data integrity
        assert df.shape == (3, 3)
    
    def test_dataframe_set_columns(self):
        """Test DataFrame.columns setter (alpha.4 feature)."""
        # Create test DataFrame
        df = DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        # Set all column names
        new_columns = ['a', 'b', 'c']
        df.columns = new_columns
        
        # Verify all names changed
        assert list(df.columns) == new_columns
        assert df.shape == (3, 3)
    
    def test_optimized_dataframe_alpha4_features(self):
        """Test OptimizedDataFrame alpha.4 features."""
        # Create OptimizedDataFrame
        df = OptimizedDataFrame()
        
        # Add columns of different types
        df.add_int_column('id', [1, 2, 3, 4])
        df.add_float_column('value', [1.1, 2.2, 3.3, 4.4])
        df.add_string_column('name', ['A', 'B', 'C', 'D'])
        df.add_boolean_column('active', [True, False, True, False])
        
        # Test rename_columns
        rename_map = {
            'id': 'identifier',
            'value': 'metric',
            'name': 'label'
        }
        df.rename_columns(rename_map)
        
        # Verify renames
        columns = df.column_names
        assert 'identifier' in columns
        assert 'metric' in columns
        assert 'label' in columns
        assert 'active' in columns  # Unchanged
        assert 'id' not in columns
        assert 'value' not in columns
        assert 'name' not in columns
        
        # Test set_column_names
        new_names = ['col1', 'col2', 'col3', 'col4']
        df.set_column_names(new_names)
        
        # Verify all names changed
        assert df.column_names == new_names
        assert df.shape == (4, 4)
    
    @pytest.mark.skipif(not hasattr(pandrs, 'read_parquet'), reason="Parquet support not available")
    def test_parquet_io_integration(self):
        """Test enhanced Parquet I/O (alpha.4 feature)."""
        # Create test data
        df = OptimizedDataFrame()
        df.add_string_column('product', ['Widget A', 'Widget B', 'Widget C'])
        df.add_int_column('quantity', [10, 20, 30])
        df.add_float_column('price', [9.99, 19.99, 29.99])
        df.add_boolean_column('in_stock', [True, False, True])
        
        # Test write/read cycle
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            try:
                # Write with compression
                df.to_parquet(tmp.name, compression='snappy')
                
                # Verify file exists
                assert Path(tmp.name).exists()
                
                # Read back
                loaded_df = OptimizedDataFrame.from_parquet(tmp.name)
                
                # Verify data integrity
                assert loaded_df.shape == df.shape
                assert loaded_df.column_names == df.column_names
                
            finally:
                # Clean up
                if Path(tmp.name).exists():
                    os.unlink(tmp.name)
    
    @pytest.mark.skipif(not hasattr(pandrs, 'read_sql'), reason="SQL support not available")
    def test_sql_io_integration(self):
        """Test enhanced SQL I/O (alpha.4 feature)."""
        # Create test data
        df = DataFrame({
            'customer_id': [1, 2, 3],
            'customer_name': ['John', 'Jane', 'Bob'],
            'order_amount': [100, 200, 150]
        })
        
        # Test write/read cycle
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            try:
                # Write to SQL
                df.to_sql('customers', tmp.name, if_exists='replace')
                
                # Verify file exists
                assert Path(tmp.name).exists()
                
                # Read back with query
                loaded_df = DataFrame.from_sql('SELECT * FROM customers', tmp.name)
                
                # Verify data integrity
                assert loaded_df.shape[0] == df.shape[0]  # Same number of rows
                assert len(loaded_df.columns) >= 3  # At least 3 columns
                
                # Test filtered query
                filtered_df = DataFrame.from_sql(
                    'SELECT * FROM customers WHERE order_amount > 120', 
                    tmp.name
                )
                assert filtered_df.shape[0] == 2  # John and Jane
                
            finally:
                # Clean up
                if Path(tmp.name).exists():
                    os.unlink(tmp.name)
    
    def test_pandas_integration_with_alpha4_features(self):
        """Test pandas integration with alpha.4 features."""
        # Create pandas DataFrame
        pd_df = pd.DataFrame({
            'department': ['Eng', 'Sales', 'Marketing'],
            'employees': [10, 5, 3],
            'budget': [100000.0, 50000.0, 30000.0]
        })
        
        # Convert to PandRS OptimizedDataFrame
        pandrs_df = OptimizedDataFrame.from_pandas(pd_df)
        
        # Use alpha.4 features
        rename_map = {
            'department': 'dept',
            'employees': 'headcount'
        }
        pandrs_df.rename_columns(rename_map)
        
        # Convert back to pandas
        result_pd = pandrs_df.to_pandas()
        
        # Verify integration
        assert 'dept' in result_pd.columns
        assert 'headcount' in result_pd.columns
        assert 'budget' in result_pd.columns
        assert len(result_pd) == 3
    
    def test_series_operations_with_alpha4(self):
        """Test Series operations in alpha.4 context."""
        # Create Series
        series = Series('test_series', ['A', 'B', 'C', 'D'])
        
        # Test name operations (alpha.4 feature)
        assert series.name == 'test_series'
        
        series.name = 'renamed_series'
        assert series.name == 'renamed_series'
        
        # Test values
        values = series.values
        assert values == ['A', 'B', 'C', 'D']
    
    def test_na_series_alpha4_integration(self):
        """Test NASeries with alpha.4 features."""
        # Create NASeries with None values
        data = ['A', None, 'C', None, 'E']
        na_series = NASeries('test_na', data)
        
        # Test NA detection
        na_mask = na_series.isna()
        assert len(na_mask) == 5
        assert na_mask[1] == True   # Second element is NA
        assert na_mask[3] == True   # Fourth element is NA
        assert na_mask[0] == False  # First element is not NA
        
        # Test dropna
        dropped = na_series.dropna()
        assert dropped.name == 'test_na'  # Name preserved
        
        # Test fillna
        filled = na_series.fillna('MISSING')
        assert filled.name == 'test_na'  # Name preserved
    
    def test_error_handling_alpha4(self):
        """Test error handling for alpha.4 features."""
        df = DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Test rename with non-existent column
        with pytest.raises(Exception):
            df.rename_columns({'nonexistent': 'new_name'})
        
        # Test set_columns with wrong number of names
        with pytest.raises(Exception):
            df.columns = ['only_one_name']  # Should have 2 names
        
        # Test OptimizedDataFrame errors
        opt_df = OptimizedDataFrame()
        opt_df.add_int_column('col1', [1, 2, 3])
        
        with pytest.raises(Exception):
            opt_df.rename_columns({'nonexistent': 'new_name'})
        
        with pytest.raises(Exception):
            opt_df.set_column_names(['wrong_count', 'too_many'])  # Only 1 column exists
    
    def test_performance_characteristics_alpha4(self):
        """Test performance characteristics of alpha.4 features."""
        import time
        
        # Create larger dataset
        size = 1000
        df = OptimizedDataFrame()
        
        df.add_int_column('id', list(range(size)))
        df.add_string_column('category', [f'Cat_{i%10}' for i in range(size)])
        df.add_float_column('value', [float(i) * 1.5 for i in range(size)])
        df.add_boolean_column('flag', [i % 2 == 0 for i in range(size)])
        
        # Test rename performance
        start_time = time.time()
        
        rename_map = {
            'id': 'identifier',
            'category': 'group',
            'value': 'metric',
            'flag': 'active'
        }
        df.rename_columns(rename_map)
        
        rename_time = time.time() - start_time
        
        # Should be fast (under 1 second for 1000 rows)
        assert rename_time < 1.0
        
        # Test set_column_names performance
        start_time = time.time()
        
        new_names = ['col1', 'col2', 'col3', 'col4']
        df.set_column_names(new_names)
        
        set_names_time = time.time() - start_time
        
        # Should also be fast
        assert set_names_time < 1.0
        
        # Verify data integrity
        assert df.shape == (size, 4)
    
    @pytest.mark.skipif(not hasattr(pandrs, 'gpu'), reason="GPU support not available")
    def test_gpu_integration_alpha4(self):
        """Test GPU integration with alpha.4 features."""
        # This test assumes GPU module is available
        try:
            from pandrs.gpu import GpuConfig, init_gpu
            
            # Initialize GPU
            status = init_gpu()
            
            if status.available:
                # Create test data
                df = OptimizedDataFrame()
                df.add_float_column('x', [1.0, 2.0, 3.0, 4.0])
                df.add_float_column('y', [2.0, 4.0, 6.0, 8.0])
                
                # Use alpha.4 features with GPU
                rename_map = {'x': 'feature1', 'y': 'feature2'}
                df.rename_columns(rename_map)
                
                # Try GPU acceleration (if implemented)
                gpu_df = df.gpu_accelerate()
                
                # Verify basic properties
                assert gpu_df.shape == df.shape
                assert gpu_df.column_names == df.column_names
            
        except ImportError:
            pytest.skip("GPU module not available")


# Additional utility functions for testing
def create_test_dataframe():
    """Create a standard test DataFrame for consistent testing."""
    return DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
        'department': ['Eng', 'Sales', 'Eng', 'Marketing', 'Sales'],
        'salary': [75000, 65000, 80000, 55000, 70000],
        'active': [True, True, False, True, True]
    })


def create_test_optimized_dataframe():
    """Create a standard test OptimizedDataFrame."""
    df = OptimizedDataFrame()
    df.add_int_column('id', [1, 2, 3, 4, 5])
    df.add_string_column('name', ['Alice', 'Bob', 'Carol', 'David', 'Eve'])
    df.add_string_column('department', ['Eng', 'Sales', 'Eng', 'Marketing', 'Sales'])
    df.add_int_column('salary', [75000, 65000, 80000, 55000, 70000])
    df.add_boolean_column('active', [True, True, False, True, True])
    return df


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])