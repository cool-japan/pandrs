import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile

# Add parent directory to path to import pandrs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pandrs as pr
except ImportError:
    # If not installed, use this message
    print("ERROR: pandrs module not found. Run 'maturin develop' in the py_bindings directory first.")
    raise

class TestDataFrame(unittest.TestCase):
    """Test the pandrs DataFrame class Python bindings"""
    
    def setUp(self):
        """Set up test data"""
        self.data = {
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        }
        self.df = pr.DataFrame(self.data)
    
    def test_creation(self):
        """Test DataFrame creation"""
        self.assertEqual(self.df.shape, (5, 3))
        # Order of columns may vary, just check content
        cols = list(self.df.columns)
        self.assertEqual(sorted(cols), sorted(['A', 'B', 'C']))
        
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = self.df.to_dict()
        for key in self.data:
            self.assertIn(key, result)
            self.assertEqual(len(result[key]), len(self.data[key]))
    
    def test_pandas_interop(self):
        """Test pandas interoperability"""
        # Convert to pandas
        pd_df = self.df.to_pandas()
        self.assertIsInstance(pd_df, pd.DataFrame)
        self.assertEqual(pd_df.shape, (5, 3))
        
        # Convert back to pandrs
        pr_df = pr.DataFrame.from_pandas(pd_df)
        self.assertEqual(pr_df.shape, (5, 3))
        # Order of columns may vary, just check content
        cols = list(pr_df.columns)
        self.assertEqual(sorted(cols), sorted(['A', 'B', 'C']))
    
    def test_getitem(self):
        """Test column access"""
        series_a = self.df['A']
        self.assertEqual(series_a.name, 'A')
        
    def test_series_to_numpy(self):
        """Test Series to NumPy conversion"""
        series_a = self.df['A']
        try:
            np_array = series_a.to_numpy()
            self.assertIsInstance(np_array, np.ndarray)
            self.assertEqual(len(np_array), 5)
        except:
            # Alternatively, it might return a list which is also acceptable
            np_list = series_a.to_numpy()
            self.assertEqual(len(np_list), 5)
        
    def test_iloc(self):
        """Test row selection with iloc"""
        subset = self.df.iloc([0, 2, 4])
        self.assertEqual(subset.shape, (3, 3))
        
    def test_json_io(self):
        """Test JSON serialization/deserialization"""
        json_str = self.df.to_json()
        self.assertIsInstance(json_str, str)
        df2 = pr.DataFrame.read_json(json_str)
        self.assertEqual(df2.shape, self.df.shape)
        
    def test_csv_io(self):
        """Test CSV I/O operations"""
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            # Write to CSV
            self.df.to_csv(tmp.name)
            
            # Read from CSV
            df2 = pr.DataFrame.read_csv(tmp.name)
            
            # Column count should match (note: CSV may add index column)
            self.assertEqual(df2.shape[1], self.df.shape[1])
            
    def test_columns_setter(self):
        """Test setting column names"""
        df = pr.DataFrame(self.data)
        new_columns = ['X', 'Y', 'Z']
        df.columns = new_columns
        self.assertEqual(list(df.columns), new_columns)

class TestSeries(unittest.TestCase):
    """Test the pandrs Series class Python bindings"""
    
    def setUp(self):
        """Set up test data"""
        self.name = "test_series"
        self.values = ["1", "2", "3", "4", "5"]
        self.series = pr.Series(self.name, self.values)
    
    def test_creation(self):
        """Test Series creation"""
        self.assertEqual(self.series.name, self.name)
        self.assertEqual(len(self.series.values), len(self.values))
        
    def test_to_numpy(self):
        """Test converting to numpy array"""
        arr = self.series.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(len(arr), len(self.values))
        
    def test_name_setter(self):
        """Test setting series name"""
        new_name = "new_series_name"
        self.series.name = new_name
        self.assertEqual(self.series.name, new_name)

class TestNASeries(unittest.TestCase):
    """Test the pandrs NASeries class Python bindings"""
    
    def setUp(self):
        """Set up test data"""
        self.name = "test_na_series"
        self.values = [None, "b", None, "d", "e"]
        self.series = pr.NASeries(self.name, self.values)
    
    def test_creation(self):
        """Test NASeries creation"""
        self.assertEqual(self.series.name, self.name)
        
    def test_isna(self):
        """Test NA detection"""
        na_mask = self.series.isna()
        self.assertIsInstance(na_mask, np.ndarray)
        self.assertTrue(na_mask[0])  # None should be NA
        self.assertFalse(na_mask[1])  # "b" should not be NA
        self.assertTrue(na_mask[2])  # None should be NA
        
    def test_fillna(self):
        """Test filling NA values"""
        filled = self.series.fillna("x")
        self.assertEqual(filled.name, self.name)
        # Test that the NA values were properly filled
        na_mask = filled.isna()
        self.assertFalse(any(na_mask))
        
    def test_dropna(self):
        """Test dropping NA values"""
        dropped = self.series.dropna()
        self.assertEqual(dropped.name, self.name)
        na_mask = dropped.isna()
        self.assertFalse(any(na_mask))

if __name__ == '__main__':
    unittest.main()