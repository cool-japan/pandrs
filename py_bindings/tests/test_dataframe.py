import unittest
import numpy as np
import pandas as pd
import os
import sys

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
        self.assertEqual(list(self.df.columns), ['A', 'B', 'C'])
        
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
        self.assertEqual(list(pr_df.columns), ['A', 'B', 'C'])
    
    def test_getitem(self):
        """Test column access"""
        series_a = self.df['A']
        self.assertEqual(series_a.name, 'A')
        
    def test_series_to_numpy(self):
        """Test Series to NumPy conversion"""
        series_a = self.df['A']
        np_array = series_a.to_numpy()
        self.assertIsInstance(np_array, np.ndarray)
        self.assertEqual(len(np_array), 5)

if __name__ == '__main__':
    unittest.main()