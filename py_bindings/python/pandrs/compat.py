"""
Compatibility module for interoperability with NumPy and pandas
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Any

from .pandrs_python import DataFrame, Series, NASeries

def to_pandas_dataframe(df: DataFrame) -> pd.DataFrame:
    """
    Convert a PandRS DataFrame to a pandas DataFrame
    
    Parameters
    ----------
    df : pandrs.DataFrame
        PandRS DataFrame to convert
        
    Returns
    -------
    pandas.DataFrame
        Equivalent pandas DataFrame
    """
    return df.to_pandas()

def from_pandas_dataframe(pd_df: pd.DataFrame) -> DataFrame:
    """
    Convert a pandas DataFrame to a PandRS DataFrame
    
    Parameters
    ----------
    pd_df : pandas.DataFrame
        pandas DataFrame to convert
        
    Returns
    -------
    pandrs.DataFrame
        Equivalent PandRS DataFrame
    """
    return DataFrame.from_pandas(pd_df)

def series_to_numpy(series: Union[Series, NASeries]) -> np.ndarray:
    """
    Convert a PandRS Series to a NumPy array
    
    Parameters
    ----------
    series : Union[pandrs.Series, pandrs.NASeries]
        PandRS Series to convert
        
    Returns
    -------
    numpy.ndarray
        NumPy array containing the Series data
    """
    return series.to_numpy()