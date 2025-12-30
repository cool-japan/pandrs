"""
Jupyter integration for PandRS
"""

import pandas as pd
from IPython.display import display
from typing import Union, Optional, Dict, Any

from .pandrs_python import DataFrame, Series, NASeries
from .compat import to_pandas_dataframe

def display_dataframe(df: DataFrame, max_rows: Optional[int] = None, max_cols: Optional[int] = None) -> None:
    """
    Display a PandRS DataFrame in a Jupyter notebook with rich formatting
    
    Parameters
    ----------
    df : pandrs.DataFrame
        The DataFrame to display
    max_rows : int, optional
        Maximum number of rows to display
    max_cols : int, optional
        Maximum number of columns to display
    """
    # Convert to pandas for rich display
    pd_df = to_pandas_dataframe(df)
    
    # Set display options
    with pd.option_context('display.max_rows', max_rows or 20, 
                          'display.max_columns', max_cols or 20):
        display(pd_df)
        
def setup_jupyter_integration() -> None:
    """
    Set up Jupyter notebook integration for PandRS
    
    This adds rich display capabilities for PandRS objects in Jupyter notebooks
    """
    try:
        from IPython import get_ipython
        from IPython.core.magic import register_line_magic
        
        ipython = get_ipython()
        if ipython is not None:
            # Register rich display formatters
            html_formatter = lambda df: to_pandas_dataframe(df)._repr_html_()
            ipython.display_formatter.formatters['text/html'].for_type(DataFrame, html_formatter)
            
            # Register line magic for PandRS
            @register_line_magic
            def pandrs(line):
                """PandRS integration commands for Jupyter"""
                if line == "version":
                    from . import __version__
                    print(f"PandRS version: {__version__}")
                elif line == "help":
                    print("PandRS Jupyter commands:")
                    print("  %pandrs version - Display PandRS version")
                    print("  %pandrs help - Display this help message")
                else:
                    print(f"Unknown command: {line}")
                    print("Use %pandrs help for available commands")
                    
            print("PandRS Jupyter integration enabled")
    except ImportError:
        print("IPython not available, Jupyter integration disabled")
        
# Automatically set up Jupyter integration when this module is imported
setup_jupyter_integration()