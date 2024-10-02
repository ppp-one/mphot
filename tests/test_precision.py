import pytest
import numpy as np
import pandas as pd
import mphot

def test_interpolate_dfs():
    # Create sample DataFrames
    df1 = pd.DataFrame({'A': [1, 3, 5]}, index=[0, 2, 4])
    df2 = pd.DataFrame({'B': [2, 4, 6]}, index=[1, 3, 5])
    index = [0, 1, 2, 3, 4, 5]
    
    result = mphot.interpolate_dfs(index, df1, df2)
    
    assert list(result.index) == index
    assert list(result['A']) == [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
    assert list(result['B']) == [2.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def test_integration_time():
    t = mphot.integration_time(1.0, 1000, 100, 10, 0.5, 100000, 0.8)
    assert t > 0  # Integration time should be positive


def test_get_precision():
    