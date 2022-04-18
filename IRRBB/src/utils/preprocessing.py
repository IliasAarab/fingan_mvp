"""Preprocessing of data before ingesting into NNs"""


# Main libraries 
from typing import Union
import sklearn.preprocessing as pp
import pandas as pd
import numpy as np

# In-house modules
from .parser import Parser


#data = Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.DataFrame]]
data = Union[pd.DataFrame, np.ndarray]

def preprocessing_pipeline(data: data) -> data:
    

    if isinstance(data, pd.DataFrame):
        df = True
        idx = data.index
        cols = data.columns
        data = data.to_numpy()
    else:
        df = False

    scaler = pp.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    
    if df:
        data_scaled = pd.DataFrame(data_scaled, index = idx, columns= cols)
    return data_scaled

    
