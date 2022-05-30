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

class Preprocesser:

    def __init__(self) -> None:
        pass

    
    def format_data(self, data: data, pre: bool = True) -> Union[None, data]:

        if pre:

            if isinstance(data, pd.DataFrame):
                self.df = True
                self.idx = data.index
                self.cols = data.columns
                data = data.to_numpy()
            else:
                self.df = False

            return data
        
        else: 

            if self.df:
                data = pd.DataFrame(data, index = self.idx, columns= self.cols)
            return data

    def preprocessing_pipeline(self, data: data) -> data:
        
        data = self.format_data(data)

        self.scaler = pp.StandardScaler().fit(data)
        data_scaled = self.scaler.transform(data)
        
        data = self.format_data(data_scaled, pre= False)
        return data

    

    def postprocessing_pipeline(self, data: data) -> data:

            data = self.format_data(data)
            data = self.scaler.inverse_transform(data)
            return self.format_data(data, pre=False)
    