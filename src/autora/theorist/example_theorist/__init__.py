"""
Example Theorist
"""
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator




class ExampleRegressor(BaseEstimator):
    """
    Examples:
        >>> regressor = ExampleRegressor()
        >>> X = np.array([1,2,3])
        >>> y = np.array([1,3,3])
        >>> regressor.fit(X,y)
        >>> regressor.predict(X)
        np.array([1,3,3])


    """

    def __init__(self):
        pass

    def newfunc(self):
        pass

    def fit(self,
            conditions: Union[pd.DataFrame, np.ndarray],
            observations: Union[pd.DataFrame, np.ndarray]):
            """

            """
        return self

    def predict(self,
                conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass
