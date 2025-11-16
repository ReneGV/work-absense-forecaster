import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#Clase para eliminar columnas que no necesitamos
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    #Elimina la columnas en la lista que recibe
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        if columns_to_drop:
            X = X.drop(columns_to_drop, axis=1)
        return X
    #Regresa los nombres de los features que se quedan
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided")
        return [col for col in input_features if col not in self.columns_to_drop]

#Clase para tratar con outliers
class IQRClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

   #Eliminamos Outliers
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - 1.5 * IQR
            self.upper_bounds_[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_bounds_.get(col),
                                 upper=self.upper_bounds_.get(col))
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []


class ToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []
