import sys
import pandas as pd
import numpy as np
from src.PriceIntelligenceSystem.exception import CustomException
from src.PriceIntelligenceSystem.logger import logging
from src.PriceIntelligenceSystem.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = r'artifacts\\model.pkl'
            preprocessor_path = r'artifacts\\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 CRIM: float,
                 ZN: float,
                 INDUS:float,
                 CHAS: float,
                 NOX: float,
                 RM: float,
                 Age: float,
                 DIS: float,
                 RAD: float,
                 TAX: float,
                 PTRATIO: float,
                 B: float,
                 LSTAT: float):
        self.CRIM = CRIM
        self.ZN = ZN
        self.INDUS = INDUS
        self.CHAS = CHAS
        self.NOX = NOX
        self.RM = RM
        self.AGE = Age
        self.DIS = DIS
        self.RAD = RAD
        self.TAX = TAX
        self.PTRATIO = PTRATIO
        self.B = B
        self.LSTAT = LSTAT

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'CRIM':[self.CRIM], 
                'ZN':[self.ZN],
                'INDUS':[self.INDUS],
                'CHAS':[self.CHAS],
                'NOX':[self.NOX],
                'RM':[self.RM],
                'AGE':[self.AGE],
                'DIS':[self.DIS],
                'RAD':[self.RAD],
                'TAX':[self.TAX],
                'PTRATIO':[self.PTRATIO],
                'B':[self.B],
                'LSTAT':[self.LSTAT]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)