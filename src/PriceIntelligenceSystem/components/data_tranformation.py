
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.PriceIntelligenceSystem.utils import save_object
from src.PriceIntelligenceSystem.exception import CustomException
from src.PriceIntelligenceSystem.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file...")

            preprocessing_obj = self.get_data_transformer_object()
            target_columns = "Price"
            numerical_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']
            logging.info('Divide the train Dataset into independent and dependent Features')
            input_features_train_df = train_df.drop(columns=[target_columns], axis=1)
            target_feature_train_df = train_df[target_columns]
            logging.info("Divide the test dataset to independent and dependent feature")
            input_features_test_df = test_df.drop(columns=[target_columns], axis=1)
            target_feature_test_df = test_df[target_columns]
            logging.info("Applying Preprocessing on training and test data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Saved Preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys, e)
