import os
import sys
from src.PriceIntelligenceSystem.exception import CustomException
from src.PriceIntelligenceSystem.logger import logging
from src.PriceIntelligenceSystem.utils import read_sql_data
from sklearn.model_selection import train_test_split
import pandas as pd


from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ## Readning the data from mySQL
            logging.info("Reading started MySQL Database...")
            # df = read_sql_data()
            df = pd.read_csv(os.path.join('artifacts', 'raw.csv'))
            logging.info("Reading completed MySQL Database...")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed...")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
        