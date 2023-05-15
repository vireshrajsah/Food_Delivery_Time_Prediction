import os
import sys
import pandas as pd
import src.statics as statics
from dataclasses import dataclass
from src.utils.data_preparation_utils import *
from src.utils.data_transformation_utils import *
from src.utils.general_utils import *
from src.utils.model_training_utils import *
from src.logging import logging
from src.exceptions import CustomException
from sklearn.model_selection import train_test_split


@dataclass
class FilePaths:
    SRC_DATA_PATH   = statics.SRC_DATA_PATH
    TRAIN_DATA_PATH = statics.TRAIN_DATA_PATH
    TEST_DATA_PATH  = statics.TEST_DATA_PATH
    ARTIFACTS_DIR   = statics.ARTIFACTS_DIR

class DataIngestor:

    def __init__(self):
        self.FILE_PATHS = FilePaths()

    def ingest(self):
        try:
            # creating target directory if not present
            os.makedirs(self.FILE_PATHS.ARTIFACTS_DIR, exist_ok=True)

            logging.info("Data ingestion initiated in ingest() at data_ingestion.py")
            data = read_csv(self.FILE_PATHS.SRC_DATA_PATH)
            logging.info(f"Input file loaded and returned to ingest() at data_ingestion.py, Dataframe shape: {data.shape}\n{data.head()}")

            train, test = train_test_split(data, train_size=0.7)
            logging.info(f"Train_test_split achieved - train shape: [{train.shape}], test shape [{test.shape}]")

            save_csv(train, self.FILE_PATHS.TRAIN_DATA_PATH)
            logging.info("Train csv saved and confirmed from DataIngestor.ingest() at data_ingestion.py")
            save_csv(test, self.FILE_PATHS.TEST_DATA_PATH)
            logging.info("Test csv saved and confirmed from DataIngestor.ingest() at data_ingestion.py")

            logging.info("Data ingestion completed in DataIngestor.ingest() at data_ingestion.py")
            return self.FILE_PATHS.TRAIN_DATA_PATH, self.FILE_PATHS.TEST_DATA_PATH
        
        except Exception as e:
            logging.critical(CustomException(e,sys))
            raise CustomException(e,sys) from e