import os
import sys
import pandas as pd
import numpy as np
import src.statics as statics
from src.utils.data_preparation_utils import *
from src.utils.data_transformation_utils import *
from src.utils.general_utils import *
from src.utils.model_training_utils import *
from src.logging import logging
from src.exceptions import CustomException

class Predictor:
    def __init__(self, input:dict) -> None:
        self.input = input

    def ingest(self):
        try:
            logging.info("Input values ingestion INITIATED")

            # LOADING to DataFrame
            input_dataframe = pd.DataFrame(self.input, index = ['values'])
            logging.info("input dictionary loaded as a dataframe")

            # CLEAN(X)
            logging.info("CALLING utils for cleaning in Predictor.ingest at prediction.py")
            input_dataframe = clean(input_dataframe)
            logging.info(f"RECEIVED \n{input_dataframe}")
            logging.info(f"RECEIVED input dataframe cleaned, with shape {input_dataframe.shape}")

            # APPLY_FE(X)
            logging.info(f"CALLING utils for feature engineering in Predictor.ingest at prediction.py, shape {input_dataframe.shape}")
            input_dataframe = apply_feature_engineering(input_dataframe)
            logging.info("RECEIVED input dataframe feature engineered")

            # TRANSFORM(X)
            logging.info("CALLING utils for transformation in Predictor.ingest at prediction.py")
            input_dataframe = fit_to_transformations(input_dataframe)
            logging.info("RECEIVED input dataframe transformed")

            self.input = input_dataframe
            logging.info("data ingestion and transformation COMPLETED in Predictor.ingest at prediction.py")

            return input_dataframe
            
        except Exception as e:
            logging.critical(CustomException(e,sys))
            raise CustomException(e,sys) from e


    def predict(self):
        try:
            logging.info("prediction INITIATED in Predictor.predict() at prediction.py")
            X=self.input

            model = load_model()
            logging.info(f"model with type [{type(model)}] and parameters [{model.get_params()} loaded in Predictor.predict() at prediction.py]")
            
            y = model.predict(X)
            logging.info(f"output for: \n[{X}]\n predicted as [{y}]; Prediction COMPLETED in Predictor.predict() at prediction.py")

            return y
        
        except Exception as e:
            logging.critical(CustomException(e,sys))
            raise CustomException(e,sys) from e