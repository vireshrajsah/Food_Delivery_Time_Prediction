import os
import sys
import pandas as pd
import numpy as np
import src.statics as statics
from src.logging import logging
from src.exceptions import CustomException
from src.components.prediction import *

def read_input():
    try:
        DIR_PATH = statics.INPUT_DIR
        FILENAME = statics.INPUT_CSV
        os.makedirs(DIR_PATH, exist_ok=True)

        INPUT_FILE = os.path.join(DIR_PATH, FILENAME)

        df = dict(pd.read_csv(INPUT_FILE).T[0])

        return df
    
    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e


if __name__ == "__main__":
    try:
        logging.info("PREDICTION PIPELINE INITIATED")

        input = read_input()
        print("INPUT:\t",input)
        logging.info(f"INPUT OBTAINED AS [{input}]")

        predictor = Predictor(input)
        predictor.ingest()
        logging.info("PREDICTOR INITIALIZED WITH INPUT")

        output = predictor.predict()
        print("OUTPUT:\t",output)
        logging.info(f"OUTPUT OBTAINED [{output}]")

        logging.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e