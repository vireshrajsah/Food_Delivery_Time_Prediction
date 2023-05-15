import os
import sys
import re
import numpy as np
import pandas as pd
import src.statics as statics
from datetime import datetime
from src.logging import logging
from src.exceptions import CustomException
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.data_preparation_utils import *
from src.utils.data_transformation_utils import *
from src.utils.general_utils import *


# MODEL TRAINING FUNCTIONS  -------------------------------------------------------------------------------------------------------

def model_report(model, xtrain, xtest, ytrain, ytest):
    try:
        model.fit(xtrain,ytrain)
        ypred = model.predict(xtest)

        mae = mean_absolute_error(ytest,ypred)
        mse = mean_squared_error(ytest,ypred)
        r2s = r2_score(ytest,ypred)

        return mae, mse, r2s

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e

def publish_algo_performance_report(df:pd.DataFrame):
    try:
        DIR = statics.ARTIFACTS_DIR
        FILENAME = statics.ALGO_REPORT_FILE
        FILE = os.path.join(DIR, FILENAME)

        df.to_excel(FILE, index=True)
        logging.info(f"Algo performance report published in path [{FILE}] in publish_algo_performance_report() at model_training_utils.py")

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e