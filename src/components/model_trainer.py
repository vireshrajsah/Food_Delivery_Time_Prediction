import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import src.statics as statics
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.utils.data_preparation_utils import *
from src.utils.data_transformation_utils import *
from src.utils.general_utils import *
from src.utils.model_training_utils import *
from src.logging import logging
from src.exceptions import CustomException


@dataclass
class Paths:
    ARTIFACTS_DIR   = statics.ARTIFACTS_DIR
    MODEL_FILENAME  = statics.MODEL_FILE

    MODEL_FILE      = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)  # full path of model file (relative)

class Trainer:
    def __init__(self,
                 X_train:pd.DataFrame,
                 X_test:pd.DataFrame,
                 y_train:pd.DataFrame,
                 y_test:pd.DataFrame):
        
        self.X_train= X_train
        self.X_test = X_test
        self.y_train= y_train
        self.y_test = y_test

        self.PATHS = Paths()

    def train(self):
        try:
            logging.info("model training INITIATED in Trainer.train() at model_trainer.py")

            algos = {'LinearRegression':LinearRegression(),
                    'Ridge':Ridge(),
                    'Lasso':Lasso(),
                    'ElasticNet':ElasticNet()}
            algo_performance_report = pd.DataFrame(index=['obj', 'mae', 'mse', 'r2s'])
            
            logging.info("looping over all algorithms initiated")
            for algo_name, algo_obj in algos.items():

                logging.info(f"CALLING on utils to evaluate algorithm [{algo_name}] from Trainer.train() at model_trainer.py")
                mae, mse, r2s = model_report(algo_obj, self.X_train, self.X_test, self.y_train, self.y_test)
                algo_performance_report[algo_name] = [algo_obj, mae, mse, r2s]
                logging.info(f"RETURNED to Trainer.train() at model_trainer.py algorithm after model evaluation of [{algo_name}] with object [{algo_obj}, r2_score is [{r2s}]] ")
                logging.info("algo performance report updated")

            # Publish Algo performance Report
            logging.info(f"CALLING on utils to publish algo performance report from Trainer.train() at model_trainer.py")
            publish_algo_performance_report(algo_performance_report)
            logging.info("Algorithm performance report PUBLISHED")

            best_algo_model = algo_performance_report.T.sort_values(by='r2s', ascending= False)['obj'][0]
            best_algo_name = algo_performance_report.T.sort_values(by='r2s', ascending= False)['obj'].index[0]
            logging.info(f"best algorithm evaluated to be [{best_algo_name}] with object type [{type(best_algo_model)}]")

            model = best_algo_model
            model.fit(self.X_train, self.y_train)
            logging.info(f"best model: [{best_algo_name}] fitted with parameters [{model.get_params()}]")

            logging.info("CALLING on utils to pickle best model")
            pickle_this_object(model, self.PATHS.MODEL_FILE)
            logging.info(f"RETURNED to to Trainer.train() at model_trainer.py with best model pickled at {self.PATHS.MODEL_FILE}")

            logging.info("model training COMPLETE in Trainer.train() at model_trainer.py")


        except Exception as e:
            logging.critical(CustomException(e,sys))
            raise CustomException(e,sys) from e