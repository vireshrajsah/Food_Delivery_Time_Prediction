import os
import sys
import pandas as pd
import numpy as np
import pickle
import shelve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from src.utils.data_preparation_utils import *
from src.utils.data_transformation_utils import *
from src.utils.general_utils import *
from src.utils.model_training_utils import *
from src.logging import logging
from src.exceptions import CustomException



class Preprocessor:
    def __init__(self, trainpath:str, testpath:str) -> None:
        self.trainpath  = trainpath
        self.testpath   = testpath

    def transform(self):
        try:
            logging.info("transformation INITIATED in Preprocessor.transform() at data_transformation.py")

            train = read_csv(self.trainpath)
            test  = read_csv(self.testpath)
            target_column = statics.TARGET_COLUMN
            logging.info(f"train and test csv loaded, train shape: {train.shape}, test shape: {test.shape}")

            # Datacleaning
            logging.info(f"CALLING on utils to clean train and test dfs in Preprocessor.transform() at data_transformation.py, train shape: {train.shape}, test shape: {test.shape}")
            train   = clean(train)
            test    = clean(test)
            logging.info(f"clean train and test dfs RECEIVED in Preprocessor.transform() at data_transformation.py, train shape: {train.shape}, test shape: {test.shape}")

            # Feature Engineering
            logging.info(f"CALLING on utils to Feature Engineer train and test dfs in Preprocessor.transform() at data_transformation.py, train shape: {train.shape}, test shape: {test.shape}")
            train   = apply_feature_engineering(train)
            test    = apply_feature_engineering(test)
            logging.info(f"Feature Engineered train and test dfs RECEIVED in Preprocessor.transform() at data_transformation.py, train shape: {train.shape}, test shape: {test.shape}")

            # X-y split
            X_train = train.drop(target_column, axis=1)
            y_train = train[target_column]
            X_test  = test.drop(target_column, axis=1)
            y_test  = test[target_column]
            logging.info("X-y split achieved")

            # Data Transformation
            logging.info(f"CALLING on utils to transform train and test dfs in Preprocessor.transform() at data_transformation.py, train shape: {train.shape}, test shape: {test.shape}")
            X_train, X_test = apply_data_transformations(X_train, X_test)
            logging.info(f"Transformed train and test dfs RECEIVED in Preprocessor.transform() at data_transformation.py, train shape: {train.shape}, test shape: {test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.critical(CustomException(e,sys))
            raise CustomException(e,sys) from e