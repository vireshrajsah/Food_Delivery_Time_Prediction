import os
import sys
import re
import numpy as np
import pandas as pd
import pickle
import shelve
import src.statics as statics
from datetime import datetime
from geopy.distance import geodesic
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from src.logging import logging
from src.exceptions import CustomException
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.data_preparation_utils import *
from src.utils.data_transformation_utils import *
from src.utils.model_training_utils import *


def read_csv(filepath:os.path)->pd.DataFrame:
    '''
    Parameters:
    :filepath: (str) Full path of csv file

    Returns: Pandas dataframe
    '''
    try:
        logging.info(f"filepath: [{filepath}] RECEIVED in general_utils.read_csv()")
        df = pd.read_csv(filepath)
        logging.info("csv read and LOADED")
        return df
    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
    
def save_csv(df:pd.DataFrame, 
             savepath:os.path)-> None:
    '''
    Takes DataFrame and saves it as csv
    Params:
    :df: Dataframe
    :savepath: (str) path where DataFrame has to be saved.
    '''
    try:
        logging.info(f"Dataframe RECEIVED for saving. Shape: {df.shape}")
        df.to_csv(savepath, index=False)
        logging.info(f"Dataframe SAVED as csv to path - [ {savepath} ]")

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e

def shelve_this_preprocessor(obj, identifier:str)->None:
    try:
        logging.info(f"{obj} RECEIVED for shelving in shelve_this_preprocessor() at general_utils.py")
        PREPROCESSOR_SHELVE = statics.PREPROCESSOR_SHELVE
        ARTIFACTS_DIR   = statics.ARTIFACTS_DIR
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        shelvefile = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_SHELVE)

        with shelve.open(shelvefile) as shelf:
            shelf[identifier] = obj

        logging.info(f"[{obj}] SHELVED with identifier [{identifier}] in shelve_this_preprocessor() at general_utils.py")

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
    
def get_preprocessor_from_shelve (identifier:str):
    try:
        logging.info(f"fetch request for [{identifier}] RECEIVED in get_preprocessor_from_shelve() at general_utils.py")
        PREPROCESSOR_SHELVE = statics.PREPROCESSOR_SHELVE
        ARTIFACTS_DIR   = statics.ARTIFACTS_DIR
        shelvefile = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_SHELVE)

        with shelve.open(shelvefile) as shelf:
            obj = shelf[identifier]

        logging.info(f"object [{obj}] of type [{type(obj)}] retrieved and RETURNED from preprosessor shelve")

        return obj
    
    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e


def pickle_this_object(obj, path:os.path)->None:
    try:
        logging.info(f"pickle request for [{obj}] RECEIVED in pickle_this_object() at general_utils.py")
        with open(path, 'wb') as pickle_file:
            pickle.dump(obj=obj, file=pickle_file)
        logging.info(f"[{obj}] PICKLED to path [{path}] in pickle_this_object() at general_utils.py")

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
    
def load_model():
    try:
        logging.info(f"fetch request for [model] RECEIVED in load_model() at general_utils.py")
        MODEL_FILE = statics.MODEL_FILE
        ARTIFACTS_DIR   = statics.ARTIFACTS_DIR
        model_pickle = os.path.join(ARTIFACTS_DIR, MODEL_FILE)

        with open(model_pickle, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        logging.info(f"model object of type [{type(model)}] retrived and RETURNED from load_model() at general_utils.py")
        return model

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e