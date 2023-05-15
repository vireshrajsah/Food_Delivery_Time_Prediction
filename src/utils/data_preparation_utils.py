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

from src.utils.data_transformation_utils import *
from src.utils.general_utils import *
from src.utils.model_training_utils import *



# DATA PREPRATION FUNCTIONS   ---------------------------------------------------------------------------------------------

def clean(df:pd.DataFrame)-> pd.DataFrame:
    try:
        logging.info(f"df RECEIVED for cleaning in clean() at data_preparation_utils.py, shape: {df.shape}")
        df.drop(['ID'], axis=1, inplace=True)
        logging.info("ID column dropped")

        HM_PAT  = statics.HM_PAT    # HH:MM Format                      --> re.compile(r'^\d\d\:\d\d$')
        HMS_PAT = statics.HMS_PAT   # HH:MM:SS Format                   --> re.compile(r'^\d\d\:\d\d\:\d\d$')
        DOT_PAT = statics.DOT_PAT   # Faulty pattern observed in Data   --> re.compile(r'\d+\.\d+')
        
        df_clean = df[df.Time_Orderd.str.contains(HM_PAT)|df.Time_Orderd.str.contains(HMS_PAT)]
        df_clean = df_clean[(df_clean.Time_Order_picked.str.contains(HM_PAT)|df_clean.Time_Order_picked.str.contains(HMS_PAT))]
        logging.info("Records with faulty time dropped")
        
        logging.info(f"df cleaning COMPLETE in clean() at data_preparation_utils.py, shape: {df_clean.shape}, and returned\n {df_clean.head()}")

        return df_clean

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
    
def get_hour(time_str:str):

    try:
        HM_PAT  = statics.HM_PAT    # HH:MM Format          --> re.compile(r'^\d\d\:\d\d$')
        HMS_PAT = statics.HMS_PAT   # HH:MM:SS Format       --> re.compile(r'^\d\d\:\d\d\:\d\d$')
        
        if time_str.startswith('24'):
            time_str = '00'+time_str[2:]
        else:
            pass

        if re.fullmatch(HM_PAT, time_str):
            hour = datetime.strptime(time_str, '%H:%M').hour
        elif re.fullmatch(HMS_PAT, time_str):
            hour = datetime.strptime(time_str, '%H:%M:%S').hour
        else:
            hour = None

        return hour

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
    
def get_great_circle_distance(pickup_coordinates:pd.DataFrame,
                              drop_coordinates:pd.DataFrame):
    try:
        dist_arr = list()

        for i in range(len(pickup_coordinates)):
            dist_arr.append(geodesic(pickup_coordinates.to_numpy()[i], drop_coordinates.to_numpy()[i]).km)
        
        return dist_arr

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
    
def apply_feature_engineering(X:pd.DataFrame)-> pd.DataFrame:
    try:
        logging.info(f"df received for Feature engineering, shape: {X.shape} in apply_feature_engineering() at data_preparation_utils.py")
        X['Time_Orderd'] = X.Time_Orderd.apply(get_hour)
        X['Time_Order_picked'] = X.Time_Order_picked.apply(get_hour)
        logging.info("Time_Orderd and Time_Order_picked columns converted to hours")

        X['Order_Day'] = X.Order_Date.apply(lambda x : datetime.strptime(x, '%d-%m-%Y').day)
        X['Order_Month'] = X.Order_Date.apply(lambda x : datetime.strptime(x, '%d-%m-%Y').month)
        X.drop(['Order_Date'], axis=1, inplace=True)
        logging.info("Order_Date split to Order_Day and Order_Month")

        X['Geo_Distance'] = get_great_circle_distance(
            X[['Restaurant_latitude','Restaurant_longitude']],
            X[['Delivery_location_latitude', 'Delivery_location_longitude']]
        )
        logging.info("Geo_Distance column added")

        logging.info(f"df feature engineering steps complete in apply_feature_engineering() at data_preparation_utils.py, shape: {X.shape}, and returned")

        return X

    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e