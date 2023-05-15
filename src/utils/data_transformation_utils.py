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
from src.utils.general_utils import *
from src.utils.model_training_utils import *



#  DATA TRANSFORMATION FUNCTIONS   -------------------------------------------------------------------------------------------


def CustomEncode (X:pd.DataFrame, x, y, encoder_mapping:dict)-> pd.DataFrame:
    df = X
    df_x = x

    for encoder, columns in encoder_mapping.items():
        if isinstance(encoder, OrdinalEncoder):
            all_columns = list(df.columns)
            ordinal_columns = columns
            non_ordinal_columns = [k for k in all_columns if k not in ordinal_columns]

            transformer = ColumnTransformer([
                ('dummy step', SimpleImputer(strategy='most_frequent'), non_ordinal_columns),
                ('encode', encoder, ordinal_columns)
            ])
            transformer.fit(X)
            shelve_this_preprocessor()
            df = pd.DataFrame(transformer.transform(X), columns=non_ordinal_columns + ordinal_columns, index=X.index)
            df_x = pd.DataFrame(transformer.transform(x), columns=non_ordinal_columns + ordinal_columns, index=x.index)
            
        else:
            for column in columns:
                column_2D = [[value] for value in df[column]]
                column_2D_x = [[value] for value in df_x[column]]
                encoder.fit(column_2D)
                df[column] = encoder.transform (column_2D)
                df_x[column] = encoder.transform (column_2D_x)
    
    return df, df_x




def apply_data_transformations(X_train:pd.DataFrame, 
                               X_test:pd.DataFrame) ->pd.DataFrame:
    try:
        logging.info(f"dfs recieved for fit-transformation in apply_data_transformations() at data_transformation_utils.py, train_shape:{X_train.shape}, test_shape: {X_test.shape}")

        # Differentiating Columns
        numeric_columns = list(X_train.select_dtypes(np.number).columns)
        categoric_columns = statics.CATEGORIC_COLUMNS
        all_columns = numeric_columns + categoric_columns
        non_numeric_columns = [i for i in list(X_train.columns) if i not in numeric_columns]


        #  IMPUTATION
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        imputer = ColumnTransformer([
            ('numeric_imputer', num_imputer, numeric_columns),
            ('categoric_imputation', cat_imputer, non_numeric_columns)
        ])

        imputer.fit(X_train)
        shelve_this_preprocessor(imputer, statics.PRPRSSR_IDENTIFIER_IMPUTER)
        X_train = pd.DataFrame(imputer.transform(X_train), columns= numeric_columns + non_numeric_columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns= numeric_columns + non_numeric_columns, index=X_test.index)
        logging.info(f"imputation completed at data_transformation_utils.py, train_shape:{X_train.shape}, test_shape: {X_test.shape}\n number of train columns with null values [{(X_train.isna().sum()>0).sum()}], number of test columns with null values [{(X_test.isna().sum()>0).sum()}]")



        #  ENCODING
        for column in categoric_columns:
            le = LabelEncoder()
            identifier = statics.PRPRSSR_IDENTIFIER_DICT_ENCODER[column]
            le.fit(X_train[column])
            shelve_this_preprocessor(le, identifier=identifier)

            X_train[column] = le.transform(X_train[column])
            X_test[column]  = le.transform(X_test[column])
        logging.info(f"encoding completed at data_transformation_utils.py, train_shape:{X_train.shape}, test_shape: {X_test.shape}")
        logging.info(f'X_train:\n{X_train.head()}')


        
        #  SCALING
        scaler = ColumnTransformer([('scaler', StandardScaler(), list(X_train.columns))])
        scaler.fit(X_train)
        shelve_this_preprocessor(scaler, statics.PRPRSSR_IDENTIFIER_SCALER)

        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        logging.info("scaling completed at data_transformation_utils.py")

        logging.info(f"dfs fit-transformation COMPLETED in apply_data_transformations() at data_transformation_utils.py, train_shape:{X_train.shape}, test_shape: {X_test.shape}")
        return X_train, X_test
        
    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e




def fit_to_transformations(X:pd.DataFrame)-> pd.DataFrame:
    try:
        logging.info(f"df RECEIVED for transformation in fit_to_transformations() at data_transformation_utils.py, train_shape:{X.shape}")

        # Differentiating Columns
        numeric_columns = list(X.select_dtypes(np.number).columns)    
        categoric_columns = statics.CATEGORIC_COLUMNS
        all_columns = numeric_columns + categoric_columns
        non_numeric_columns = [i for i in list(X.columns) if i not in numeric_columns]


        #  IMPUTATION
        imputer = get_preprocessor_from_shelve(statics.PRPRSSR_IDENTIFIER_IMPUTER)

        X = pd.DataFrame(imputer.transform(X), columns= numeric_columns + non_numeric_columns, index=X.index)
        
        logging.info(f"imputation completed at data_transformation_utils.py, train_shape:{X.shape}\n number of train columns with null values [{(X.isna().sum()>0).sum()}]")



        #  ENCODING
        for column in categoric_columns:
            identifier = statics.PRPRSSR_IDENTIFIER_DICT_ENCODER[column]

            le = get_preprocessor_from_shelve(identifier)

            X[column] = le.transform(X[column])
            
        logging.info(f"encoding completed at data_transformation_utils.py, train_shape:{X.shape}")
        logging.info(f'X:\n{X.head()}')


        
        #  SCALING
        scaler = get_preprocessor_from_shelve(statics.PRPRSSR_IDENTIFIER_SCALER)

        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        
        logging.info("scaling completed at data_transformation_utils.py")

        logging.info(f"dfs fit-transformation COMPLETED in fit_to_transformations() at data_transformation_utils.py, train_shape:{X.shape}")
        return X
        
    except Exception as e:
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e
