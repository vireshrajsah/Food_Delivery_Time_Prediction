import os
import sys
import re


# PATH VARIABLES
SRC_DATA_PATH   = os.path.join(os.getcwd(), 'dataset', 'food_delivery.csv')
TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'artifacts', 'train.csv')
TEST_DATA_PATH  = os.path.join(os.getcwd(), 'artifacts', 'test.csv')
ARTIFACTS_DIR   = os.path.join(os.getcwd(), 'artifacts')
INPUT_DIR       = os.path.join(os.getcwd(), 'input')



# FILENAME VARIABLES
MODEL_FILE = 'model.pkl'                        # trained model pickle file name
PREPROCESSOR_SHELVE = 'preprocessors.shelve'    # Preprocessor shelve file name
INPUT_CSV = 'input.csv'                         # input csv
ALGO_REPORT_FILE = 'algorithm_performance_report.xlsx' # algorithm performance sheet



# REGEX PATTERNS FOR DATA CLEANING
HM_PAT = re.compile(r'^\d\d\:\d\d$')            # HH:MM Format
HMS_PAT = re.compile(r'^\d\d\:\d\d\:\d\d$')     # HH:MM:SS Format
DOT_PAT = re.compile(r'\d+\.\d+')               # Faulty pattern observed in Data



# COLUMN LIST VARIABLES
ALL_ORIGINAL_COLUMNS = [ 'Delivery_person_ID', 'Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Order_Date', 'Time_Orderd', 'Time_Order_picked', 'Weather_conditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City', 'Time_taken (min)']
TARGET_COLUMN = 'Time_taken (min)'
ALL_INPUT_COLUMNS = [ 'Delivery_person_ID', 'Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Order_Date', 'Time_Orderd', 'Time_Order_picked', 'Weather_conditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City' ]
# POST PREPROCESSING COLUMNS
CATEGORIC_COLUMNS = ['Delivery_person_ID', 'Road_traffic_density', 'Type_of_vehicle', 'City', 'Weather_conditions', 'Type_of_order', 'Festival']
NUMERIC_COLUMNS = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Time_Orderd', 'Time_Order_picked', 'Vehicle_condition', 'multiple_deliveries', 'Order_Day', 'Order_Month', 'Geo_Distance']
ALL_COLUMNS = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Time_Orderd', 'Time_Order_picked', 'Vehicle_condition', 'multiple_deliveries', 'Order_Day', 'Order_Month', 'Geo_Distance', 'Delivery_person_ID', 'Road_traffic_density', 'Type_of_vehicle', 'City', 'Weather_conditions', 'Type_of_order', 'Festival']



# PREPROCESSOR IDENTIFIERS
PRPRSSR_IDENTIFIER_IMPUTER  = 'imputer'
PRPRSSR_IDENTIFIER_SCALER   = 'scaler'
PRPRSSR_IDENTIFIER_DICT_ENCODER={'Delivery_person_ID': 'encoder_Delivery_person_ID',
                                'Road_traffic_density': 'encoder_Road_traffic_density',
                                'Type_of_vehicle': 'encoder_Type_of_vehicle',
                                'City': 'encoder_City',
                                'Weather_conditions': 'encoder_Weather_conditions',
                                'Type_of_order': 'encoder_Type_of_order',
                                'Festival': 'encoder_Festival'}
