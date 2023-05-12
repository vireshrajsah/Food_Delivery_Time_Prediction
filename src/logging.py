import os
import sys
import logging
from datetime import datetime

LOG_FILE_NAME = f"log - {datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), 'logs')

os.makedirs(LOG_FILE_PATH, exist_ok=True)

logging.basicConfig(filename=os.path.join(LOG_FILE_PATH, LOG_FILE_NAME), 
                    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s: %(message)s",
                    level= logging.INFO)