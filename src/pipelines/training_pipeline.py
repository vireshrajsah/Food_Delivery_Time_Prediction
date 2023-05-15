import os
import sys
import pandas as pd
import src.statics as statics
from src.components.data_ingestion import *
from src.components.data_transformation import *
from src.components.model_trainer import *
from src.logging import logging
from src.exceptions import CustomException


if __name__ == "__main__":
    try:
        logging.info("TRAINING PIPELINE INITIATED")
        print("TRAINING PIPELINE INITIATED")

        # Data Ingestion
        ingestor = DataIngestor()
        trainpath, testpath = ingestor.ingest()
        print(f"Train path:{trainpath}\nTest path:{testpath}")

        logging.info("DATA INGESTION COMPLETE")

        # Data Preprocessing
        transformer = Preprocessor(trainpath, testpath)
        X_train, X_test, y_train, y_test = transformer.transform()
        print("Train Data:\n",X_train.head(),"\n\n\nTest Data:\n",X_test.head())

        logging.info("DATA PREPROCESSING COMPLETE")

        # Model Training
        model_trainer = Trainer(X_train, X_test, y_train, y_test)
        model_trainer.train()
        print(f"Model trained and saved in {statics.ARTIFACTS_DIR} with name {statics.MODEL_FILE}")

        logging.info("MODEL TRAINING COMPLETE")

        print("PIPELINE TERMINATED SUCCESSFULLY")
        logging.info("TRAINING PIPELINE SUCCESSFULLY TERMINATED")

    except Exception as e:
        print("PIPELINE TERMINATED DUE TO ERROR")
        logging.critical(CustomException(e,sys))
        raise CustomException(e,sys) from e