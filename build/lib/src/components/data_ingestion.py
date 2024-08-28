import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData

import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the initiate_data_ingestion method")
        try:
            df = pd.read_csv('jupyter_notebook/StudentsPerformance.csv')
            logging.info("Read dataset as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Read and saved data in the raw_data folder.")

            return(
                self.ingestion_config.raw_data_path
                )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    raw_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(raw_data)

    model_trainer = ModelTrainer()
    best_model = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

    print("Best Model:", best_model)

    pipeline = PredictPipeline()
    data = CustomData(gender='female', race_ethnicity='group B', parental_level_of_education='high school', lunch='standard', test_preparation_course='none', reading_score = 89, writing_score = 90)
    input_data = data.get_data_as_dataframe()
    prediction = pipeline.predict(input_data.iloc[0].to_dict())
    print(f"Predicted Math Score: {prediction}")

