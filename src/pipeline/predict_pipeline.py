import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', "model.pkl")
        self.encoders_path = os.path.join('artifacts', "encoders.pkl")

        # Load the model and encoders
        self.model = load_object(self.model_path)
        self.encoders = load_object(self.encoders_path)
        logging.info("Model and encoder loaded")

    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data])

            # Apply binary encoding
            df['gender'] = df['gender'].map({'female': 0, 'male': 1})
            df['lunch'] = df['lunch'].map({'standard': 0, 'free/reduced': 1})
            df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})

            # Apply ordinal encoding
            df['parental level of education'] = self.encoders['ordinal_encoder'].transform(df[['parental level of education']])

            # Apply one-hot encoding
            race_ethnicity_encoded = self.encoders['one_hot_encoder'].transform(df[['race/ethnicity']])
            race_ethnicity_encoded_df = pd.DataFrame(race_ethnicity_encoded, columns=self.encoders['one_hot_encoder'].get_feature_names_out(['race/ethnicity']))
            df = pd.concat([df, race_ethnicity_encoded_df], axis=1).drop(columns=['race/ethnicity'])

            logging.info("Binary, ordinal and one-hot encoding applied successfully.")

            # Make prediction
            prediction = self.model.predict(df)
            logging.info(f"math score prediction: {prediction}")
            return prediction[0]
        

        except Exception as e:
            raise CustomException("Error occurred during prediction.", sys)

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        return pd.DataFrame([{
            'gender': self.gender,
            'race/ethnicity': self.race_ethnicity,
            'parental level of education': self.parental_level_of_education,
            'lunch': self.lunch,
            'test preparation course': self.test_preparation_course,
            'reading score': self.reading_score,
            'writing score': self.writing_score
        }])