import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import pickle

class DataTransformationConfig:
    def __init__(self):
        self.processed_data_path = os.path.join('artifacts', "processed_data.csv")
        self.encoders_path = os.path.join('artifacts', "encoders.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info("Entered the initiate_data_transformation method")
            
            df = pd.read_csv(raw_data_path)
            logging.info("Read raw data as dataframe")

            # Outlier Treatment
            def remove_outliers(df, column):
                q1 = np.percentile(df[column], 25)
                q3 = np.percentile(df[column], 75)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                higher_fence = q3 + 1.5 * iqr
                return df[(df[column] > lower_fence) & (df[column] < higher_fence)]

            df = remove_outliers(df, 'math score')
            df = remove_outliers(df, 'reading score')
            df = remove_outliers(df, 'writing score')

            # Split Data
            X = df.drop(columns=['math score'])
            y = df['math score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature Transformation
            # Binary Encoding
            def binary_encoding(df):
                df['gender'] = df['gender'].map({'female': 0, 'male': 1})
                df['lunch'] = df['lunch'].map({'standard': 0, 'free/reduced': 1})
                df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})
                return df

            X_train = binary_encoding(X_train)
            X_test = binary_encoding(X_test)

            # Ordinal Encoding
            education_order = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
            ordinal_encoder = OrdinalEncoder(categories=[education_order])

            # Fit and transform on training data
            X_train['parental level of education'] = ordinal_encoder.fit_transform(X_train[['parental level of education']])
            # Transform test data
            X_test['parental level of education'] = ordinal_encoder.transform(X_test[['parental level of education']])

            # One-Hot Encoding
            one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)

            # Fit and transform on training data
            X_train_encoded = one_hot_encoder.fit_transform(X_train[['race/ethnicity']])
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=one_hot_encoder.get_feature_names_out(['race/ethnicity']))
            X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded_df], axis=1).drop(columns=['race/ethnicity'])

            # Transform test data
            X_test_encoded = one_hot_encoder.transform(X_test[['race/ethnicity']])
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=one_hot_encoder.get_feature_names_out(['race/ethnicity']))
            X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded_df], axis=1).drop(columns=['race/ethnicity'])

            # Save the encoders and expected columns
            encoders = {
                'ordinal_encoder': ordinal_encoder,
                'one_hot_encoder': one_hot_encoder,
                'expected_columns': X_train.columns.tolist()  # Save the list of columns after transformation
            }
            os.makedirs(os.path.dirname(self.transformation_config.encoders_path), exist_ok=True)
            with open(self.transformation_config.encoders_path, 'wb') as f:
                pickle.dump(encoders, f)

            # Save Processed Data
            df_processed = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
            os.makedirs(os.path.dirname(self.transformation_config.processed_data_path), exist_ok=True)
            df_processed.to_csv(self.transformation_config.processed_data_path, index=False, header=True)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
