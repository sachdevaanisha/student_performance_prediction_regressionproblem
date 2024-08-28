# Student Exam Performance Predictor

The **Student Exam Performance Predictor** is a web-based application designed to predict a student's math score based on various input parameters. This project demonstrates a modular approach to handling a regression problem, showcasing a well-structured pipeline for data processing, model training, and prediction.

## Key Components

1. **Data Ingestion (`data_ingestion.py`)** 
   - Handles the loading and initial preprocessing of the dataset.

2. **Data Transformation (`data_transformation.py`)**
   - Performs detailed data processing tasks including:
     - Outlier detection and removal.
     - Encoding categorical features using binary, one-hot, and ordinal encoding techniques.
     - Splitting the data into training and testing sets for model evaluation.

3. **Model Training (`model_training.py`)**
   - Involves training various regression models on the processed data.
   - Evaluates model performance using metrics such as:
     - R² score
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)

4. **Prediction Pipeline (`predict_pipeline.py`)**
   - Defines a `PredictPipeline` class for making predictions with pre-trained models.

5. **Web Application (`app.py` and `home.html`)**
   - A Flask-based web application that provides an intuitive interface for users to input their data and receive predictions. Port 'http://127.0.0.1:5000' is used to run the application.
