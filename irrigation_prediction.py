"""
AI Precision Agriculture System
Irrigation Prediction Module

Author: Final Year Data Science Project

This module predicts whether irrigation is required
based on environmental conditions.

Dataset Used:
DailyDelhiClimateTrain.csv
DailyDelhiClimateTest.csv

Features Used:
meantemp
humidity
wind_speed
meanpressure

Target:
irrigation

0 -> No irrigation required
1 -> Irrigation required
"""

# -------------------------------------------------
# Import Required Libraries
# -------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# Irrigation Prediction System Class
# -------------------------------------------------

class IrrigationPredictionSystem:

    """
    This class handles the complete irrigation prediction pipeline.

    Functions included:

    1. Load dataset
    2. Explore dataset
    3. Preprocess data
    4. Train machine learning model
    5. Evaluate model performance
    6. Save trained model
    7. Load trained model
    8. Predict irrigation requirement
    9. Batch prediction
    """
     # Constructor
    def __init__(self):

        """
        Initialize system variables
        """

        self.model = None
        self.scaler = StandardScaler()

        # Files for saving model and scaler
        self.model_file = "irrigation_model.pkl"
        self.scaler_file = "irrigation_scaler.pkl"

        print("\nIrrigation Prediction System Initialized")

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    def load_dataset(self):

        """
        Load training and testing dataset
        """

        print("\nLoading datasets...")

        train_path = "C:/Users/Admin/OneDrive/Desktop/agriculture/Datasets/DailyDelhiClimateTrain.csv"
        test_path = "C:/Users/Admin/OneDrive/Desktop/agriculture/Datasets/DailyDelhiClimateTest.csv"

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        print("Training dataset shape:", train_data.shape)
        print("Testing dataset shape:", test_data.shape)

        return train_data, test_data

    # -------------------------------------------------
    # Explore Dataset
    # -------------------------------------------------

    def explore_data(self, data):

        """
        Perform basic dataset exploration
        """

        print("\nDataset Information")
        print(data.info())

        print("\nFirst 5 rows")
        print(data.head())

        print("\nMissing values")
        print(data.isnull().sum())

        print("\nStatistical Summary")
        print(data.describe())

    # -------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------

    def preprocess_data(self, train_data, test_data):

        """
        Prepare dataset for machine learning
        """

        print("\nPreprocessing dataset...")

        # Create target variable
        train_data["irrigation"] = (train_data["humidity"] < 60).astype(int)
        test_data["irrigation"] = (test_data["humidity"] < 60).astype(int)

        # Feature columns
        features = [
            "meantemp",
            "humidity",
            "wind_speed",
            "meanpressure"
        ]

        target = "irrigation"

        # Split features and labels
        X_train = train_data[features]
        y_train = train_data[target]

        X_test = test_data[features]
        y_test = test_data[target]

        # Feature Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Data preprocessing completed")

        return X_train_scaled, X_test_scaled, y_train, y_test

    # -------------------------------------------------
    # Train Machine Learning Model
    # -------------------------------------------------

    def train_model(self):

        """
        Train Random Forest model
        """

        print("\nTraining irrigation prediction model...")

        train_data, test_data = self.load_dataset()

        # Dataset exploration
        self.explore_data(train_data)

        # Preprocess dataset
        X_train, X_test, y_train, y_test = self.preprocess_data(
            train_data,
            test_data
        )

        # Initialize Random Forest
        self.model = RandomForestClassifier(

            n_estimators=200,
            max_depth=10,
            random_state=42

        )

        # Train model
        self.model.fit(X_train, y_train)

        print("\nModel training completed")

        # Evaluate model
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print("\nModel Accuracy:", accuracy)

        print("\nClassification Report")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, predictions))

        # Save model and scaler
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)

        print("\nModel saved as:", self.model_file)
        print("Scaler saved as:", self.scaler_file)

    # -------------------------------------------------
    # Load Saved Model
    # -------------------------------------------------

    def load_model(self):

        """
        Load saved model and scaler
        """

        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):

            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)

            print("\nModel and scaler loaded successfully")

        else:

            print("\nModel not found. Training new model...")
            self.train_model()

    # -------------------------------------------------
    # Irrigation Prediction
    # -------------------------------------------------

    def predict_irrigation(self, meantemp, humidity, wind_speed, meanpressure):

        """
        Predict irrigation requirement
        """

        if self.model is None:
            raise Exception("Model is not loaded")

        # Create feature array
        features = np.array([[
            meantemp,
            humidity,
            wind_speed,
            meanpressure
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)

        if prediction[0] == 1:
            return "Irrigation Required"

        else:
            return "No Irrigation Needed"

    # -------------------------------------------------
    # Batch Prediction
    # -------------------------------------------------

    def batch_prediction(self, file_path):

        """
        Predict irrigation for multiple rows
        """

        print("\nRunning batch prediction...")

        data = pd.read_csv(file_path)

        features = data[[
            "meantemp",
            "humidity",
            "wind_speed",
            "meanpressure"
        ]]

        features_scaled = self.scaler.transform(features)

        predictions = self.model.predict(features_scaled)

        data["irrigation_prediction"] = predictions

        print("\nBatch prediction completed")

        return data

    # -------------------------------------------------
    # Display Feature Importance
    # -------------------------------------------------

    def feature_importance(self):

        """
        Display feature importance
        """

        if self.model is None:
            print("Model not trained")
            return

        features = [
            "meantemp",
            "humidity",
            "wind_speed",
            "meanpressure"
        ]

        importance = self.model.feature_importances_

        print("\nFeature Importance")

        for feature, score in zip(features, importance):

            print(feature, ":", score)

    # -------------------------------------------------
    # System Summary
    # -------------------------------------------------

    def system_summary(self):

        """
        Display system information
        """

        print("\nAI Precision Agriculture System")
        print("--------------------------------")

        print("Machine Learning Model: Random Forest")
        print("Number of Trees: 200")
        print("Maximum Depth: 10")

        print("\nFeatures Used:")

        print("1. Mean Temperature")
        print("2. Humidity")
        print("3. Wind Speed")
        print("4. Mean Pressure")

        print("\nPrediction Target:")
        print("Irrigation Requirement")


# -------------------------------------------------
# Testing Module
# -------------------------------------------------

if __name__ == "__main__":

    system = IrrigationPredictionSystem()

    # Train model
    system.train_model()

    # Load trained model
    system.load_model()

    # Example prediction
    result = system.predict_irrigation(

        meantemp=32,
        humidity=55,
        wind_speed=12,
        meanpressure=1012

    )

    print("\nPrediction Result:", result)

    # Display feature importance
    system.feature_importance()

    # Show system summary
    system.system_summary()