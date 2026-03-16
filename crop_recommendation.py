"""
AI Precision Agriculture System
Crop Recommendation Module

Predicts best crop using:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CropRecommendationSystem:

    """
    Machine Learning system for crop prediction
    """

    def __init__(self):

        self.model = None
        self.scaler = StandardScaler()

        self.model_file = "crop_model.pkl"
        self.scaler_file = "scaler.pkl"

    # ------------------------------
    # Load Dataset
    # ------------------------------

    def load_dataset(self):

        print("Loading crop dataset...")

        data = pd.read_csv(
            "C:/Users/Admin/OneDrive/Desktop/agriculture/Datasets/Crop_recommendation.csv"
        )

        print("Dataset Shape:", data.shape)

        return data

    # ------------------------------
    # Data Preprocessing
    # ------------------------------

    def preprocess_data(self, data):

        X = data.drop("label", axis=1)
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Fit scaler ONLY on training data
        X_train = self.scaler.fit_transform(X_train)

        # Transform test data
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    # ------------------------------
    # Train Model
    # ------------------------------

    def train_model(self):

        data = self.load_dataset()

        X_train, X_test, y_train, y_test = self.preprocess_data(data)

        print("Training Random Forest Model...")

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        print("Model Accuracy:", round(acc * 100, 2), "%")

        # Save model and scaler
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)

        print("Model saved as:", self.model_file)
        print("Scaler saved as:", self.scaler_file)

    # ------------------------------
    # Load Model
    # ------------------------------

    def load_model(self):

        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):

            print("Loading trained model...")

            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)

        else:

            print("Model not found. Training new model...")
            self.train_model()

    # ------------------------------
    # Crop Prediction
    # ------------------------------

    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):

        if self.model is None:
            self.load_model()

        features = np.array([[

            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall

        ]])

        # Use transform ONLY
        features = self.scaler.transform(features)

        prediction = self.model.predict(features)

        return prediction[0]


# ------------------------------
# Test Module
# ------------------------------

if __name__ == "__main__":

    system = CropRecommendationSystem()

    system.train_model()

    crop = system.recommend_crop(

        N=90,
        P=40,
        K=40,
        temperature=26,
        humidity=80,
        ph=6.5,
        rainfall=200

    )

    print("\nRecommended Crop:", crop)