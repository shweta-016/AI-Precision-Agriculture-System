"""
Data Processing Module
AI Precision Agriculture System

Handles:
- Dataset loading
- Cleaning
- Normalization
- Feature extraction
- Visualization support
"""

import pandas as pd
import numpy as np


class DataProcessor:

    def __init__(self):

        print("Data Processor Initialized")

    # ------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------

    def load_dataset(self, file_path):

        """
        Load dataset from CSV
        """

        data = pd.read_csv(file_path)

        print("Dataset Loaded")

        print("Rows:", data.shape[0])
        print("Columns:", data.shape[1])

        return data

    # ------------------------------------------------
    # DISPLAY BASIC INFO
    # ------------------------------------------------

    def dataset_info(self, data):

        print("\nDataset Information")

        print(data.info())

    # ------------------------------------------------
    # DISPLAY STATISTICS
    # ------------------------------------------------

    def dataset_statistics(self, data):

        print("\nStatistical Summary")

        print(data.describe())

    # ------------------------------------------------
    # CHECK MISSING VALUES
    # ------------------------------------------------

    def check_missing(self, data):

        print("\nMissing Values")

        print(data.isnull().sum())

    # ------------------------------------------------
    # REMOVE MISSING VALUES
    # ------------------------------------------------

    def remove_missing(self, data):

        print("\nRemoving Missing Values")

        data = data.dropna()

        return data

    # ------------------------------------------------
    # REMOVE DUPLICATES
    # ------------------------------------------------

    def remove_duplicates(self, data):

        before = len(data)

        data = data.drop_duplicates()

        after = len(data)

        print("Duplicates Removed:", before - after)

        return data

    # ------------------------------------------------
    # NORMALIZE DATA
    # ------------------------------------------------

    def normalize_data(self, data):

        numeric = data.select_dtypes(include=np.number)

        data[numeric.columns] = (

            numeric - numeric.min()

        ) / (

            numeric.max() - numeric.min()

        )

        return data

    # ------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------

    def select_features(self, data, features):

        selected = data[features]

        return selected

    # ------------------------------------------------
    # SPLIT DATASET
    # ------------------------------------------------

    def split_features_target(self, data, target):

        X = data.drop(target, axis=1)

        y = data[target]

        return X, y

    # ------------------------------------------------
    # DATA PIPELINE
    # ------------------------------------------------

    def process_pipeline(self, file_path):

        data = self.load_dataset(file_path)

        self.dataset_info(data)

        self.dataset_statistics(data)

        self.check_missing(data)

        data = self.remove_missing(data)

        data = self.remove_duplicates(data)

        data = self.normalize_data(data)

        return data


# ------------------------------------------------
# TEST FUNCTION
# ------------------------------------------------

def test_processor():

    processor = DataProcessor()

    file_path = input("Enter dataset path: ")

    data = processor.process_pipeline(file_path)

    print("\nProcessed Data Preview")

    print(data.head())


if __name__ == "__main__":

    test_processor()