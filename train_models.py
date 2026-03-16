"""
Train Models Module
AI Precision Agriculture System

Used to train all ML models
"""

from crop_recommendation import CropRecommendationSystem
from irrigation_prediction import IrrigationPredictionSystem


class ModelTrainer:

    def __init__(self):

        print("Model Trainer Initialized")

    # ---------------------------------------
    # TRAIN CROP MODEL
    # ---------------------------------------

    def train_crop_model(self):

        crop_system = CropRecommendationSystem()

        print("\nTraining Crop Recommendation Model")

        crop_system.train_model()

        print("Crop Model Training Completed")

    # ---------------------------------------
    # TRAIN IRRIGATION MODEL
    # ---------------------------------------

    def train_irrigation_model(self):

        irrigation_system = IrrigationPredictionSystem()

        print("\nTraining Irrigation Prediction Model")

        irrigation_system.train_model()

        print("Irrigation Model Training Completed")

    # ---------------------------------------
    # TRAIN ALL MODELS
    # ---------------------------------------

    def train_all_models(self):

        print("\nStarting Training Pipeline")

        self.train_crop_model()

        self.train_irrigation_model()

        print("\nAll Models Trained Successfully")

    # ---------------------------------------
    # VERIFY MODELS
    # ---------------------------------------

    def verify_models(self):

        print("\nVerifying Trained Models")

        try:

            crop_system = CropRecommendationSystem()
            crop_system.load_model()

            print("Crop Model Loaded Successfully")

        except:

            print("Crop Model Missing")

        try:

            irrigation_system = IrrigationPredictionSystem()
            irrigation_system.load_model()

            print("Irrigation Model Loaded Successfully")

        except:

            print("Irrigation Model Missing")


# ---------------------------------------
# MAIN FUNCTION
# ---------------------------------------

def main():

    trainer = ModelTrainer()

    print("\n1 Train Models")
    print("2 Verify Models")

    choice = input("Enter choice: ")

    if choice == "1":

        trainer.train_all_models()

    elif choice == "2":

        trainer.verify_models()

    else:

        print("Invalid choice")


if __name__ == "__main__":

    main()