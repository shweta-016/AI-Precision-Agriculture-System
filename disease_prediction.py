# """
# Plant Disease Detection Module
# AI Precision Agriculture System

# This module:
# 1. Loads trained CNN model
# 2. Processes plant leaf images
# 3. Predicts plant disease
# 4. Returns disease name and confidence
# """

# import os
# import numpy as np
# import tensorflow as tf
# import cv2

# from image_processing import ImageProcessor


# # -------------------------------------------------
# # MODEL CONFIGURATION
# # -------------------------------------------------

# MODEL_PATH = "plant_disease_model.h5"

# IMAGE_SIZE = 224


# # -------------------------------------------------
# # DISEASE CLASSES
# # -------------------------------------------------

# DISEASE_CLASSES = [

#     "Apple - Apple Scab",
#     "Apple - Black Rot",
#     "Apple - Cedar Apple Rust",
#     "Apple - Healthy",

#     "Blueberry - Healthy",

#     "Cherry - Powdery Mildew",
#     "Cherry - Healthy",

#     "Corn - Cercospora Leaf Spot",
#     "Corn - Common Rust",
#     "Corn - Northern Leaf Blight",
#     "Corn - Healthy",

#     "Grape - Black Rot",
#     "Grape - Esca",
#     "Grape - Leaf Blight",
#     "Grape - Healthy",

#     "Orange - Citrus Greening",

#     "Peach - Bacterial Spot",
#     "Peach - Healthy",

#     "Pepper - Bacterial Spot",
#     "Pepper - Healthy",

#     "Potato - Early Blight",
#     "Potato - Late Blight",
#     "Potato - Healthy",

#     "Strawberry - Leaf Scorch",
#     "Strawberry - Healthy",

#     "Tomato - Bacterial Spot",
#     "Tomato - Early Blight",
#     "Tomato - Late Blight",
#     "Tomato - Leaf Mold",
#     "Tomato - Septoria Leaf Spot",
#     "Tomato - Spider Mites",
#     "Tomato - Target Spot",
#     "Tomato - Yellow Leaf Curl Virus",
#     "Tomato - Mosaic Virus",
#     "Tomato - Healthy"

# ]


# # -------------------------------------------------
# # MODEL LOADING FUNCTION
# # -------------------------------------------------

# def load_disease_model():

#     """
#     Loads the trained CNN disease model
#     """

#     if not os.path.exists(MODEL_PATH):

#         raise Exception(
#             "Disease model not found. Train model first."
#         )

#     model = tf.keras.models.load_model(MODEL_PATH)

#     return model


# # -------------------------------------------------
# # IMAGE PREPROCESSING
# # -------------------------------------------------

# def preprocess_image(image_path):

#     """
#     Converts image into CNN input format
#     """

#     processor = ImageProcessor()

#     image = processor.preprocess_image(image_path)

#     return image


# # -------------------------------------------------
# # PREDICT DISEASE
# # -------------------------------------------------

# def predict_disease(image_path):
    

#     # Load model
#     model = load_disease_model()

#     # Preprocess image
#     image = preprocess_image(image_path)

#     # Predict
#     prediction = model.predict(image)

#     # Get class index
#     class_index = np.argmax(prediction)

#     confidence = float(np.max(prediction))

#     # Get disease name
#     disease_name = DISEASE_CLASSES[class_index]

#     confidence = round(confidence * 100, 2)

#     return disease_name, confidence


# # -------------------------------------------------
# # IMAGE VALIDATION
# # -------------------------------------------------

# def validate_image(image_path):

#     """
#     Ensures the image is valid
#     """

#     if not os.path.exists(image_path):

#         raise FileNotFoundError(
#             "Image file not found."
#         )

#     image = cv2.imread(image_path)

#     if image is None:

#         raise Exception(
#             "Invalid image format."
#         )

#     return True


# # -------------------------------------------------
# # TEST FUNCTION
# # -------------------------------------------------

# def test_prediction():

#     """
#     Used for testing the model
#     """

#     image_path = input(
#         "Enter leaf image path: "
#     )

#     try:

#         validate_image(image_path)

#         disease, confidence = predict_disease(
#             image_path
#         )

#         print("\nPrediction Result")
#         print("----------------------")
#         print("Disease:", disease)
#         print("Confidence:", confidence, "%")

#     except Exception as e:

#         print("Error:", e)


# # -------------------------------------------------
# # MAIN FUNCTION
# # -------------------------------------------------

# def main():

#     """
#     CLI interface for disease detection
#     """

#     print("\nPlant Disease Detection System")

#     print("1 Predict Disease")
#     print("2 Exit")

#     choice = input("Enter choice: ")

#     if choice == "1":

#         test_prediction()

#     else:

#         print("Exiting...")


# # -------------------------------------------------
# # RUN MODULE
# # -------------------------------------------------

# if __name__ == "__main__":

#     main()
import tensorflow as tf
import numpy as np
import cv2


MODEL_PATH = "plant_disease_model.h5"

IMAGE_SIZE = 224


DISEASE_CLASSES = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry___Powdery_mildew",
"Cherry___healthy",
"Corn___Cercospora_leaf_spot",
"Corn___Common_rust",
"Corn___Northern_Leaf_Blight",
"Corn___healthy",
"Grape___Black_rot",
"Grape___Esca",
"Grape___Leaf_blight",
"Grape___healthy",
"Orange___Haunglongbing",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper___Bacterial_spot",
"Pepper___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites",
"Tomato___Target_Spot",
"Tomato___Yellow_Leaf_Curl_Virus",
"Tomato___Mosaic_virus",
"Tomato___healthy"
]


model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):

    image = cv2.resize(image,(224,224))

    image = image / 255.0

    image = np.expand_dims(image,axis=0)

    return image


def predict_disease(image):

    image = preprocess_image(image)

    prediction = model.predict(image)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    disease = DISEASE_CLASSES[index]

    return disease, confidence