"""
AI Precision Agriculture System
Streamlit User Interface
"""

from turtle import color

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from weather_api import WeatherAPI

from crop_recommendation import CropRecommendationSystem
from irrigation_prediction import IrrigationPredictionSystem
from disease_prediction import predict_disease


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Precision Agriculture System",
    layout="wide"
)

import base64

# --------------------------------------------------
# CUSTOM STYLE (BACKGROUND + COLORS)
# --------------------------------------------------

def set_background():

    with open("background.jpg", "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Background image */
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark overlay for readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.35);
            z-index: 0;
        }}

        .block-container {{
            position: relative;
            z-index: 1;
        }}

        /* MAIN TITLE */
        h1 {{
            color: black;
            font-size: 50px;
            text-align: center;
            background: rgba(255,255,255,0.75);
            padding: 12px 25px;
            border-radius: 12px;
            display: inline-block;
            animation: fadeDown 1s ease-in-out;
        }}

        /* SECTION HEADINGS */
        h2, h3 {{
            color: black;
            font-weight: bold;
            background: rgba(255,255,255,0.75);
            padding: 8px 18px;
            border-radius: 10px;
            display: inline-block;
            animation: fadeUp 1s ease-in-out;
        }}

        /* NORMAL TEXT */
        p {{
            color: white;
            font-size: 18px;
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            background: rgba(0,0,0,0.7);
            color: white;
        }}

        /* BUTTON STYLE */
        .stButton>button {{
            background: linear-gradient(90deg,#2ecc71,#27ae60);
            color: white;
            border-radius: 12px;
            font-size: 18px;
            padding: 8px 20px;
            border: none;
            transition: 0.3s;
        }}

        .stButton>button:hover {{
            transform: scale(1.08);
            background: linear-gradient(90deg,#27ae60,#1e8449);
        }}

        /* Animations */
        @keyframes fadeDown {{
            from {{opacity:0; transform:translateY(-30px);}}
            to {{opacity:1; transform:translateY(0);}}
        }}

        @keyframes fadeUp {{
            from {{opacity:0; transform:translateY(30px);}}
            to {{opacity:1; transform:translateY(0);}}
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

st.title("🌱 AI Precision Agriculture System")

st.write("""
This intelligent system helps farmers make better decisions using AI.

Modules included:

• Crop Recommendation  
• Irrigation Prediction  
• Plant Disease Detection  
• Weather Forecast
""")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    crop_model = CropRecommendationSystem()
    crop_model.load_model()

    irrigation_model = IrrigationPredictionSystem()
    irrigation_model.load_model()

    return crop_model, irrigation_model


crop_model, irrigation_model = load_models()


# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------

menu = st.sidebar.selectbox(

    "Select Module",

    [
        "Home",
        "Crop Recommendation",
        "Irrigation Prediction",
        "Disease Detection",
        "Weather Forecast",
        "Dataset Analysis"
    ]

)

# --------------------------------------------------
# HOME
# --------------------------------------------------

if menu == "Home":

    st.header("Welcome to AI Precision Agriculture")

    

    st.write("""
This project uses Machine Learning and Deep Learning
to assist farmers in making data-driven agricultural decisions.
""")

    st.subheader("Project Modules")

    st.success("🌾 Crop Recommendation System")
    st.info("💧 Irrigation Prediction System")
    st.warning("🍃 Plant Disease Detection")
    st.error("☁ Weather Forecast")


# --------------------------------------------------
# CROP RECOMMENDATION
# --------------------------------------------------

elif menu == "Crop Recommendation":

    st.header("🌾 Crop Recommendation System")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen",0,200,90)
        P = st.number_input("Phosphorus",0,200,40)
        K = st.number_input("Potassium",0,200,40)

    with col2:
        temperature = st.number_input("Temperature",0.0,50.0,25.0)
        humidity = st.number_input("Humidity",0.0,100.0,80.0)

    with col3:
        ph = st.number_input("pH Value",0.0,14.0,6.5)
        rainfall = st.number_input("Rainfall",0.0,300.0,200.0)

    if st.button("Recommend Crop"):

        crop = crop_model.recommend_crop(
            N,P,K,
            temperature,
            humidity,
            ph,
            rainfall
        )

        st.success(f"🌾 Recommended Crop: {crop}")

        st.image(
            f"https://source.unsplash.com/600x400/?{crop},farm",
            caption=f"{crop} crop",
            use_container_width=True
        )


# --------------------------------------------------
# IRRIGATION
# --------------------------------------------------

elif menu == "Irrigation Prediction":

    st.header("💧 Irrigation Prediction")

    temperature = st.slider("Temperature",0,50,30)
    humidity = st.slider("Humidity",0,100,60)
    wind_speed = st.slider("Wind Speed",0,50,10)
    pressure = st.slider("Pressure",900,1100,1010)

    if st.button("Predict Irrigation"):

        result = irrigation_model.predict_irrigation(
            temperature,
            humidity,
            wind_speed,
            pressure
        )

        st.success(result)


# --------------------------------------------------
# DISEASE DETECTION
# --------------------------------------------------

elif menu == "Disease Detection":

    st.header("🍃 Plant Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes,1)

        st.image(image, caption="Uploaded Leaf")

        if st.button("Predict Disease"):

            disease, confidence = predict_disease(image)

            st.success(f"Disease: {disease}")

            st.write(
                f"Confidence: {round(confidence*100,2)}%"
            )


# --------------------------------------------------
# WEATHER FORECAST
# --------------------------------------------------

elif menu == "Weather Forecast":

    st.header("☁️ Weather Information")

    city = st.text_input("Enter City Name", "Pune")

    if st.button("Get Weather"):

        try:

            api = WeatherAPI()

            weather = api.weather_summary(city)

            st.subheader(f"Weather in {city}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Temperature", f"{weather['temperature']} °C")

            with col2:
                st.metric("Humidity", f"{weather['humidity']} %")

            with col3:
                st.metric("Wind Speed", f"{weather['wind_speed']} m/s")

            with col4:
                st.metric("Pressure",f"{weather['pressure']} hPa")

        except:

            st.error("Unable to fetch weather data")


# --------------------------------------------------
# DATASET ANALYSIS
# --------------------------------------------------

elif menu == "Dataset Analysis":

    st.header("📊 Dataset Visualization")

    dataset = pd.read_csv("Datasets/crop_recommendation.csv")

    st.write(dataset.head())

    st.subheader("Rainfall Distribution")
    st.bar_chart(dataset["rainfall"])

    st.subheader("Temperature Distribution")
    st.line_chart(dataset["temperature"])

    st.subheader("Humidity Distribution")
    st.area_chart(dataset["humidity"])


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.write("🌱 AI Precision Agriculture Project")