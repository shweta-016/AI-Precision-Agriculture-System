"""
Weather API Module
AI Precision Agriculture System
"""

import requests


class WeatherAPI:

    def __init__(self):

        # Replace with your OpenWeather API key
        self.api_key = '53577eae2500efd2bf617e666d72239b'

    # ------------------------------------------------
    # FETCH WEATHER DATA
    # ------------------------------------------------

    def get_weather(self, city):

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"

        response = requests.get(url)

        data = response.json()

        return data

    # ------------------------------------------------
    # PARSE WEATHER DATA
    # ------------------------------------------------

    def parse_weather(self, data):

        if "main" not in data:

            print("Weather API Error:", data)

            return None, None, None, None

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        wind_speed = data["wind"]["speed"]

        return temperature, humidity, pressure, wind_speed

    # ------------------------------------------------
    # DISPLAY WEATHER
    # ------------------------------------------------

    def display_weather(self, city):

        data = self.get_weather(city)

        temp, humidity, pressure, wind = self.parse_weather(data)

        if temp is None:

            print("Unable to fetch weather data.")
            return

        print("\nWeather Report")
        print("-------------------------")
        print("City:", city)
        print("Temperature:", temp, "°C")
        print("Humidity:", humidity, "%")
        print("Pressure:", pressure, "hPa")
        print("Wind Speed:", wind, "m/s")

    # ------------------------------------------------
    # RETURN WEATHER SUMMARY (FOR STREAMLIT)
    # ------------------------------------------------

    def weather_summary(self, city):

        data = self.get_weather(city)

        temp, humidity, pressure, wind = self.parse_weather(data)

        summary = {

            "temperature": temp,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind

        }

        return summary


# ------------------------------------------------
# TEST WEATHER API
# ------------------------------------------------

def test_weather():

    api = WeatherAPI()

    city = input("Enter city name: ")

    api.display_weather(city)


if __name__ == "__main__":

    test_weather()


# import requests

# API_KEY = "53577eae2500efd2bf617e666d72239b"


# def get_weather_data(city):

#     url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

#     response = requests.get(url)

#     data = response.json()

#     weather = {
#         "temperature": data["main"]["temp"],
#         "humidity": data["main"]["humidity"],
#         "pressure": data["main"]["pressure"],
#         "wind_speed": data["wind"]["speed"]
#     }

#     return weather


# # optional test when running this file directly
# if __name__ == "__main__":

#     city = input("Enter city name: ")

#     weather = get_weather_data(city)

#     print("\nWeather Report")
#     print("----------------------")
#     print("City:", city)
#     print("Temperature:", weather["temperature"], "°C")
#     print("Humidity:", weather["humidity"], "%")
#     print("Pressure:", weather["pressure"], "hPa")
#     print("Wind Speed:", weather["wind_speed"], "m/s")