from smolagents import tool
import os
import requests
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """ Get REAL-TIME current weather conditions for a city using OpenWeather API.

    ALWAYS use this tool when the user asks about:
    - Current weather
    - Current temperature
    - Humidity
    - Wind speed
    - Weather conditions

    Do not use web search for current weather
    
    Args:
        city: The name of the city (e.g Toronto, London)

    Returns: Current weather within that city from OpenWeather.

    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "OpenWeather API key is missing."
    
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Weather in {city}:\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Conditions: {description}\n"
            f"Humidity: {humidity}%\n"
            f"Wind speed: {wind_speed} m/s"
        )

    except requests.exceptions.HTTPError as e:
        return f"OpenWeather API error: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.RequestException as e:
        return f"Weather API request failed: {e}"
