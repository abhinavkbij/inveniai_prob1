# 'http://0.0.0.0:8000/weather'
# Method: GET
# Parameters: 
# 1. q - the city name (Here, 'London')
# 2. appid - your API key

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Pydantic schema for response: 
class Weather(BaseModel):
    id: int
    # main: str
    description: str
    # icon: str

 

# class Main(BaseModel):
#     temp: float
#     feels_like: float
#     temp_min: float
#     temp_max: float
#     pressure: int
#     humidity: int

 

class Coordinates(BaseModel):
    lon: float
    lat: float

 

class WeatherData(BaseModel): # [The JSON response object to return]
    coord: Coordinates
    weather: List[Weather]
    # main: Main


@app.get("/weather/", response_model=WeatherData)
async def weatherService(city: str, api_key: str):
    weather = Weather(id=1, description=f"{city} is a great city")

    coords = Coordinates(lon=61.33,lat=62.44)

    wdata = WeatherData(weather=[weather], coord=coords)

    return wdata

uvicorn.run(app, host="0.0.0.0", port=8000)