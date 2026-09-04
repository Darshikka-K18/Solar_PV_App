import json
import requests
from crewai.tools import tool

# --------------------------------------------------------------------------
# TOOL: RESOLVE A DISTRICT/LOCATION NAME TO COORDINATES
# (the agent calls this itself -- app.py never computes lat/lon)
# --------------------------------------------------------------------------

@tool("Get Coordinates for Location")
def get_coordinates_for_location(location_name: str) -> str:
    """Looks up latitude/longitude for a Sri Lankan district or city name
    using the Open-Meteo Geocoding API, with a safe Kilinochchi fallback if
    the lookup fails."""
    try:
        url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={location_name}&count=1&language=en&format=json"
        )
        res = requests.get(url, timeout=4).json()
        if "results" in res and len(res["results"]) > 0:
            top = res["results"][0]
            return json.dumps({
                "latitude": top["latitude"],
                "longitude": top["longitude"],
                "resolved_name": top.get("name", location_name),
                "fallback_used": False
            })
    except Exception:
        pass

    return json.dumps({
        "latitude": 9.3961,
        "longitude": 80.3982,
        "resolved_name": "Kilinochchi (Default)",
        "fallback_used": True
    })


# --------------------------------------------------------------------------
# TOOL: LIVE WEATHER
# --------------------------------------------------------------------------

@tool("Fetch Live Weather with Fallback")
def fetch_weather_data(latitude: float, longitude: float) -> str:
    """Fetches ambient temp (°C), cloud cover (%), and irradiance (W/m²) with retry & historical fallback."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&"
        f"current=temperature_2m,cloud_cover,direct_normal_irradiance,diffuse_radiation"
    )
    for attempt in range(3):
        try:
            res = requests.get(url, timeout=4).json()
            curr = res.get("current", {})
            direct = curr.get("direct_normal_irradiance", 0.0) or 0.0
            diffuse = curr.get("diffuse_radiation", 0.0) or 0.0
            return json.dumps({
                "temperature_c": curr.get("temperature_2m", 25.0),
                "cloud_cover_pct": curr.get("cloud_cover", 0),
                "total_irradiance_wm2": round(direct + diffuse, 2),
                "fallback_used": False
            })
        except Exception:
            continue

    return json.dumps({
        "temperature_c": 25.0,
        "cloud_cover_pct": 45.0,
        "total_irradiance_wm2": 600.0,
        "fallback_used": True
    })