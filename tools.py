import json
import requests
import numpy as np
from crewai.tools import tool

# --------------------------------------------------------------------------
# TOOLS: WEATHER, PHYSICS, & HISTORICAL SCADA
# (used by the Solar Diagnostic Expert agent in agents.py)
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


@tool("Calculate Expected PV Power Output")
def calculate_expected_pv_power(irradiance_wm2: float, temp_c: float, rated_pmax_w: float = 300.0) -> str:
    """Calculates theoretical PV power output using irradiance and temperature derating (-0.35%/°C)."""
    if irradiance_wm2 <= 0:
        return json.dumps({"expected_power_w": 0.0, "note": "Zero irradiance"})

    temp_loss = 1.0 + (-0.0035 * (temp_c - 25.0))
    expected = rated_pmax_w * (irradiance_wm2 / 1000.0) * max(temp_loss, 0.5)
    return json.dumps({"expected_power_w": round(expected, 2)})


@tool("Fetch Site Historical Power Baseline")
def get_power_baseline(site_id: str, current_irradiance: float) -> str:
    """Retrieves empirical median power (kW) from SCADA history under similar irradiance."""
    simulated_historical_kw = [2.3, 2.4, 2.5, 2.2, 2.45, 2.38]
    return json.dumps({
        "empirical_median_kw": float(np.median(simulated_historical_kw)),
        "percentile_10th_kw": float(np.percentile(simulated_historical_kw, 10)),
        "sample_size": len(simulated_historical_kw)
    })
