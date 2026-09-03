import json
import requests
import numpy as np
from crewai.tools import tool

# --------------------------------------------------------------------------
# AGENT 1 TOOLS: WEATHER, PHYSICS, & HISTORICAL SCADA
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
    """Retrieves empirical median power (kW) from SCADA history under similar irradiance (±100 W/m²)."""
    simulated_historical_kw = [2.3, 2.4, 2.5, 2.2, 2.45, 2.38]
    return json.dumps({
        "empirical_median_kw": float(np.median(simulated_historical_kw)),
        "percentile_10th_kw": float(np.percentile(simulated_historical_kw, 10)),
        "sample_size": len(simulated_historical_kw)
    })


# --------------------------------------------------------------------------
# AGENT 2 TOOLS: IEC DECISION TREES & INVENTORY
# --------------------------------------------------------------------------

@tool("Get Deterministic Repair Procedure")
def get_repair_procedure(fault_type: str) -> str:
    """Retrieves standard operating procedures (SOP), required tools, torque specs, and safety precautions."""
    trees = {
        "Shading": {
            "sop": "Inspect array perimeter for foliage growth or physical obstructions.",
            "safety": "Standard PPE.",
            "action": "Clear physical obstructions or re-string modules if shading is permanent."
        },
        "Short": {
            "sop": "Isolate string at DC combiner box. Check module diode resistance with DMM (>10kΩ expected).",
            "safety": "HIGH VOLTAGE RISK. Arc-flash face shield and insulated tools required.",
            "action": "Replace failed bypass diode (Torque: 2.1 Nm) or swap defective module."
        },
        "Connector": {
            "sop": "Inspect MC4 connectors using thermal camera for high resistance joints.",
            "safety": "De-energize string before uncoupling.",
            "action": "Re-crimp MC4 connector with calibrated crimping tool."
        },
        "OC": {
            "sop": "Open circuit fault. Measure VOC along string to isolate broken conductor/fuse.",
            "safety": "Check for floating ground voltage.",
            "action": "Replace blown inline DC fuse or repair broken interconnect wire."
        }
    }
    return json.dumps(trees.get(fault_type, {"sop": "General physical inspection required per IEC 62446."}))


@tool("Check Warehouse Parts Inventory")
def check_parts_inventory(fault_type: str) -> str:
    """Checks stock levels for required replacement parts based on fault type."""
    inventory_db = {
        "Short": {"part": "Bypass Diode Pack", "in_stock": True, "qty": 12, "lead_time_days": 0},
        "Connector": {"part": "MC4 Connector Pairs", "in_stock": True, "qty": 45, "lead_time_days": 0},
        "OC": {"part": "15A DC Fuse", "in_stock": False, "qty": 0, "lead_time_days": 3},
        "Shading": {"part": "N/A (Perimeter clearing)", "in_stock": True, "qty": 1, "lead_time_days": 0}
    }
    return json.dumps(inventory_db.get(fault_type, {"part": "Generic Hardware", "in_stock": True, "qty": 5, "lead_time_days": 0}))


# --------------------------------------------------------------------------
# AGENT 3 TOOLS: FINANCIAL DISPATCH ENGINE
# --------------------------------------------------------------------------

@tool("Calculate Repair ROI and Emergency Dispatch")
def calculate_repair_roi(power_loss_kw: float = 2.5, electricity_rate: float = 0.18) -> str:
    """Calculates daily revenue loss ($/day) and compares emergency dispatch vs scheduled maintenance."""
    daily_kwh_lost = power_loss_kw * 5.5  # Average 5.5 peak sun hours/day
    daily_revenue_loss = daily_kwh_lost * electricity_rate
    
    emergency_dispatch_cost = 450.0
    scheduled_dispatch_cost = 120.0
    
    breakeven_days = (emergency_dispatch_cost - scheduled_dispatch_cost) / max(daily_revenue_loss, 0.01)
    
    return json.dumps({
        "daily_revenue_loss_usd": round(daily_revenue_loss, 2),
        "emergency_cost_usd": emergency_dispatch_cost,
        "scheduled_cost_usd": scheduled_dispatch_cost,
        "breakeven_days": round(breakeven_days, 1),
        "recommendation": "EMERGENCY_DISPATCH" if breakeven_days <= 3 else "SCHEDULED_MAINTENANCE"
    })