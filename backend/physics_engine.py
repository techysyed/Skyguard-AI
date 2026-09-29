import math
from dataclasses import dataclass
import numpy as np
import pandas as pd

EPSILON = 0.62198          
R_D = 287.05                
R_V = 461.5                 
C_P_D = 1004.0              
G = 9.80665                 
KELVIN_OFFSET = 273.15
MAGNUS_A = 6.1365           
MAGNUS_B = 17.502
MAGNUS_C = 240.97           

@dataclass(frozen=True)
class QCConfig:
    min_temperature_c: float = -60.0
    max_temperature_c: float = 70.0
    min_pressure_hpa: float = 500.0
    max_pressure_hpa: float = 1100.0
    max_wind_speed_ms: float = 130.0
    max_temperature_rate_c_h: float = 19.44
    max_rh_rate_pct_h: float = 50.0
    max_pressure_rate_hpa_h: float = 15.0
    max_wind_rate_ms_h: float = 10.29
    event_temperature_fall_c_h: float = 4.0
    event_rh_rise_pct_h: float = 10.0
    event_pressure_change_hpa_h: float = 2.0
    event_wind_rise_ms_h: float = 6.0
    minimum_event_evidence: int = 3
    comparison_window_minutes: float = 30.0
    reference_tolerance_minutes: float = 5.0
    persistence_window_minutes: float = 180.0
    minimum_persistence_samples: int = 10

CONFIG = QCConfig()

def saturation_vapor_pressure_hpa(temperature_c):
    temperature_c = np.asarray(temperature_c, dtype=float)
    return MAGNUS_A * np.exp(MAGNUS_B * temperature_c / (MAGNUS_C + temperature_c))

def add_physics_features(data):
    df = data.copy()
    temperature_c = pd.to_numeric(df["temperature_c"], errors="coerce")
    rh_pct = pd.to_numeric(df["rh_pct"], errors="coerce")
    pressure_hpa = pd.to_numeric(df["pressure_hpa"], errors="coerce")
    
    temperature_k = temperature_c + KELVIN_OFFSET
    saturation_pressure = saturation_vapor_pressure_hpa(temperature_c)
    vapor_pressure = (rh_pct / 100.0) * saturation_pressure
    dry_air_pressure = pressure_hpa - vapor_pressure
    mixing_ratio = EPSILON * vapor_pressure / dry_air_pressure
    
    df["dew_point_c"] = MAGNUS_C * np.log(vapor_pressure / MAGNUS_A) / (MAGNUS_B - np.log(vapor_pressure / MAGNUS_A))
    df["moist_air_density_kg_m3"] = ((pressure_hpa * 100.0 - vapor_pressure * 100.0) / (R_D * temperature_k)) + (vapor_pressure * 100.0 / (R_V * temperature_k))
    return df

def run_skyguard_physics(df):
    """Clean entry point for the FastAPI server"""
    # The server will pass a clean DataFrame with history here
    return add_physics_features(df)