import json
import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from physics_engine import run_skyguard_physics
from feature_builder import build_features, FEATURES as ML_FEATURES

# --- AUTOENCODER HEALTH ENGINE ---
class SkyGuardAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, x):
        return self.decoder(self.encoder(x))

try:
    ae_bundle = joblib.load("skyguard_pytorch_autoencoder.joblib")
    HEALTH_MODEL = SkyGuardAutoencoder()
    HEALTH_MODEL.load_state_dict(ae_bundle["model_state"])
    HEALTH_MODEL.eval()
    HEALTH_SCALER = ae_bundle["scaler"]
    HEALTH_BASELINE_ERROR = ae_bundle["baseline_max_error"]
    print("✅ PyTorch Health Engine Loaded")
except Exception as e:
    print("⚠️ Autoencoder missing or failed to load:", e)
    HEALTH_MODEL = None

app = FastAPI(title="SkyGuard Secure ML Pipeline")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("🚀 Waking up SkyGuard Spatial ML Engine...")

BUNDLE = joblib.load(
    "skyguard_isolation_forest_v2.joblib"
)

MODEL = BUNDLE["model"]

FEATURES = BUNDLE["feature_columns"]

THRESHOLD = BUNDLE["threshold"]

print(
    f"✅ Isolation Forest V2 loaded "
    f"with {len(FEATURES)} features"
)

print(
    f"✅ Anomaly threshold: {THRESHOLD:.6f}"
)

# Initialize the Real SHAP Explainer using our trained Isolation Forest
EXPLAINER = shap.TreeExplainer(MODEL)

class ConnectionManager:
    def __init__(self): 
        self.active = []
        
    async def connect(self, ws: WebSocket): 
        await ws.accept()
        self.active.append(ws)
        
    def disconnect(self, ws: WebSocket): 
        if ws in self.active:
            self.active.remove(ws)
            
    async def broadcast(self, data: dict):
        for ws in self.active:
            try: 
                await ws.send_text(json.dumps(data))
            except Exception: 
                pass

manager = ConnectionManager()

class TelemetryData(BaseModel):
    station_id: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float

telemetry_buffers = {}

@app.post("/ingest")
async def ingest_data(data: TelemetryData, x_api_token: str = Header(None)):
    if x_api_token != "moes2026_secure":
        raise HTTPException(status_code=401, detail="ACCESS DENIED")
        
    now = pd.Timestamp.now()
    station = data.station_id
    
    if station not in telemetry_buffers:
        telemetry_buffers[station] = []
        
    buffer = telemetry_buffers[station]
    buffer.append({
    "timestamp": now,
    "temperature": data.temperature,
    "humidity": data.humidity,
    "pressure": data.pressure,
    "wind_speed": data.wind_speed,
    "temperature_c": data.temperature,
    "rh_pct": data.humidity,
    "pressure_hpa": data.pressure
})
    
    if len(buffer) > 35:
        buffer.pop(0)
        
    status = "BUFFERING"
    xai_blame = None
    
    # Check if all three regional stations have active readings
    has_all_stations = (
        len(telemetry_buffers.get("AWS-Odisha-01", [])) > 0 and
        len(telemetry_buffers.get("AWS-Odisha-02", [])) > 0 and
        len(telemetry_buffers.get("AWS-Odisha-03", [])) > 0
    )
    
    # Only process the heavy ML and Physics if we are the primary station and buffers are full
    if station == "AWS-Odisha-01" and has_all_stations and len(buffer) == 35:
        
        # Calculate thermodynamics on the primary AWS
        df_physics = pd.DataFrame(buffer)
        df_physics = run_skyguard_physics(df_physics)
        
        s1 = telemetry_buffers["AWS-Odisha-01"][-1]
        s2 = telemetry_buffers["AWS-Odisha-02"][-1]
        s3 = telemetry_buffers["AWS-Odisha-03"][-1]

        spatial_dict = {
            "temperature_01": s1["temperature"],
            "humidity_01": s1["humidity"],
            "pressure_01": s1["pressure"],

            "temp_diff_01_02":
                s1["temperature"] - s2["temperature"],

            "temp_diff_01_03":
                s1["temperature"] - s3["temperature"],

            "pressure_diff_01_02":
                s1["pressure"] - s2["pressure"],

            "pressure_diff_01_03":
                s1["pressure"] - s3["pressure"]
        }
        
        # Build the exact spatial vector expected by the new Isolation Forest
               # ---------------------------------------------------------
        # BUILD 25 TEMPORAL FEATURES FOR ISOLATION FOREST V2
        # ---------------------------------------------------------

        # Use the primary station's buffered historical readings.
        # The new Isolation Forest was trained using feature_builder.py,
        # so inference must use the exact same feature-generation process.

        df_ml = pd.DataFrame(
            telemetry_buffers["AWS-Odisha-01"]
        ).copy()

        # Make sure timestamp is a proper datetime column
        df_ml["timestamp"] = pd.to_datetime(
            df_ml["timestamp"]
        )

        # Build the same 25 features used during training
        df_ml = build_features(df_ml)

        # Take the newest feature row
        latest_features = df_ml[ML_FEATURES].tail(1)

        # Check for invalid / unavailable features
        if latest_features.isna().any().any():

            print("⚠️ Still buffering temporal ML features...")

            anomaly_score = 0.0

        else:

            # -----------------------------------------------------
            # ISOLATION FOREST V2
            # -----------------------------------------------------

            anomaly_score = float(
                -MODEL.score_samples(latest_features)[0]
            )

            print(
                f"🌲 Isolation Forest V2 score: "
                f"{anomaly_score:.6f}"
            )
        
        if latest_features.isna().any().any():
            print("⚠️ Still buffering spatial data...")
        else:

            
            # --- GRANULAR DECISION ENGINE ---
            max_temp_diff = max(abs(spatial_dict["temp_diff_01_02"]), abs(spatial_dict["temp_diff_01_03"]))
            max_pressure_diff = max(abs(spatial_dict["pressure_diff_01_02"]), abs(spatial_dict["pressure_diff_01_03"]))
            
            if data.temperature <= -40.0 or data.temperature >= 60.0:
                status = "OUT_OF_RANGE"
            elif max_temp_diff > 8.0 or max_pressure_diff > 15.0:
                status = "SPATIAL_INCONSISTENCY_FAULT"
            elif anomaly_score >= THRESHOLD:
                status = "REGIONAL_METEOROLOGICAL_ANOMALY"
            else:
                status = "NORMAL"
                
            # --- REAL SHAP XAI ENGINE ---
                        # -----------------------------------------------------
            # REAL SHAP XAI ENGINE
            # -----------------------------------------------------

            shap_vals = EXPLAINER.shap_values(latest_features)

            vals = (
                shap_vals[0]
                if len(shap_vals.shape) > 1
                else shap_vals
            )

            xai_impact = {
                ML_FEATURES[i]: abs(float(vals[i]))
                for i in range(len(ML_FEATURES))
            }
            total_shap = sum(xai_impact.values()) + 0.0001
            danger_multiplier = 1.0 if status != "NORMAL" else 0.3
            xai_blame = {k: round((v / total_shap) * 100 * danger_multiplier, 1) for k, v in xai_impact.items()}
                
            # --- NEURAL NETWORK HEALTH SCORE & SELF-HEALING ---
            health_score = 100.0
            health_status = "Healthy"
            estimated_vals = {"temp": data.temperature, "pressure": data.pressure, "humidity": data.humidity}

            if HEALTH_MODEL is not None:
                raw_inputs = np.array([[data.temperature, data.pressure, data.humidity]])
                scaled_inputs = HEALTH_SCALER.transform(raw_inputs)
                tensor_inputs = torch.FloatTensor(scaled_inputs)

                with torch.no_grad():
                    reconstructed_tensor = HEALTH_MODEL(tensor_inputs)
                
                reconstructed_scaled = reconstructed_tensor.numpy()
                mse = float(np.mean(np.square(scaled_inputs - reconstructed_scaled)))
                
                health_penalty = (mse / HEALTH_BASELINE_ERROR) * 100
                health_score = max(0.0, min(100.0, 100.0 - health_penalty))
                
                reconstructed_actual = HEALTH_SCALER.inverse_transform(reconstructed_scaled)[0]
                estimated_vals = {
                    "temp": round(reconstructed_actual[0], 2),
                    "pressure": round(reconstructed_actual[1], 2),
                    "humidity": round(reconstructed_actual[2], 2)
                }

                if health_score >= 80: health_status = "Healthy"
                elif health_score >= 60: health_status = "Minor degradation"
                elif health_score >= 40: health_status = "Warning"
                elif health_score >= 20: health_status = "Critical"
                else: health_status = "Failed or unreliable"

            print(f"📡 {station} | SCORE: {anomaly_score:.3f} | {status} | HEALTH: {health_score:.1f}")

            await manager.broadcast({
                "timestamp": now.isoformat(),
                "station_id": station,
                "status": status,
                "metrics": {
                    "temperature": data.temperature,
                    "pressure": data.pressure,
                    "humidity": data.humidity,
                    "wind_speed": data.wind_speed,
                    "anomaly_score": anomaly_score,
                    "xai_engine": xai_blame,
                    "health": {
                        "score": round(health_score, 1),
                        "status": health_status,
                        "estimated_true_values": estimated_vals
                    }
                }
            })

    return {"status": "SUCCESS"}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: 
            await websocket.receive_text()
    except WebSocketDisconnect: 
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)