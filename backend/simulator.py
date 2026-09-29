import time
import requests
import pandas as pd

URL = "http://127.0.0.1:8000/ingest"
HEADERS = {"x-api-token": "moes2026_secure"}

print("📡 Loading multi-station historical data...")
df_01 = pd.read_csv("odisha_25years_clean.csv")  # Puri
df_02 = pd.read_csv("bhubaneswar_25years.csv")   # Bhubaneswar
df_03 = pd.read_csv("cuttack_25years.csv")       # Cuttack

print("🚀 Streaming synchronized telemetry for 3 AWS stations...")

session = requests.Session()
session.headers.update(HEADERS)

# Stream data synchronously across all three stations
for i in range(len(df_01)):
    payloads = [
        {
            "station_id": "AWS-Odisha-01",
            "temperature": float(df_01.at[i, 'temperature']),
            "humidity": float(df_01.at[i, 'humidity']),
            "pressure": float(df_01.at[i, 'pressure']),
            "wind_speed": float(df_01.at[i, 'wind_speed'])
        },
        {
            "station_id": "AWS-Odisha-02",
            "temperature": float(df_02.at[i, 'temperature']),
            "humidity": float(df_02.at[i, 'humidity']),
            "pressure": float(df_02.at[i, 'pressure']),
            "wind_speed": float(df_02.at[i, 'wind_speed'])
        },
        {
            "station_id": "AWS-Odisha-03",
            "temperature": float(df_03.at[i, 'temperature']),
            "humidity": float(df_03.at[i, 'humidity']),
            "pressure": float(df_03.at[i, 'pressure']),
            "wind_speed": float(df_03.at[i, 'wind_speed'])
        }
    ]
    
    for payload in payloads:
        try:
            res = session.post(URL, json=payload)
            if res.status_code != 200:
                print(f"❌ Blocked: {res.text}")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            
    print(f"✅ Tick {i}: Real data from Puri, Bhubaneswar, and Cuttack ingested")
    time.sleep(1)