import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest

from feature_builder import build_features, FEATURES


# =========================================================
# 1. DATASET FILES
# =========================================================

files = [
    "odisha_25years_clean.csv",
    "bhubaneswar_25years.csv",
    "cuttack_25years.csv"
]


# =========================================================
# 2. LOAD AND BUILD FEATURES
# =========================================================

dataframes = []

for file in files:

    print(f"\nLoading {file}...")

    df = pd.read_csv(file)

    print("Original rows:", len(df))

    df = build_features(df)

    dataframes.append(df)

    print("Features created successfully.")


# =========================================================
# 3. COMBINE ALL THREE DATASETS
# =========================================================

data = pd.concat(
    dataframes,
    ignore_index=True
)

print("\n===================================")
print("COMBINED DATA")
print("===================================")

print("Total rows:", len(data))


# =========================================================
# 4. SELECT FEATURES
# =========================================================

X = data[FEATURES].copy()

print(
    "Number of features:",
    len(FEATURES)
)


# =========================================================
# 5. REMOVE INVALID VALUES
# =========================================================

X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

before = len(X)

X = X.dropna()

after = len(X)

print(
    "Rows removed:",
    before - after
)

print(
    "Rows remaining:",
    after
)


# =========================================================
# 6. TRAIN ISOLATION FOREST
# =========================================================

print("\n===================================")
print("TRAINING ISOLATION FOREST")
print("===================================")

model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X)

print("Training completed.")


# =========================================================
# 7. CALCULATE ANOMALY SCORES
# =========================================================

scores = -model.score_samples(X)


# =========================================================
# 8. CALIBRATE ANOMALY THRESHOLD
# =========================================================

threshold = float(
    pd.Series(scores).quantile(0.995)
)

print(
    "\nAnomaly threshold:",
    threshold
)


# =========================================================
# 9. SAVE MODEL
# =========================================================

bundle = {

    "model": model,

    "feature_columns": FEATURES,

    "threshold": threshold,

    "sampling_minutes": 60,

    "sampling_description":
        "Hourly meteorological observations",

    "version":
        "SkyGuard Isolation Forest V2"

}


output_file = (
    "skyguard_isolation_forest_v2.joblib"
)

joblib.dump(
    bundle,
    output_file
)


# =========================================================
# 10. FINAL INFORMATION
# =========================================================

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print(
    "File:",
    output_file
)

print(
    "Number of features:",
    len(FEATURES)
)

print(
    "Threshold:",
    threshold
)

print("\nFeatures:")

for i, feature in enumerate(FEATURES, 1):

    print(
        f"{i:02d}. {feature}"
    )