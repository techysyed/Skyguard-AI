import joblib
import pandas as pd

from feature_builder import build_features


MODEL_FILE = "skyguard_isolation_forest_v2.joblib"


# Load model
bundle = joblib.load(MODEL_FILE)

model = bundle["model"]
features = bundle["feature_columns"]
threshold = bundle["threshold"]


# Load one dataset
df = pd.read_csv(
    "odisha_25years_clean.csv"
)


# Build exactly the same features used during training
df = build_features(df)


# Select model features
X = df[features].copy()


# Remove invalid values
X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

X = X.dropna()


# Calculate anomaly scores
scores = -model.score_samples(X)


print("\n===================================")
print("SKYGUARD ISOLATION FOREST V2 TEST")
print("===================================")

print(
    "Model loaded successfully."
)

print(
    "Number of features:",
    len(features)
)

print(
    "Number of test samples:",
    len(X)
)

print(
    "Anomaly threshold:",
    threshold
)

print(
    "Minimum anomaly score:",
    scores.min()
)

print(
    "Maximum anomaly score:",
    scores.max()
)

print(
    "Average anomaly score:",
    scores.mean()
)

print(
    "Detected anomalies:",
    (scores >= threshold).sum()
)

print(
    "Anomaly percentage:",
    100 * (scores >= threshold).mean()
)

print("\nTEST COMPLETED.")