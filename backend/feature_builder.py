import numpy as np
import pandas as pd


FEATURES = [
    "temp_c",
    "temp_delta_1h",
    "temp_delta_5h",
    "temp_dev_10h",
    "temp_std_10h",
    "temp_trend_30h",
    "temp_stuck_run",

    "pressure_hpa",
    "pressure_delta_1h",
    "pressure_delta_5h",
    "pressure_dev_10h",
    "pressure_std_10h",
    "pressure_trend_30h",
    "pressure_stuck_run",

    "rh_pct",
    "rh_delta_1h",
    "rh_delta_5h",
    "rh_dev_10h",
    "rh_std_10h",
    "rh_trend_30h",
    "rh_stuck_run",

    "time_sin",
    "time_cos",
    "season_sin",
    "season_cos",
]


def add_stuck_run(series):
    """
    Count consecutive repeated sensor values.
    """

    values = series.to_numpy()

    result = np.zeros(len(values), dtype=float)

    if len(values) == 0:
        return pd.Series(result, index=series.index)

    run = 1

    for i in range(1, len(values)):

        if pd.isna(values[i]) or pd.isna(values[i - 1]):
            run = 1

        elif values[i] == values[i - 1]:
            run += 1

        else:
            run = 1

        result[i] = run

    result[0] = 1

    return pd.Series(
        result,
        index=series.index
    )


def add_variable_features(df, column, prefix):

    df = df.copy()

    # Difference from previous hour
    df[f"{prefix}_delta_1h"] = (
        df[column].diff(1)
    )

    # Difference from 5 hours ago
    df[f"{prefix}_delta_5h"] = (
        df[column].diff(5)
    )

    # Rolling 10-hour statistics
    rolling_10 = df[column].rolling(
        window=10,
        min_periods=3
    )

    df[f"{prefix}_dev_10h"] = (
        df[column] - rolling_10.mean()
    )

    df[f"{prefix}_std_10h"] = (
        rolling_10.std()
    )

    # Change over 30 hours
    df[f"{prefix}_trend_30h"] = (
        df[column] - df[column].shift(30)
    )

    # Repeated-value detection
    df[f"{prefix}_stuck_run"] = (
        add_stuck_run(df[column])
    )

    return df


def build_features(df):

    df = df.copy()

    # -------------------------------------------------
    # TIMESTAMP
    # -------------------------------------------------

    if "date" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    elif "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    else:

        raise ValueError(
            "Dataset must contain a date or timestamp column."
        )


    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    # -------------------------------------------------
    # NUMERIC SENSOR VALUES
    # -------------------------------------------------

    sensor_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed"
    ]

    for column in sensor_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # -------------------------------------------------
    # STANDARD NAMES
    # -------------------------------------------------

    df["temp_c"] = df["temperature"]

    df["rh_pct"] = df["humidity"]

    df["pressure_hpa"] = df["pressure"]


    # -------------------------------------------------
    # TEMPERATURE FEATURES
    # -------------------------------------------------

    df = add_variable_features(
        df,
        "temp_c",
        "temp"
    )


    # -------------------------------------------------
    # PRESSURE FEATURES
    # -------------------------------------------------

    df = add_variable_features(
        df,
        "pressure_hpa",
        "pressure"
    )


    # -------------------------------------------------
    # HUMIDITY FEATURES
    # -------------------------------------------------

    df = add_variable_features(
        df,
        "rh_pct",
        "rh"
    )


    # -------------------------------------------------
    # TIME FEATURES
    # -------------------------------------------------

    hour = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60.0
    )

    month = df["timestamp"].dt.month


    df["time_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["time_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    df["season_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["season_cos"] = np.cos(
        2 * np.pi * month / 12
    )


    return df