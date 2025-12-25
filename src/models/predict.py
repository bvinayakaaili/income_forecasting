from pathlib import Path
import pandas as pd
import joblib

# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_v1.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v1.csv"

# ---------------- CONFIG ----------------
TARGET_COLUMN = "Desired_Savings"

LEAKAGE_COLS = [
    "Disposable_Income",
    "Desired_Savings_Percentage"
]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def predict():
    model = load_model()
    df = load_data()

    # numeric only
    df = df.select_dtypes(include="number")

    # remove leakage
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COLUMN])

    predictions = model.predict(X)

    df["Predicted_Desired_Savings"] = predictions

    print("\nSample predictions:")
    print(df[["Predicted_Desired_Savings"]].head())

    return df

if __name__ == "__main__":
    predict()
