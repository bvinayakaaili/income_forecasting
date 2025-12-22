from pathlib import Path
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------- paths --------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v1.csv"

# -------- config --------
TARGET_COLUMN = "Desired_Savings"
  # <-- CHANGE if your target column is different
RANDOM_STATE = 42

def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
    return pd.read_csv(FEATURES_PATH)

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }

def main():
    df = load_features()

    # Drop non-numeric columns if any (safe baseline)
    df = df.select_dtypes(include="number")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    LEAKAGE_COLS = [
    "Disposable_Income",
    "Desired_Savings_Percentage"
    ]

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    print("\n=== MODEL COMPARISON ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
