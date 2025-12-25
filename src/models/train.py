from pathlib import Path
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v1.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "random_forest_v1.joblib"
FEATURES_PATH = MODEL_DIR / "random_forest_features_v1.pkl"

# ---------------- MLflow SETUP ----------------
MLRUNS_DIR = PROJECT_ROOT / "experiments" / "mlruns"
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.set_experiment("Savings_Prediction_Experiment")

# ---------------- CONFIG ----------------
TARGET_COLUMN = "Desired_Savings"
RANDOM_STATE = 42

LEAKAGE_COLS = [
    "Disposable_Income",
    "Desired_Savings_Percentage"
]

def train():
    with mlflow.start_run(run_name="RandomForest_v1"):

        # ---------------- LOAD DATA ----------------
        df = pd.read_csv(DATA_PATH)

        # Keep numeric columns only
        df = df.select_dtypes(include="number")

        # Remove leakage columns
        df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

        # Split features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # 🔑 SAVE FEATURE NAMES (CRITICAL FOR DEPLOYMENT)
        feature_names = X.columns.tolist()
        joblib.dump(feature_names, FEATURES_PATH)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # ---------------- TRAIN MODEL ----------------
        n_estimators = 200
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE
        )

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", RANDOM_STATE)

        model.fit(X_train, y_train)

        # ---------------- EVALUATE ----------------
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        print("\nRandom Forest Training Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

        # ---------------- SAVE MODEL ----------------
        joblib.dump(model, MODEL_PATH)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="SavingsPredictionModel"
        )

        print(f"\n✅ Model saved at: {MODEL_PATH}")
        print("✅ Model logged to MLflow")

        return model, {"MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    train()
