import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Savings_Prediction_Experiments")

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v1.csv"

TARGET_COLUMN = "Desired_Savings"
   # change if needed
RANDOM_STATE = 42

def eval_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.select_dtypes(include="number")
    df = pd.get_dummies(df, drop_first=True)


    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

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
            random_state=RANDOM_STATE,
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = eval_metrics(y_test, preds)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            # 👇 THIS REGISTERS THE MODEL
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="IncomeInequalityModel"
            )

            print(f"Logged & registered {name}")

if __name__ == "__main__":
    main()
