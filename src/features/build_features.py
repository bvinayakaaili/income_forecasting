import math
from src.utils.config import PROCESSED_DATA_DIR

def build_features(df):
    # Polynomial time feature
    if "year" in df.columns:
        df["year_squared"] = df["year"] ** 2

    # Log transform income
    if "income" in df.columns:
        df["log_income"] = df["income"].apply(
            lambda x: 0 if x <= 0 else math.log(x)
        )

    output_path = PROCESSED_DATA_DIR / "features_v2.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Feature-engineered data saved to {output_path}")
    return df
