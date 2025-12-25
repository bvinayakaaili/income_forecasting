import pandas as pd
from src.utils.config import RAW_DATA_DIR

def load_raw_data():
    path = RAW_DATA_DIR / "income_ineuality_india.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_csv(path)
