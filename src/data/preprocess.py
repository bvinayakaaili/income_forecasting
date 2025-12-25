import pandas as pd
from src.utils.config import PROCESSED_DATA_DIR

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def remove_outliers_iqr(df, numeric_cols):
    """
    Remove outliers using IQR method
    """
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def preprocess_data(df):
    # 1️⃣ Remove duplicates
    df = df.drop_duplicates()

    # 2️⃣ Handle missing values
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3️⃣ Outlier removal (only numeric columns)
    numeric_cols = df.select_dtypes(include="number").columns
    df = remove_outliers_iqr(df, numeric_cols)

    # 4️⃣ Encode categorical variables (One-Hot Encoding)
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 5️⃣ Save cleaned data
    output_path = PROCESSED_DATA_DIR / "cleaned_v2.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Preprocessed data (with outliers removed & encoded) saved to {output_path}")
    return df
