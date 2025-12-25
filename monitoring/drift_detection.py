from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v1.csv"
CURRENT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.csv"

REPORT_DIR = PROJECT_ROOT / "monitoring" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "data_drift_report.html"


def generate_drift_report():
    # Load datasets
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    current_df = pd.read_csv(CURRENT_DATA_PATH)

    # Keep numeric columns only
    reference_df = reference_df.select_dtypes(include="number")
    current_df = current_df.select_dtypes(include="number")

    # Create Evidently report
    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    # Save report
    report.save_html(str(REPORT_PATH))

    print(f"✅ Data drift report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    generate_drift_report()
