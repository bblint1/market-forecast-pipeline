from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
REPORTS_DIR = BASE_DIR / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_drift_report():
    files = sorted(FEATURES_DIR.glob("features_20*.parquet"))
    if len(files) < 2:
        print("Need at least 2 features_YYYYMMDD files to compute drift.")
        return

    reference_df = pd.read_parquet(files[0])
    current_df = pd.read_parquet(files[-1])

    common_cols = [
        c
        for c in reference_df.columns
        if c in current_df.columns and c not in ["datetime"] and "target" not in c
    ]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[common_cols],
        current_data=current_df[common_cols],
    )

    report_path = REPORTS_DIR / "data_drift_report.html"
    report.save_html(str(report_path))
    print(f"Drift report generated at {report_path}")


if __name__ == "__main__":
    generate_drift_report()
