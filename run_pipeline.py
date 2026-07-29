import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


def run_command(cmd_description, python_module_path):
    print("\n==========================================")
    print(f"▶ STARTING STAGE: {cmd_description}")
    print("==========================================")
    result = os.system(f"{sys.executable} {python_module_path}")
    if result != 0:
        print(f"❌ Stage failed: {cmd_description}")
        sys.exit(1)
    print(f"✅ COMPLETED: {cmd_description}")


def main():
    print("🚀 Running Full End-to-End Market Prediction Pipeline...")

    # Stage 1: Ingestion
    run_command(
        "1. Data Ingestion (Crypto & Stocks)", BASE_DIR / "src" / "data" / "ingest.py"
    )

    # Stage 2: Cleaning
    run_command("2. Data Cleaning & Validation", BASE_DIR / "src" / "data" / "clean.py")

    # Stage 3: Feature Engineering
    run_command(
        "3. Feature Engineering & Merging",
        BASE_DIR / "src" / "features" / "build_features.py",
    )

    # Stage 4: Model Training & Tuning
    run_command(
        "4. XGBoost Model Training & Optuna Tuning",
        BASE_DIR / "databricks" / "notebooks" / "02_train.py",
    )

    # Stage 5: Unit & API Testing
    run_command("5. API & Model Unit Tests", BASE_DIR / "tests" / "test_api.py")

    # Stage 6: Data Drift Monitoring
    run_command(
        "6. Evidently AI Data Drift Monitoring",
        BASE_DIR / "monitoring" / "drift_report.py",
    )

    print("\n🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
