import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.providers.standard.operators.python import PythonOperator

from airflow import DAG

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.clean import clean_market_data  # noqa
from src.data.ingest import fetch_crypto_data, fetch_stock_data  # noqa
from src.features.build_features import merge_market_features  # noqa

default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="crypto_stock_market_pipeline",
    default_args=default_args,
    description="End-to-End Market Data Ingestion, Cleaning, and Feature Engineering Pipeline",
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["mlops", "market_data", "crypto", "stocks"],
) as dag:
    ingest_crypto = PythonOperator(
        task_id="ingest_crypto_data",
        python_callable=fetch_crypto_data,
        op_kwargs={"symbol": "BTCUSDT", "interval": "1h", "limit": 1000},
    )

    ingest_stocks = PythonOperator(
        task_id="ingest_stock_data",
        python_callable=fetch_stock_data,
        op_kwargs={"ticker": "SPY", "period": "1mo", "interval": "1h"},
    )

    clean_crypto = PythonOperator(
        task_id="clean_crypto_data",
        python_callable=clean_market_data,
        op_kwargs={"file_type": "crypto"},
    )

    clean_stocks = PythonOperator(
        task_id="clean_stock_data",
        python_callable=clean_market_data,
        op_kwargs={"file_type": "stocks"},
    )

    merge_features = PythonOperator(
        task_id="merge_market_features",
        python_callable=merge_market_features,
    )

    [ingest_crypto >> clean_crypto, ingest_stocks >> clean_stocks] >> merge_features
