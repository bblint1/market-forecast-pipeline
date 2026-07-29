FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic xgboost pandas pyarrow scikit-learn numpy

# Copy application source code
COPY src/ src/
COPY data/ data/
COPY models/ models/

EXPOSE 7000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "7000"]
