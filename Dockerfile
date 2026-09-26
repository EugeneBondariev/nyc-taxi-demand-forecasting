FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    pydantic==2.13.5 \
    sqlalchemy==2.0.54 \
    psycopg2-binary==2.9.13 \
    pandas==3.0.6 \
    numpy==2.5.3 \
    joblib==1.6.0 \
    scikit-learn==1.9.1 \
    xgboost==3.4.1 \
    python-dotenv==1.2.3 \
    colorlog==6.12.0

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
