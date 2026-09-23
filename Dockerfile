FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas==2.3.2 pyarrow==25.0.1 xgboost==3.4.1 scikit-learn==1.9.0
RUN pip install --no-cache-dir sqlalchemy==2.0.54 psycopg2-binary==2.9.13
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.52.4
RUN pip install --no-cache-dir joblib==1.6.0 python-dotenv==1.2.3 colorlog==6.12.0

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
