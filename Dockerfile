FROM python:3.12-slim

WORKDIR /app

COPY trainer.py ./

RUN pip install --no-cache-dir \
    scikit-learn \
    xgboost \
    pandas \
    numpy \
    joblib

ENTRYPOINT ["python", "trainer.py"]
