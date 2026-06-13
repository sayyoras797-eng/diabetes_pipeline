FROM python:3.11-slim
WORKDIR /app

COPY dagster_pipeline/requirements.txt /app/dagster_pipeline/requirements.txt
RUN pip install --no-cache-dir -r /app/dagster_pipeline/requirements.txt

COPY . /app/diabetes_pipeline

ENV DAGSTER_HOME=/app/diabetes_pipeline/dagster_pipeline
WORKDIR /app/diabetes_pipeline/dagster_pipeline

EXPOSE 3000
