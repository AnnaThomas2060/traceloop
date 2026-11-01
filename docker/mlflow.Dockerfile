FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow==2.11.3

RUN mkdir -p /mlflow_artifacts && chmod 777 /mlflow_artifacts

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow_artifacts", "--host", "0.0.0.0", "--port", "5000"]
