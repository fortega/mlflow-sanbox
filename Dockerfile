FROM python:3.9

WORKDIR /app

COPY pyproject.toml .
COPY mlflow-fp mlflow-fp

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

CMD mlflow server -h 0.0.0.0