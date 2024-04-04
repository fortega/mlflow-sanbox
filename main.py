from abc import abstractmethod
import logging
from typing import Any, Dict, Generic, Optional, TypeVar
import os
import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
from pandas import DataFrame
import numpy as np

logging.basicConfig(level=logging.INFO)

T = TypeVar("T")


class AbstractModel(Generic[T]):
    def __init__(self, model: T):
        self.model = model

    @abstractmethod
    def transform(self, input: DataFrame) -> DataFrame:
        pass

class MlFlowModel(PythonModel):
    def __init__(self, model: AbstractModel):
        self.model = model
    def predict(self, context, model_input, params: Optional[Dict[str, Any]] = None):
        return self.model.transform(model_input)

class AbstractTrain(Generic[T]):
    def __init__(
        self,
        name: str,
        author: str,
    ):
        self.name = name
        self.author = author

    @abstractmethod
    def train(self) -> AbstractModel[T]:
        pass


class AddModel(AbstractModel[int]):
    def transform(self, input: DataFrame) -> DataFrame:
        for column in input.columns:
            input[f"{column}_plus_{self.model}"] = input[column] + self.model
        return input


def mlflow_config(env: Dict[str, str] = os.environ) -> mlflow.ActiveRun:
    mlflow_tracking_uri = env.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

def run(name: str, model : AbstractModel):
    mlflow_config()

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name,
            python_model=MlFlowModel(model),
            pip_requirements=["pandas"],
        )
        run_id = run.info.run_id

    url = f"runs:/{run_id}/{name}"
    print(url)
    model = mlflow.pyfunc.load_model(url)

    input = pd.DataFrame([{"n": i} for i in range(10)])

    result = model.predict(data=input)
    print(result)


if __name__ == "__main__":
    run("model", AddModel(2))
