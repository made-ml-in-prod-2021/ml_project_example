import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ml_example.enities.train_params import TrainingParams

SklearnRegressionModel = Union[RandomForestRegressor, LinearRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LinearRegression":
        model = LinearRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame, use_log_trick: bool = True
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
