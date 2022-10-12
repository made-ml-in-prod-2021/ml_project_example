import json
import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd

from ml_example.data import read_data, split_train_val_data
from ml_example.data.make_dataset import download_data_from_s3
from ml_example.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_example.features import make_features
from ml_example.features.build_features import extract_target, build_transformer
from ml_example.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)
import mlflow

from ml_example.models.model_fit_predict import create_inference_pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)

    if training_pipeline_params.use_mlflow:

        mlflow.set_tracking_uri(training_pipeline_params.mlflow_uri)
        mlflow.set_experiment(training_pipeline_params.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_artifact(config_path)
            model_path, metrics = run_train_pipeline(training_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
    else:
        return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    # downloading_params = training_pipeline_params.downloading_params
    # if downloading_params:
    #     os.makedirs(downloading_params.output_folder, exist_ok=True)
    #     for path in downloading_params.paths:
    #         download_data_from_s3(
    #             downloading_params.s3_bucket,
    #             path,
    #             os.path.join(downloading_params.output_folder, Path(path).name),
    #         )

    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    logger.info(f"train_features.shape is {train_features.shape}")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts = predict_model(
        inference_pipeline,
        val_df,
        training_pipeline_params.feature_params.use_log_trick,
    )
    metrics = evaluate_model(
        predicts,
        val_target,
        use_log_trick=training_pipeline_params.feature_params.use_log_trick,
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
