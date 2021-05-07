from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ml_example.data.make_dataset import read_data
from ml_example.enities.feature_params import FeatureParams
from ml_example.features.build_features import make_features, extract_target, build_transformer


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
        use_log_trick=True,
    )
    return params


def test_make_features(
    feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    #assert not pd.isnull(features).any().any()
    #assert all(x not in features.columns for x in feature_params.features_to_drop)


def test_extract_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    target = extract_target(data, feature_params)
    assert_allclose(
        np.log(data[feature_params.target_col].to_numpy()), target.to_numpy()
    )
