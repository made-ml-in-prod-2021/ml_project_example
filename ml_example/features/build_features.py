from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ml_example.enities.feature_params import FeatureParams


def impute_features(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    features_transformed = imputer.fit_transform(df)
    features_pandas = pd.DataFrame(
        features_transformed, columns=df.columns, index=df.index,
    )
    return features_pandas


def impute_categorical_features(df: pd.DataFrame):
    return impute_features(df, strategy="most_frequent")


def impute_numerical_features(df: pd.DataFrame):
    return impute_features(df, strategy="mean")


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_df = impute_categorical_features(categorical_df)
    return pd.get_dummies(categorical_df, dummy_na=False)


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    return impute_numerical_features(numerical_df)


def drop_features(
    df: pd.DataFrame, params: FeatureParams
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    num_features = params.numerical_features.copy()
    cat_features = params.categorical_features.copy()

    df = df.drop(params.features_to_drop, axis=1)
    for x in params.features_to_drop:
        if x in num_features:
            num_features.remove(x)
        if x in cat_features:
            cat_features.remove(x)
    return df, cat_features, num_features


def make_features(
    df: pd.DataFrame, params: FeatureParams, test_mode: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    features = df[params.numerical_features + params.categorical_features]
    features, categorical_features, numerical_features = drop_features(features, params)

    categorical_features_df = features[categorical_features]
    categorical_features_transformed = process_categorical_features(
        categorical_features_df
    )

    numerical_features_df = process_numerical_features(features[numerical_features])

    ready_features_df = pd.concat(
        [categorical_features_transformed, numerical_features_df], axis=1
    )
    if test_mode:
        return ready_features_df, None
    else:
        target = df[params.target_col]
        if params.use_log_trick:
            target = pd.Series(np.log(target.to_numpy()))
        return ready_features_df, target
