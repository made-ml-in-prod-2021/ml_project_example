# -*- coding: utf-8 -*-
from typing import Tuple, NoReturn

import pandas as pd
from boto3 import client
from sklearn.model_selection import train_test_split

from ml_example.enities import SplittingParams


def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> NoReturn:
    s3 = client("s3")
    s3.download_file(s3_bucket, s3_path, output)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :rtype: object
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
