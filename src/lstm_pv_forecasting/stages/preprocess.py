import argparse
from datetime import datetime
import logging
import os
import pathlib
import pickle
from typing import List

import keras
import pandas as pd

from lstm_pv_forecasting.util import logging as util_logging
from lstm_pv_forecasting.util.load import load_dataset
from lstm_pv_forecasting.util.load import normalize_datasets
from lstm_pv_forecasting.util.load_config import load_config

logger = logging.getLogger(__name__)


def preprocess_dataset(data_filename, dataset_params: dict, split_dates: list[datetime], num_fourier_terms: int = 1,
                       output_dir: str | os.PathLike = "data/",
                       random_seed: int = 1,
                       discard_test_year_not_full: bool = False):
    """
    Preprocess dataset and save it to disk.

    Args:
        data_filename (str): Path to the raw dataset.
        dataset_params (dict): Parameters to load and preprocess the dataset.
        split_dates (list[datetime]): List of dates to split the dataset.
        num_fourier_terms (int, optional): Number of fourier terms to add for each seasonality. Defaults to 1.
        output_dir (str | os.PathLike, optional): Path to save the preprocessed dataset. Defaults to "data/".
        random_seed (int, optional): Random seed. Defaults to 1.
        discard_test_year_not_full (bool, optional): Only retains a year in the test set if it is full. Defaults to False.
    """
    keras.utils.set_random_seed(random_seed)

    logger.info("Prepare stage")
    logger.info(f"reading raw dataset from {data_filename}")
    dataset = load_dataset(data_filename, split_method="datetimes",
                           split_dates=split_dates, num_fourier_terms=num_fourier_terms,
                           **dataset_params)
    [train_df, val_df, test_df], scaler, target_scaler = normalize_datasets(
        dataset, target_column_name='radiance', columns_to_normalize=['radiance', 'temperature'])

    logger.info(f"val dataset start date {val_df.index[0].date()}")
    logger.info(f"test dataset start date {test_df.index[0].date()}")

    output_dir_path = pathlib.Path(output_dir)
    train_df.to_pickle(output_dir_path.joinpath("train.pkl"))
    val_df.to_pickle(output_dir_path.joinpath("val.pkl"))

    if discard_test_year_not_full:
        test_end_date = test_df.index[-1].date()
        if test_end_date.month != 12 or test_end_date.day != 31:
            logger.info("Discarding the last incomplete year in the test set")
            new_end_timestamp = pd.Timestamp(test_end_date.replace(
                month=1, day=1)) - pd.Timedelta(minutes=1)
            test_df = test_df.loc[test_df.index[0]: new_end_timestamp, :]
            logger.info(f"test dataset end date {test_df.index[-1].date()}")

    test_df.to_pickle(output_dir_path.joinpath("test.pkl"))

    save_path = output_dir_path.joinpath("target_scaler.pkl")
    logger.info(f"saving prepared data to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(target_scaler, f)


if __name__ == "__main__":
    util_logging.setup_logging(logfilename_prefix="preprocess")

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_config(config_path=args.config)

    data_dir = params["base"]["data_dir"]
    data_filename = params["dataset"]["data_filename"]
    data_filename = pathlib.Path(data_dir).joinpath(data_filename).as_posix()
    num_fourier_terms = params["dataset"]["preprocess"]["num_fourier_terms"]
    random_seed = params["base"]["random_seed"]

    split_method = params["dataset"]["preprocess"]["split_method"]
    split_dates = params["dataset"]["preprocess"]["split_dates"]
    split_dates = [datetime.strptime(d, '%d/%m/%Y') for d in split_dates]

    discard_test_year_not_full = params["dataset"]["preprocess"].get(
        "discard_test_year_not_full", False)

    dataset_params = params["dataset"]["parameters"]
    preprocess_dataset(data_filename, dataset_params=dataset_params,
                       split_dates=split_dates,
                       num_fourier_terms=num_fourier_terms,
                       output_dir=data_dir,
                       random_seed=random_seed,
                       discard_test_year_not_full=discard_test_year_not_full)
