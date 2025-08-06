import argparse
import datetime
from json import load
import logging
import os
import pathlib

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from lstm_pv_forecasting.util import logging as util_logging
from lstm_pv_forecasting.util.load import load_from_file
from lstm_pv_forecasting.util.load import normalize_datasets
from lstm_pv_forecasting.util.load import split_datasets
from lstm_pv_forecasting.util.load_config import load_config
from lstm_pv_forecasting.util.period_stats import calculate_period_stats

logger = logging.getLogger(__name__)


def calc_daily_stats(data_filename, dataset_params: dict, split_dates: list[datetime.datetime],
                     latitude: float,
                     std: list[float],
                     noise_dataset_selection: list[bool],
                     output_dir: str | os.PathLike = "data/",
                     random_seed: int = 1):
    """Calculate the daily statistics and split the dataset into train, validation, dev and test datasets.

    Results are stored in the output directory using pickle files.

    Args:
        data_filename (str | os.PathLike): The path to the data file.
        dataset_params (dict): The parameters to load the dataset.
        split_dates (list[datetime.datetime]): The dates to split the dataset.
        latitude (float): The latitude of the location.
        std (list[float]): The standard deviation values for each column.
        noise_dataset_selection (list[bool]): The selection of the datasets to add noise to.
        output_dir (str | os.PathLike, optional): The output directory. Defaults to "data/".
        random_seed (int, optional): The random seed. Defaults to 1.
    """
    keras.utils.set_random_seed(random_seed)

    logger.info("calc daily stats stage")

    # noise_dataset_selection must contain a number of elements equal to the number of datasets
    if len(noise_dataset_selection) != (len(split_dates) + 1):
        raise ValueError(
            "The noise_dataset_selection list must contain a number of elements equal to the number of datasets")

    dataset = load_from_file(data_filename, **dataset_params)

    daily_stats = calculate_period_stats(
        dataset, stat_period_duration=datetime.timedelta(days=1), latitude=latitude)

    # Split the dataset
    splitted_datasets = split_datasets(
        dataset=daily_stats,
        split_method="datetimes",
        split_dates=split_dates,
        split_ratios=None,
        used_columns=["mean_temp", "Kt"]
    )

    # add gaussian noise to the selected dataset datasets
    noisy_datasets = []
    for i, ds in enumerate(splitted_datasets):
        if noise_dataset_selection[i] == True:
            noisy_datasets.append(add_gaussian_noise(
                ds, mean=[0, 0], std=std))
        else:
            noisy_datasets.append(ds)

    output_dir_path = pathlib.Path(output_dir)
    # train_daily_stats.to_pickle(
    #     output_dir_path.joinpath("train_daily_stats.pkl"))
    # val_daily_stats.to_pickle(output_dir_path.joinpath("val_daily_stats.pkl"))
    # dev_daily_stats.to_pickle(output_dir_path.joinpath("dev_daily_stats.pkl"))
    # test_daily_stats.to_pickle(
    #     output_dir_path.joinpath("test_daily_stats.pkl"))

    # normalize the datasets
    normalized_datasets, _, _ = normalize_datasets(noisy_datasets,
                                                   columns_to_normalize=["mean_temp", "Kt"])

    dataset_names = ["train", "val", "test"]

    for i, ds in enumerate(normalized_datasets):
        ds.to_pickle(output_dir_path.joinpath(
            f"{dataset_names[i]}_daily_stats_normalized.pkl"))

    logger.info("calc daily stats stage completed")


def add_gaussian_noise(daily_data: pd.DataFrame, mean: list[float], std: list[float]) -> pd.DataFrame:
    """Add gaussian noise to each column of the daily data.

    Args:
        daily_data (pd.DataFrame): The daily data.
        mean (List[float]): The mean values for each column.
        std (List[float]): The standard deviation values for each column.
        random_seed (int, optional): The random seed. Defaults to 1.

    Returns:
        pd.DataFrame: The daily data with added noise.
    """
    noisy_data = daily_data.copy()
    for col, m, s in zip(daily_data.columns, mean, std):
        noise = np.random.normal(size=daily_data[col].shape, loc=m, scale=s)
        noisy_data[col] = daily_data[col] + noise
    return noisy_data


if __name__ == "__main__":
    util_logging.setup_logging(logfilename_prefix="calc_daily_stats")

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_config(config_path=args.config)

    data_dir = params["base"]["data_dir"]
    data_filename = params["dataset"]["data_filename"]
    data_filename = pathlib.Path(data_dir).joinpath(data_filename).as_posix()

    random_seed = params["base"]["random_seed"]

    split_method = params["dataset"]["preprocess"]["split_method"]
    split_dates = params["dataset"]["preprocess"]["split_dates"]
    split_dates = [datetime.datetime.strptime(
        d, '%d/%m/%Y') for d in split_dates]

    latitude = params["calc_daily_stats"]["latitude"]
    noise_std = params["calc_daily_stats"]["noise_std"]

    noise_dataset_selection = params["calc_daily_stats"]["noise_dataset_selection"]

    dataset_params = params["dataset"]["parameters"]
    calc_daily_stats(data_filename, dataset_params=dataset_params,
                     split_dates=split_dates,
                     latitude=latitude, std=noise_std,
                     output_dir=data_dir,
                     noise_dataset_selection=noise_dataset_selection,
                     random_seed=random_seed)
