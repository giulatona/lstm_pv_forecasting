from datetime import datetime
from datetime import timezone
from math import isclose
from re import T
from typing import Literal

import numpy as np
import pandas as pd

SPLIT_METHODS = Literal["ratios", "datetimes"]
DATASET_TYPES = Literal["sias", "pvgis"]


def load_dataset(data_filename: str, dataset_type: DATASET_TYPES = "sias",
                 split_method: SPLIT_METHODS = "ratios",
                 split_ratios: list[float] | None = None,
                 split_dates: list[datetime] | None = None,
                 num_fourier_terms: int = 1,
                 rename_columns: dict | None = None) -> list[pd.DataFrame]:
    """Loads datasets, adds fourier terms, and splits the dataset.

    Args:
        data_filename (str): String of the path of the raw dataset.
        dataset_type (DATASET_TYPES, optional): Type of dataset. Can be "sias" or "pvgis". Defaults to "sias".
        split_method (SPLIT_METHODS, optional): Method used to split the dataset. 
            Can be "ratios" or "datetimes". Defaults to "ratios".
        split_ratios (list[float] | None, optional): List of ratios of each dataset sub dataset length. 
            If 'split_method' is not "ratios" this parameter is ignored. Defaults to None.
        split_dates (list[datetime] | None, optional): List of datetimes that separate each sub dataset. 
            If 'split_method' is not "datetimes" this parameter is ignored. Defaults to None.
        num_fourier_terms (int, optional): Number of fourier terms to add for each seasonality. Defaults to 1.
        rename_columns (dict | None, optional): Dictionary to rename columns. Defaults to None.

    Returns:
        list[pd.DataFrame]: List of datasets splitted according to 'split_method'.
    """

    dataset = load_from_file(data_filename, dataset_type, rename_columns)

    used_columns = ['radiance', 'temperature']

    # add time features
    timestamp_s = dataset.index.map(
        lambda t: t.replace(tzinfo=timezone.utc).timestamp())

    day = 24 * 60 * 60
    year = (365.2425) * day

    for k in range(1, num_fourier_terms + 1):
        prefix = 'Day'
        col_name = prefix + f" sin{k}"
        dataset[col_name] = np.sin(timestamp_s * (2 * k * np.pi / day))
        used_columns.append(col_name)

        col_name = prefix + f" cos{k}"
        dataset[col_name] = np.cos(timestamp_s * (2 * k * np.pi / day))
        used_columns.append(col_name)

        prefix = 'Year'
        col_name = prefix + f" sin{k}"
        dataset[col_name] = np.sin(timestamp_s * (2 * k * np.pi / year))
        used_columns.append(col_name)

        col_name = prefix + f" cos{k}"
        dataset[col_name] = np.cos(timestamp_s * (2 * k * np.pi / year))
        used_columns.append(col_name)

    # split dataset
    datasets = split_datasets(
        split_method, split_ratios, split_dates, dataset, used_columns)

    return datasets


def load_from_file(data_filename: str, dataset_type: DATASET_TYPES,
                   rename_columns: dict | None = None) -> pd.DataFrame:
    if dataset_type == "sias":
        dataset = pd.read_csv(data_filename,
                              parse_dates=True, index_col=0)
    elif dataset_type == "pvgis":
        dataset = pd.read_csv(data_filename, sep=",", skiprows=8,
                              skipfooter=10, engine='python',
                              date_format="%Y%m%d:%H%M",
                              parse_dates=True,
                              index_col=0)
        dataset = dataset.shift(periods=-1, freq='10min')

    if rename_columns is not None:
        dataset.rename(columns=rename_columns, inplace=True)

    if dataset_type == "pvgis":
        # radiance is in W/m^2, convert to irradiation in MJ/m^2
        dataset.loc[:, 'radiance'] = dataset.loc[:, 'radiance'] * 1e-6 * 3600

    start_date = dataset.index[0].date()
    end_date = dataset.index[-1].date()

    # dataset = dataset.resample('30T').mean()

    # dataset = fill_missing_values(dataset)
    return dataset


def split_datasets(split_method, split_ratios, split_dates, dataset, used_columns) -> list[pd.DataFrame]:

    start_date = dataset.index[0].date()
    end_datetime = dataset.index[-1]
    # make sure that the last full day is included in the last dataset
    # if the last day is not full it is discarded
    timesteps_in_day = pd.Timedelta(
        1, 'd') / (dataset.index[1] - dataset.index[0])
    if dataset.loc[end_datetime.date():end_datetime].shape[0] < timesteps_in_day:
        end_datetime = end_datetime.date()
    else:
        end_datetime = end_datetime.date() + pd.Timedelta(1, 'd')

    datasets = []
    if (split_method == "ratios"):
        assert split_ratios is not None, f"split_ratios must be specified if split_method is {
            split_method}"
        assert isclose(sum(split_ratios),
                       1.), "The sum of split_ratios must be 1.0"
        last_index = 0
        for split_ratio in split_ratios:
            next_index = last_index + \
                int(np.floor(len(dataset.index) * split_ratio) + 1)
            datasets.append(dataset.loc[dataset.index[last_index: next_index],
                                        used_columns])
            last_index = next_index
    else:
        assert split_dates is not None, f"split_dates must be specified if split_method is {
            split_method}"
        for (start_index, end_index) in zip([start_date] + split_dates, split_dates + [end_datetime]):
            datasets.append(
                dataset.loc[start_index: end_index - pd.tseries.offsets.Minute(1), used_columns])

    return datasets


def normalize_datasets(datasets: list[pd.DataFrame], target_column_name: str | None = None,
                       columns_to_normalize: list[str] | None = None) -> tuple[list[pd.DataFrame], object, object]:
    """Normalizes the datasets.

    Considers the first dataset as the training dataset and uses its scaler to normalize the other datasets.

    Args:
        datasets (list[pd.DataFrame]): List of datasets to normalize.
        target_column_name (str | None, optional): Name of the target column. Defaults to None.
        columns_to_normalize (list[str] | None, optional): List of columns to normalize. If None, all the columns are normalized. Defaults to None.

    Returns:    
        tuple[list[pd.DataFrame], object, object]: Tuple containing the normalized datasets, the scaler, and the target scaler.
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn_pandas import DataFrameMapper
    from sklearn_pandas import gen_features

    if columns_to_normalize is None:
        columns_to_normalize = datasets[0].columns.tolist()

    # this is needed because using a list of lists passes a two-dimensional array to
    # the transformer, which is what is expected by MinMaxScaler
    columns_list_list = [[col] for col in columns_to_normalize]

    feature_def = gen_features(columns=columns_list_list, classes=[
                               {'class': MinMaxScaler, 'feature_range': (-1, 1)}])
    # feature_def.append((['weekend'], [OneHotEncoder(drop='if_binary')], {}))
    scaler = DataFrameMapper(feature_def, input_df=True,
                             df_out=True, default=None)  # type: ignore

    scaler.fit(datasets[0])
    if target_column_name is not None:
        target_scaler = MinMaxScaler(
            feature_range=(-1, 1)).fit(datasets[0].loc[:, [target_column_name]])
    else:
        target_scaler = None

    normalized_datasets = []
    for dataset in datasets:
        normalized_datasets.append(scaler.transform(dataset))

    return normalized_datasets, scaler, target_scaler
