from datetime import datetime

import numpy as np
import pandas as pd
from pyparsing import col
import pytest

import lstm_pv_forecasting.util.load as load


def test_load_dataset_split_ratios():
    data_filename = "test/sample_raw_dataset.csv"
    ratios = [0.3, 0.7]
    datasets = load.load_dataset(
        data_filename, split_method="ratios", split_ratios=ratios)
    assert len(datasets) == len(ratios)
    ratios = [0.2, 0.7]
    with pytest.raises(AssertionError) as error:
        datasets = load.load_dataset(
            data_filename, split_method="ratios", split_ratios=ratios)


def test_load_dataset_split_dates():
    data_filename = "test/sample_raw_dataset.csv"
    datetimes = [datetime(2002, 5, 10), datetime(2002, 8, 11)]
    datasets = load.load_dataset(
        data_filename, split_method="datetimes", split_dates=datetimes)
    assert len(datasets) == len(datetimes) + 1
    assert datasets[1].index[0] == datetimes[0]
    assert datasets[0].index[-1] < datasets[1].index[0]


def test_load_dataset_split_dates_last_day_alignment(tmp_path):
    # write temp file with a dataset that has a last day that is not full
    index = pd.date_range(start='2002-01-01',
                          end='2002-12-31 21:00', freq='30min')
    dataset = pd.DataFrame(
        {'radiance': np.random.random((len(index),)),
         'temperature': np.random.random((len(index),))},
        index=index)
    data_filename = tmp_path / "sample_raw_dataset.csv"
    dataset.to_csv(data_filename, index_label='timestamp', columns=['radiance', 'temperature'],
                   sep=',')

    datetimes = [datetime(2002, 5, 10), datetime(2002, 8, 11)]
    datasets = load.load_dataset(
        data_filename, split_method="datetimes", split_dates=datetimes)
    assert len(datasets) == len(datetimes) + 1

    assert datasets[2].index[-1] == pd.Timestamp('2002-12-30 23:30:00')

    # write temp file with a dataset that has a last day that is full
    index = pd.date_range(start='2002-01-01',
                          end='2002-12-31 23:30', freq='30min')
    dataset = pd.DataFrame(
        {'radiance': np.random.random((len(index),)),
         'temperature': np.random.random((len(index),))},
        index=index)
    data_filename = tmp_path / "sample_raw_dataset1.csv"
    dataset.to_csv(data_filename, index_label='timestamp', columns=['radiance', 'temperature'],
                   sep=',')
    datasets = load.load_dataset(
        data_filename, split_method="datetimes", split_dates=datetimes)
    assert len(datasets) == len(datetimes) + 1
    assert datasets[2].index[-1] == pd.Timestamp('2002-12-31 23:30:00')

    # write temp file with a dataset that has only one time step in the last day at midnigth
    index = pd.date_range(start='2002-01-01',
                          end='2002-12-31 00:00', freq='30min')
    dataset = pd.DataFrame(
        {'radiance': np.random.random((len(index),)),
         'temperature': np.random.random((len(index),))},
        index=index)
    data_filename = tmp_path / "sample_raw_dataset2.csv"
    dataset.to_csv(data_filename, index_label='timestamp', columns=['radiance', 'temperature'],
                   sep=',')
    datasets = load.load_dataset(
        data_filename, split_method="datetimes", split_dates=datetimes)
    assert len(datasets) == len(datetimes) + 1
    assert datasets[2].index[-1] == pd.Timestamp('2002-12-30 23:30:00')


def test_normalize_datasets():
    data_filename = "test/sample_raw_dataset.csv"
    datetimes = [datetime(2002, 5, 10), datetime(2002, 8, 11)]
    datasets = load.load_dataset(
        data_filename, split_method="datetimes", split_dates=datetimes)
    normalized_datasets, scaler, target_scaler = load.normalize_datasets(
        datasets, target_column_name='radiance', columns_to_normalize=['radiance', 'temperature'])
    assert len(normalized_datasets) == len(datasets)
    assert scaler is not None
    assert target_scaler is not None
    assert normalized_datasets[0].columns[0] == 'radiance'
    assert normalized_datasets[0].columns[1] == 'temperature'

    assert normalized_datasets[0].loc[:,
                                      'radiance'].max() == pytest.approx(1.0)
    assert normalized_datasets[0].loc[:,
                                      'radiance'].min() == pytest.approx(-1.0)
