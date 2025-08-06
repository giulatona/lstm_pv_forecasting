import datetime
from operator import index

import numpy as np
import pandas as pd
import pytest

from lstm_pv_forecasting.util.period_stats import calculate_period_stats


def test_calc_period_stats():

    num_days = 30
    # Create a sample dataset

    timestamp = pd.date_range(
        start='2022-01-01', end='2022-01-30 23:00', freq='1h')
    temperature = np.random.rand(len(timestamp))
    radiance = np.array([0., 0., 0., 0., 0., 0., 0., 0.01, 0.07, 0.22, 0.85,
                         0.79, 1.11, 1.01, 0.93, 0.76, 0.08, 0., 0., 0., 0., 0.,
                         0., 0.])  # first day of SIAS dataset
    radiance = np.tile(radiance, num_days)
    dataset = pd.DataFrame({
        'temperature': temperature,
        'radiance': radiance  # it is actually the hourly total irradiation in MJ/m^2
    }, index=timestamp)

    # Set the stat_period_duration and first_offset
    stat_period_duration = datetime.timedelta(days=1)
    first_offset = datetime.timedelta(days=1)

    latitude = 38  # latitude of Palermo in degrees

    # Call the function
    result = calculate_period_stats(
        dataset, stat_period_duration, latitude)

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the result has the correct size
    assert result.shape[0] == num_days

    # Check if the result has the correct columns
    assert result.columns.tolist() == ['mean_temp', 'Kt']

    # check that the mean of mean_temp is equal to the mean of temperature
    assert np.isclose(result['mean_temp'].mean(),
                      temperature.mean())

    # check that the first Kt is the same as calculated by Matlab code
    assert np.isclose(result['Kt'].iloc[0], 0.387339685215413)

    # check that the index is a DatetimeIndex and the first date is correct
    assert result.index[0] == pd.Timestamp('2022-01-01')
