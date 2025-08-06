import numpy as np
import pandas as pd
import pytest

import lstm_pv_forecasting.util.windowed_dataset as wd


def create_data():
    data = {
        'feat1': np.arange(0, 10),
        'feat2': np.arange(100, 110),
        'ex_past': np.arange(50, 60),
        'ex_future': np.arange(150, 160)
    }
    data = pd.DataFrame.from_dict(data)
    return data


def test_make_windowed_dataset_no_exogenous():
    data = create_data()

    input_window_size = 3
    output_window_size = 2
    batch_size = 1
    label_columns = ['feat1', 'feat2']
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, label_columns=label_columns, shuffle=False)

    input_el, target_el = next(dataset.as_numpy_iterator())
    assert input_el.shape == (
        batch_size, input_window_size, len(label_columns))
    np.testing.assert_array_equal(input_el, [[[0, 100],
                                             [1, 101],
                                             [2, 102]]])
    assert target_el.shape == (
        batch_size, output_window_size, len(label_columns))
    np.testing.assert_array_equal(target_el, [[[3, 103],
                                               [4, 104]]])

    batch_size = 2
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, shuffle=False)

    input_el, target_el = next(dataset.as_numpy_iterator())
    assert input_el.shape == (batch_size, input_window_size, len(data.columns))
    assert target_el.shape == (
        batch_size, output_window_size, len(data.columns))


def test_make_windowed_dataset_with_exogenous():
    data = create_data()

    input_window_size = 3
    output_window_size = 2
    batch_size = 2
    label_columns = ['feat1', 'feat2']
    exogenous_columns = ['ex_past']
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, label_columns=label_columns, shuffle=False,
        exogenous_columns=exogenous_columns)

    (input_el, past_exogenous, future_exogenous), target_el = next(
        dataset.as_numpy_iterator())  # type: ignore
    assert input_el.shape == (batch_size, input_window_size, len(
        label_columns))
    assert past_exogenous.shape == (
        batch_size, input_window_size, len(exogenous_columns))
    assert future_exogenous.shape == (
        batch_size, output_window_size, len(exogenous_columns))

    batch_size = 1
    exogenous_columns = (['ex_past', 'ex_future'], ['ex_future'])
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, label_columns=label_columns, shuffle=False,
        exogenous_columns=exogenous_columns)

    (input_el, past_exogenous, future_exogenous), target_el = next(
        dataset.as_numpy_iterator())  # type: ignore
    assert input_el.shape == (batch_size, input_window_size, len(
        label_columns))
    assert past_exogenous.shape == (
        batch_size, input_window_size, len(exogenous_columns[0]))
    assert future_exogenous.shape == (
        batch_size, output_window_size, len(exogenous_columns[1]))

    np.testing.assert_array_equal(input_el, [[[0, 100],
                                             [1, 101],
                                             [2, 102]]])
    np.testing.assert_array_equal(past_exogenous, [[[50, 150],
                                                    [51, 151],
                                                    [52, 152]]])
    np.testing.assert_array_equal(future_exogenous, [[[153],
                                                      [154]]])


def test_make_windowed_dataset_with_next_period_stats():
    data = create_data()

    input_window_size = 3
    output_window_size = 2
    batch_size = 2

    # dataframe with two columns and a number of rows equal to the integer division of the number of rows of data and the output_window_size
    period_stats = pd.DataFrame({
        'stat1': np.arange(0, len(data.index) // output_window_size),
        'stat2': np.arange(0, len(data.index) // output_window_size)
    })

    label_columns = ['feat1', 'feat2']
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, label_columns=label_columns, shuffle=False,
        shift=output_window_size,
        next_period_stats=period_stats)

    (input_el, next_period_stats), target_el = next(
        dataset.as_numpy_iterator())  # type: ignore
    assert next_period_stats.shape == (
        batch_size, len(period_stats.columns))

    np.testing.assert_array_equal(next_period_stats, [[1, 1], [2, 2]])

    assert input_el.shape == (batch_size, input_window_size, len(
        label_columns))
    np.testing.assert_array_equal(input_el, [[[0, 100],
                                             [1, 101],
                                             [2, 102]],
                                             [[2, 102],
                                             [3, 103],
                                             [4, 104]]])


def test_make_windowed_dataset_with_next_period_stats_and_exogenous():
    data = create_data()

    input_window_size = 3
    output_window_size = 2
    batch_size = 2

    # dataframe with two columns and a number of rows equal to the integer division of the number of rows of data and the output_window_size
    period_stats = pd.DataFrame({
        'stat1': np.arange(0, len(data.index) // output_window_size),
        'stat2': np.arange(0, len(data.index) // output_window_size)
    })

    label_columns = ['feat1', 'feat2']
    exogenous_columns = (['ex_past', 'ex_future'], ['ex_future'])
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, label_columns=label_columns, shuffle=False,
        exogenous_columns=exogenous_columns, next_period_stats=period_stats,
        shift=output_window_size)

    (input_el, past_exogenous, future_exogenous, next_period_stats), target_el = next(
        dataset.as_numpy_iterator())  # type: ignore
    assert next_period_stats.shape == (
        batch_size, len(period_stats.columns))

    np.testing.assert_array_equal(next_period_stats, [[1, 1], [2, 2]])

    assert input_el.shape == (batch_size, input_window_size, len(
        label_columns))
    assert past_exogenous.shape == (
        batch_size, input_window_size, len(exogenous_columns[0]))
    assert future_exogenous.shape == (
        batch_size, output_window_size, len(exogenous_columns[1]))

    # check first batch of input and exogenous variables
    np.testing.assert_array_equal(input_el, [[[0, 100],
                                             [1, 101],
                                             [2, 102]],
                                             [[2, 102],
                                             [3, 103],
                                             [4, 104]]])
    np.testing.assert_array_equal(past_exogenous, [[[50, 150],
                                                    [51, 151],
                                                    [52, 152]],
                                                   [[52, 152],
                                                    [53, 153],
                                                    [54, 154]]])
    np.testing.assert_array_equal(future_exogenous, [[[153],
                                                      [154]],
                                                     [[155],
                                                      [156]]])


def test_make_windowed_dataset_throws_with_wrong_names():
    data = create_data()

    input_window_size = 3
    output_window_size = 2
    batch_size = 2

    with pytest.raises(AssertionError):
        wd.make_windowed_dataset(
            data, input_window_size=input_window_size, output_window_size=output_window_size,
            batch_size=batch_size, label_columns=['missing'], shuffle=False,
            exogenous_columns=None)


def test_make_windowed_dataset_throws_with_incompatible_columns():
    data = create_data()

    input_window_size = 3
    output_window_size = 2
    batch_size = 2

    with pytest.raises(ValueError):
        wd.make_windowed_dataset(
            data, input_window_size=input_window_size, output_window_size=output_window_size,
            batch_size=batch_size, label_columns=None, shuffle=False,
            exogenous_columns=['ex_past'])

    with pytest.raises(ValueError):
        wd.make_windowed_dataset(
            data, input_window_size=input_window_size, output_window_size=output_window_size,
            batch_size=batch_size, label_columns=None, shuffle=False,
            exogenous_columns=(['ex_past'], ['ex_future']))


def test_make_windowed_dataset_supports_last_batch_different_size():
    data = create_data()

    input_window_size = 3
    output_window_size = 2

    batch_size = 4
    # expect two full batches of 4 windows and one batch of 2 windows

    label_columns = ['feat1', 'feat2']
    dataset = wd.make_windowed_dataset(
        data, input_window_size=input_window_size, output_window_size=output_window_size,
        batch_size=batch_size, label_columns=label_columns, shuffle=False)

    input_el, target_el = next(dataset.as_numpy_iterator())
    assert input_el.shape == (
        batch_size, input_window_size, len(label_columns))

    assert target_el.shape == (
        batch_size, output_window_size, len(label_columns))

    values = list(dataset.as_numpy_iterator())
    assert values[0][0].shape != values[-1][0].shape
    assert values[0][1].shape != values[-1][1].shape
