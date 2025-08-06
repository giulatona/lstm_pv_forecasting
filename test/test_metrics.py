from statistics import mean

from matplotlib import axis
from matplotlib.pylab import normal
import numpy as np
import pytest

from lstm_pv_forecasting.metrics.metrics import MeanWindowNRMSE
from lstm_pv_forecasting.metrics.metrics import NMAX
from lstm_pv_forecasting.metrics.metrics import NRMSE


def test_nrmse():
    y_true = np.array([0., 0., 1., 1., 0., 0.], dtype=np.float32)
    y_true = np.expand_dims(np.expand_dims(y_true, axis=0), axis=2)
    y_pred = np.array([0., 0., 2., 2., 0., 0.], dtype=np.float32)
    y_pred = np.expand_dims(np.expand_dims(y_pred, axis=0), axis=2)
    nrmse = NRMSE()
    nrmse.update_state(y_true, y_pred)
    result = nrmse.result()
    assert np.isclose(result, np.sqrt(2 / 6) / 1)

    nrmse.reset_state()
    assert nrmse.max_val == -np.inf
    assert nrmse.min_val == np.inf

    y_true = np.array([0., 0., 1., 1., 0., 0.], dtype=np.float32)
    y_true = np.repeat(np.expand_dims(y_true, axis=0), repeats=2, axis=0)
    y_true = np.expand_dims(y_true, axis=2)
    y_pred = np.array([0., 0., 2., 2., 0., 0.], dtype=np.float32)
    y_pred = np.repeat(np.expand_dims(y_pred, axis=0), repeats=2, axis=0)
    y_pred = np.expand_dims(y_pred, axis=2)
    nrmse = NRMSE()
    nrmse.update_state(y_true, y_pred)
    result = nrmse.result()
    assert np.isclose(result, np.sqrt(2 / 6) / 1)


def test_nmax():
    y_true = np.array([0., 0., 1., 1., 0., 0.], dtype=np.float32)
    y_true = np.expand_dims(np.expand_dims(y_true, axis=0), axis=2)
    y_pred = np.array([0., 0., 2., 2., 0., 0.], dtype=np.float32)
    y_pred = np.expand_dims(np.expand_dims(y_pred, axis=0), axis=2)
    nmax = NMAX()
    nmax.update_state(y_true, y_pred)
    result = nmax.result()
    assert np.isclose(result, 1)

    nmax.reset_state()
    assert nmax.max_val == -np.inf
    assert nmax.min_val == np.inf
    assert nmax.max_error == 0.0

    y_true = np.array([0., 0., 1., 1., 0., 0.], dtype=np.float32)
    y_true = np.repeat(np.expand_dims(y_true, axis=0), repeats=2, axis=0)
    y_true = np.expand_dims(y_true, axis=2)
    y_pred = np.array([0., 0., 2., 2., 0., 0.], dtype=np.float32)
    y_pred = np.repeat(np.expand_dims(y_pred, axis=0), repeats=2, axis=0)
    y_pred = np.expand_dims(y_pred, axis=2)
    nmax = NMAX()
    nmax.update_state(y_true, y_pred)
    result = nmax.result().numpy()
    assert np.isclose(result, 1)


def test_mean_window_nrmse_same_as_nrmse_on_one_window():
    y_true = np.array([0., 0., 1., 1., 0., 0.], dtype=np.float32)
    y_true = np.expand_dims(np.expand_dims(y_true, axis=0), axis=2)
    y_pred = np.array([0., 0., 2., 2., 0., 0.], dtype=np.float32)
    y_pred = np.expand_dims(np.expand_dims(y_pred, axis=0), axis=2)

    nrmse = NRMSE()
    nrmse.update_state(y_true, y_pred)
    nrmse_result = nrmse.result()

    mean_window_nrmse = MeanWindowNRMSE()
    mean_window_nrmse.update_state(y_true, y_pred)
    mean_window_nrmse_result = mean_window_nrmse.result()

    assert np.isclose(mean_window_nrmse_result, nrmse_result)


def test_mean_window_nrmse():
    y_true = np.array([[0., 0., 1., 1., 0., 0.],
                       [1.0, 1.0, 0., 0., 1., 1.]], dtype=np.float32)
    y_true = y_true.transpose()
    y_true = np.expand_dims(y_true, axis=2)
    assert y_true.shape == (6, 2, 1)
    y_pred = np.array([[0., 0., 2., 2., 0., 0.],
                       [2., 2., 0., 0., 2., 2.]], dtype=np.float32)
    y_pred = y_pred.transpose()
    y_pred = np.expand_dims(y_pred, axis=2)

    mean_window_nrmse = MeanWindowNRMSE()
    mean_window_nrmse.update_state(y_true, y_pred)
    mean_window_nrmse_result = mean_window_nrmse.result()

    assert np.isclose(mean_window_nrmse_result, 1 / np.sqrt(2))

    # Test with a specified normalization factor
    normalization_factor = 3.0
    mean_window_nrmse = MeanWindowNRMSE(
        normalization_factor=normalization_factor)
    mean_window_nrmse.update_state(y_true, y_pred)
    mean_window_nrmse_result = mean_window_nrmse.result()

    assert np.isclose(mean_window_nrmse_result, 1 /
                      np.sqrt(2) / normalization_factor)
