import numpy as np
import pytest

from lstm_pv_forecasting.model import Naive


def test_naive_forward():
    out_steps = 4
    exogenous = False
    model = Naive(out_steps=out_steps, exogenous=exogenous)
    model.compile()

    x = np.arange(10)
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)
    prediction = model(x)
    np.testing.assert_array_equal(prediction, x[:, -out_steps:, :])

    future_exogenous = np.arange(out_steps)
    future_exogenous = np.reshape(
        future_exogenous, [1, len(future_exogenous), 1])
    x = (x, x, future_exogenous)
    exogenous = True
    model = Naive(out_steps=out_steps, exogenous=exogenous)
    prediction = model(x)
    np.testing.assert_array_equal(prediction, x[0][:, -out_steps:, :])
