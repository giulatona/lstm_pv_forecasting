from asyncio import futures

import numpy as np

from lstm_pv_forecasting.model.deep_next_day_stat import DeepNextDayStat


def test_deep_next_day_stat():

    out_steps = 12
    batch_size = 10
    num_features = 1
    input_window_size = 24
    lstm_units = [32, 32]
    dense_units = [32, 32]
    intermediate_units = [64, 32]

    model = DeepNextDayStat(lstm_units=lstm_units, dense_units=dense_units,
                            out_steps=out_steps, num_features=num_features,
                            intermediate_units=intermediate_units,
                            intermediate_kernel_regularizer="l1",
                            intermediate_bias_regularizer="l1",)

    # Create sample inputs of shape (batch, num_timesteps, num_features)
    inputs = np.random.rand(
        batch_size, input_window_size, num_features)
    past_exogenous = np.random.rand(
        batch_size, input_window_size, 6)
    futures_exogenous = np.random.rand(batch_size, out_steps, 4)
    next_day_statistics = np.random.rand(batch_size, 2)

    # Call the model
    outputs = model.call(
        (inputs, past_exogenous, futures_exogenous, next_day_statistics))

    # Check the shape of the outputs
    assert outputs.shape == (batch_size, out_steps, num_features)


def test_deep_next_day_stat_bidirectional():
    out_steps = 12
    batch_size = 10
    num_features = 1
    input_window_size = 24
    lstm_units = [32, 32]
    dense_units = [32, 32]
    intermediate_units = [64, 32]

    model = DeepNextDayStat(lstm_units=lstm_units, dense_units=dense_units,
                            out_steps=out_steps, num_features=num_features,
                            intermediate_units=intermediate_units,
                            use_bidirectional=True,  # Enable bidirectional
                            intermediate_kernel_regularizer="l1",
                            intermediate_bias_regularizer="l1",)

    # Create sample inputs of shape (batch, num_timesteps, num_features)
    inputs = np.random.rand(
        batch_size, input_window_size, num_features)
    past_exogenous = np.random.rand(
        batch_size, input_window_size, 6)
    futures_exogenous = np.random.rand(batch_size, out_steps, 4)
    next_day_statistics = np.random.rand(batch_size, 2)

    # Call the model
    outputs = model.call(
        (inputs, past_exogenous, futures_exogenous, next_day_statistics))

    # Check the shape of the outputs
    assert outputs.shape == (batch_size, out_steps, num_features)
