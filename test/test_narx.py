import keras
import numpy as np
import numpy.testing as npt
import pytest

from lstm_pv_forecasting.model.narx import NARX


def test_narx_regularizer_class():

    units = [32, 32]
    delay = 6
    out_steps = 12
    num_features = 1

    model = NARX(units, delay, out_steps, num_features,
                 kernel_regularizer='l1', bias_regularizer='l2')

    kernel_regularizer = keras.regularizers.l1(l1=0.01)
    model1 = NARX(units, delay, out_steps, num_features,
                  kernel_regularizer=kernel_regularizer, bias_regularizer='l2')

    assert isinstance(model1.kernel_regularizer, keras.regularizers.L1)
    npt.assert_array_equal(model1.kernel_regularizer.l1,
                           np.array(0.01, dtype='float32'))

    kernel_regularizer = {
        "module": "keras.regularizers",
        "class_name": "L1",
        "config":
            {"l1": 0.01
             }
    }

    kernel_regularizer = keras.regularizers.l1(l1=0.01)
    model1 = NARX(units, delay, out_steps, num_features,
                  kernel_regularizer=kernel_regularizer, bias_regularizer='l2')

    assert isinstance(model1.kernel_regularizer, keras.regularizers.L1)
    npt.assert_array_equal(model1.kernel_regularizer.l1,
                           np.array(0.01, dtype='float32'))


def test_dense_output_initializer():
    dense_units = [32, 32]
    delay = 10
    out_steps = 12
    one_shot_steps = 1
    num_features = 1

    output_init_bias = np.random.rand(1).astype('float32')

    model = NARX(units=dense_units, delay=delay,
                 out_steps=out_steps, num_features=num_features,
                 output_init_bias=output_init_bias,
                 one_shot_steps=one_shot_steps)

    model.compile(run_eagerly=True)

    # call model with random input to build the model
    input_window_size = 10
    input_data = np.random.rand(
        1, input_window_size, num_features).astype('float32')
    model(input_data)

    assert model.step_layer.dense_out.bias.shape == (
        one_shot_steps * num_features,)
    npt.assert_array_equal(model.step_layer.dense_out.bias.numpy(),
                           np.tile(output_init_bias, one_shot_steps))

    # in case of more than one features, there should be two biases
    num_features = 2
    output_init_bias = np.random.rand(num_features).astype('float32')

    model = NARX(units=dense_units, delay=delay,
                 out_steps=out_steps, num_features=num_features,
                 output_init_bias=output_init_bias)

    model.compile(run_eagerly=True)

    # call model with random input to build the model
    input_window_size = 10
    input_data = np.random.rand(
        1, input_window_size, num_features).astype('float32')

    model(input_data)

    assert model.step_layer.dense_out.bias.shape == (
        one_shot_steps * num_features,)
    npt.assert_array_equal(model.step_layer.dense_out.bias.numpy(),
                           np.tile(output_init_bias, one_shot_steps))

    # when exogenous inputs are present
    model = NARX(units=dense_units, delay=delay,
                 out_steps=out_steps, num_features=num_features,
                 exogenous=True,
                 output_init_bias=output_init_bias,
                 one_shot_steps=one_shot_steps)

    model.compile(run_eagerly=True)

    # call model with random input to build the model
    input_window_size = 10
    input_data = (np.random.rand(1, input_window_size, num_features).astype('float32'),
                  np.random.rand(1, input_window_size,
                                 num_features).astype('float32'),
                  np.random.rand(1, out_steps, num_features).astype('float32'))
    model(input_data)

    assert model.step_layer.dense_out.bias.shape == (
        one_shot_steps * num_features,)
    npt.assert_array_equal(model.step_layer.dense_out.bias.numpy(),
                           np.tile(output_init_bias, one_shot_steps))
