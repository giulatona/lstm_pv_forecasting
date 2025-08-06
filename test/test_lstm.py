import keras
import numpy as np
import numpy.testing as npt
import pytest

from lstm_pv_forecasting.model.lstm import Lstm


def test_lstm_regularizer_class():

    lstm_units = [32, 32]
    dense_units = [32, 32]
    out_steps = 12
    num_features = 1

    model = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                 out_steps=out_steps, num_features=num_features,
                 dense_kernel_regularizer='l1', dense_bias_regularizer='l2')

    kernel_regularizer = keras.regularizers.l1(l1=0.01)

    model1 = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                  out_steps=out_steps, num_features=num_features,
                  dense_kernel_regularizer=kernel_regularizer, dense_bias_regularizer='l2')

    assert isinstance(model1.dense_kernel_regularizer, keras.regularizers.L1)
    npt.assert_array_equal(model1.dense_kernel_regularizer.l1,
                           np.array(0.01, dtype='float32'))

    kernel_regularizer = {
        "module": "keras.regularizers",
        "class_name": "L1",
        "config":
            {"l1": 0.01
             }
    }

    model2 = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                  out_steps=out_steps, num_features=num_features,
                  dense_kernel_regularizer=kernel_regularizer, dense_bias_regularizer='l2')

    assert isinstance(model2.dense_kernel_regularizer, keras.regularizers.L1)
    npt.assert_array_equal(model2.dense_kernel_regularizer.l1,
                           np.array(0.01, dtype='float32'))


def test_dense_output_initializer():
    lstm_units = [32, 32]
    dense_units = [32, 32]
    out_steps = 12
    num_features = 1

    output_init_bias = np.random.rand(1).astype('float32')

    model = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                 out_steps=out_steps, num_features=num_features,
                 output_init_bias=output_init_bias)

    model.compile(run_eagerly=True)

    # call model with random input to build the model
    input_window_size = 10
    input_data = (np.random.rand(1, input_window_size, num_features).astype('float32'),
                  np.random.rand(1, input_window_size,
                                 num_features).astype('float32'),
                  np.random.rand(1, out_steps, num_features).astype('float32'))
    model(input_data)

    npt.assert_array_equal(model.dense_out.bias.numpy(), output_init_bias)

    # in case of more than one features, there should be two biases
    num_features = 2
    output_init_bias = np.random.rand(num_features).astype('float32')

    model = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                 out_steps=out_steps, num_features=num_features,
                 output_init_bias=output_init_bias)

    model.compile(run_eagerly=True)

    # call model with random input to build the model
    input_window_size = 10
    input_data = (np.random.rand(1, input_window_size, num_features).astype('float32'),
                  np.random.rand(1, input_window_size,
                                 num_features).astype('float32'),
                  np.random.rand(1, out_steps, num_features).astype('float32'))

    model(input_data)

    npt.assert_array_equal(model.dense_out.bias.numpy(), output_init_bias)


def test_lstm_bidirectional_option():
    lstm_units = [32, 32]
    dense_units = [32, 32]
    out_steps = 12
    num_features = 1

    # Test without bidirectional option
    model = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                 out_steps=out_steps, num_features=num_features,
                 use_bidirectional=False)

    model.compile(run_eagerly=True)

    input_window_size = 10
    input_data = (np.random.rand(1, input_window_size, num_features).astype('float32'),
                  np.random.rand(1, input_window_size, num_features).astype('float32'),
                  np.random.rand(1, out_steps, num_features).astype('float32'))

    output = model(input_data)
    assert output.shape == (1, out_steps, num_features)

    # Test with bidirectional option
    model = Lstm(lstm_units=lstm_units, dense_units=dense_units,
                 out_steps=out_steps, num_features=num_features,
                 use_bidirectional=True)

    model.compile(run_eagerly=True)

    output = model(input_data)
    assert output.shape == (1, out_steps, num_features)
