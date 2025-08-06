from typing import List, Union

import keras
import numpy as np
import pandas as pd


@keras.saving.register_keras_serializable(package="lstm_pv_forecasting.model")
class ForwardStepLayer(keras.layers.Layer):
    def __init__(self, units: Union[int, List[int]], delay: int, out_steps: int,
                 num_features: int, exogenous: bool = False,
                 output_init_bias: np.ndarray | None = None,
                 activation: str | None = 'relu',
                 kernel_initializer: str | keras.initializers.Initializer | None = 'glorot_uniform',
                 bias_initializer: str | keras.initializers.Initializer | None = 'zeros',
                 kernel_regularizer: str | keras.regularizers.Regularizer | None = None,
                 bias_regularizer: str | keras.regularizers.Regularizer | None = None,
                 **kwargs):
        """Feed forward layer called recursively by Narx model. 

        Args:
            units (Union[int, List[int]]): number of units in hidden layer or list of number of units for each hidden layer
            delay (int): number of past time steps considered in input
            out_steps (int): number of time steps to predict in the future
            num_features (int): number of features of the time series
            exogenous (bool, optional): whether to consider exogenous features or not. Defaults to False.
            output_init_bias: (np.ndarray, optional): constant init value for the output layer bias. Defaults to None.
            activation (str, optional): activation function for the hidden layers. Defaults to 'relu'.
            kernel_initializer (str | keras.initializers.Initializer, optional): initializer for the kernel weights. Defaults to 'glorot_uniform'.
            bias_initializer (str, optional): initializer for the bias. Defaults to 'zeros'.
            kernel_regularizer (str | keras.regularizers.Regularizer, optional): regularizer for the kernel weights. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
            bias_regularizer (str| keras.regularizers.Regularizer, optional): regularizer for the bias. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
        """
        super().__init__(**kwargs)
        self.units = units
        self.delay = delay
        self.exogenous = exogenous
        self.output_init_bias = output_init_bias
        self.num_features = num_features
        self.out_steps = out_steps
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.get_delay_time_steps = keras.layers.Lambda(
            lambda x: x[:, -self.delay:, :])
        self.flatten = keras.layers.Flatten()
        if isinstance(units, int):
            units = [units]
        assert isinstance(
            units, list), 'units must be eighter an int or a list of int'
        self.dense_layers = []
        for num_units in units:
            self.dense_layers.append(keras.layers.Dense(num_units, activation=activation,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=bias_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        bias_regularizer=self.bias_regularizer))
        self.dense_out = keras.layers.Dense(
            self.out_steps * self.num_features)

        if self.output_init_bias is not None:
            self.dense_out.bias_initializer = keras.initializers.Constant(
                value=np.tile(self.output_init_bias, self.out_steps))

        self.reshape = keras.layers.Reshape(
            (self.out_steps, self.num_features))

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'delay': self.delay,
            'out_steps': self.out_steps,
            'num_features': self.num_features,
            'exogenous': self.exogenous,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
        })
        if self.output_init_bias is not None:
            config.update({
                'output_init_bias': self.output_init_bias.tolist()
            })
        else:
            config.update({
                'output_init_bias': None
            })
        return config

    @classmethod
    def from_config(cls, config):
        if config["output_init_bias"] is not None:
            config["output_init_bias"] = np.array(config["output_init_bias"])
        return cls(**config)

    def call(self, inputs):
        if self.exogenous:
            features, past_exogenous, future_exogenous = inputs
            x = keras.ops.concatenate([features, past_exogenous], axis=2)
        else:
            x = inputs
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, delay, features)
        x = self.get_delay_time_steps(x)
        # x.shape => (batch, delay * features)
        x = self.flatten(x)
        if self.exogenous:
            x = keras.ops.concatenate(
                [x, self.flatten(future_exogenous)], axis=1)
        # x.shape => (batch, units)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        # predictions.shape => (batch, features)
        prediction = self.dense_out(x)
        # prediction.shape => (batch, out_steps, features)
        prediction = self.reshape(prediction)
        return prediction


@keras.saving.register_keras_serializable(package="lstm_pv_forecasting.model")
class NARX(keras.Model):
    '''NARX model.'''

    def __init__(self, units: Union[int, List[int]], delay: int, out_steps: int, num_features: int,
                 exogenous: bool = False, output_init_bias: np.ndarray | None = None,
                 one_shot_steps: int = 1, open_loop_training_mode: bool = False,
                 activation: str | None = 'relu',
                 kernel_initializer: str | keras.initializers.Initializer | None = 'glorot_uniform',
                 bias_initializer: str | keras.initializers.Initializer | None = 'zeros',
                 kernel_regularizer: str | dict | None = None,
                 bias_regularizer: str | dict | None = None, **kwargs):
        """Create NARX model

        Args:
            units (Union[int, List[int]]): Number of units in the hidden layers or list of number of units for each hidden unit
            delay (int): number of past time steps considered in input
            out_steps (int): number of time steps to predict in the future. Note: if exogenous is true, this is ignored and the
              number of time steps to predict is given by the shape of the future exogenous variables.
            num_features (int): number of features of the time series
            exogenous (bool, optional): whether to consider exogenous features or not. Defaults to False.
            output_init_bias: (np.ndarray, optional): constant init value for the output layer bias. Defaults to None.
            one_shot_steps (int, optional): number of time steps to predict in a single shot before recurrence. Defaults to 1.
            open_loop_training_mode (bool, optional): whether to train in open loop mode or not. Defaults to False.
            activation (str, optional): activation function for the hidden layers. Defaults to 'relu'.
            kernel_initializer (str, optional): initializer for the kernel weights. Defaults to 'glorot_uniform'.
            bias_initializer (str, optional): initializer for the bias. Defaults to 'zeros'.
            kernel_regularizer (str, dict, optional): regularizer for the kernel weights. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
            bias_regularizer (str, dict, optional): regularizer for the bias. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
        """
        super().__init__(**kwargs)
        self.out_steps = out_steps
        self.one_shot_steps = one_shot_steps
        self.units = units
        self.delay = delay
        self.num_features = num_features
        self.exogenous = exogenous
        self.output_init_bias = output_init_bias

        # dense layers regularizers
        if isinstance(kernel_regularizer, dict):
            self.kernel_regularizer = keras.regularizers.get(
                kernel_regularizer)
        else:
            self.kernel_regularizer = kernel_regularizer
        if isinstance(bias_regularizer, dict):
            self.bias_regularizer = keras.regularizers.get(
                bias_regularizer)
        else:
            self.bias_regularizer = bias_regularizer

        self.step_layer = ForwardStepLayer(units=self.units, delay=self.delay,
                                           out_steps=self.one_shot_steps,
                                           num_features=self.num_features,
                                           exogenous=self.exogenous,
                                           output_init_bias=self.output_init_bias,
                                           activation=activation,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=self.kernel_regularizer,
                                           bias_regularizer=self.bias_regularizer)

        self.shift = keras.layers.Lambda(
            lambda x: x[:, self.one_shot_steps:, :])
        if self.exogenous:
            self.next_exogenous = keras.layers.Lambda(
                lambda x: x[:, :self.one_shot_steps, :])
        self.concat = keras.layers.Concatenate(axis=1)
        self.open_loop_training_mode = open_loop_training_mode

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'delay': self.delay,
            'out_steps': self.out_steps,
            'num_features': self.num_features,
            'exogenous': self.exogenous,
            'one_shot_steps': self.one_shot_steps,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        if self.output_init_bias is not None:
            config.update({
                'output_init_bias': self.output_init_bias.tolist()
            })
        else:
            config.update({
                'output_init_bias': None
            })
        return config

    @classmethod
    def from_config(cls, config):
        if config["output_init_bias"] is not None:
            config["output_init_bias"] = np.array(config["output_init_bias"])
        return cls(**config)

    def call(self, inputs, training=False):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        x = inputs
        if self.exogenous:
            features, past_exogenous, future_exogenous = inputs
            out_steps = future_exogenous.shape[1]
            next_exogenous = self.next_exogenous(future_exogenous)
            future_exogenous = self.shift(future_exogenous)
            x = features, past_exogenous, next_exogenous
        else:
            out_steps = self.out_steps

        # first step
        prediction = self.step_layer(x)
        # Insert the first prediction
        predictions.append(prediction)

        # if training mode is open loop, train only on one shot,
        # no recursion
        # otherwise execute recursively for training and prediction
        if not self.open_loop_training_mode or not training:
            # Run the rest of the prediction steps
            for n in range(self.one_shot_steps, out_steps,
                           self.one_shot_steps):
                # Concat last prediction to input.
                if not self.exogenous:
                    x = self.concat([self.shift(x), prediction])
                else:
                    features = self.concat([self.shift(features),
                                           prediction])
                    past_exogenous = self.concat([self.shift(past_exogenous),
                                                 next_exogenous])
                    next_exogenous = self.next_exogenous(future_exogenous)
                    future_exogenous = self.shift(future_exogenous)
                    x = (features, past_exogenous, next_exogenous)

                prediction = self.step_layer(x)
                # Add the prediction to the output
                predictions.append(prediction)

        # predictions.shape => (batch, time, features)
        predictions = keras.ops.concatenate(predictions, axis=1)
        return predictions
