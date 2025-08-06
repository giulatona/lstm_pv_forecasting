import keras
import numpy as np
import pandas as pd

from lstm_pv_forecasting.model.lstm import Lstm


@keras.saving.register_keras_serializable(package="lstm_pv_forecasting.model")
class DeepNextDayStat(Lstm):
    """DeepNextDayStat model.

    This model is an LSTM encoder-decoder architecture that also includes
    next day statistics like clearness index and mean temperature (obtained from an external
    forecasting service) and 
    concanenates them to the result of the encoder layers before the decoder.
    """

    def __init__(self, lstm_units: int | list[int], dense_units: int | list[int],
                 out_steps: int, num_features: int,
                 intermediate_units: int | list[int],
                 intermediate_kernel_regularizer: str | dict[str,
                                                             object] | None = None,
                 intermediate_bias_regularizer: str | dict[str,
                                                           object] | None = None,
                 *args, **kwargs):
        """ Create DeepNextDayStat model.

        Args:
            lstm_units (Union[int, List[int]]): number of units in encoder and decoder LSTMs, or list of number of units per LSTM encoder and decoder layers
            dense_units (Union[int, List[int]]): Number of units in hidden layer, or list of number of units per hidden layer, of fully connected part
            out_steps (int): number of steps to forecast
            num_features (int): number of feautures of the time series
            intermediate_units (Union[int, List[int]]): Number of units, or list of number of units per intermediate FC layer combining states and next day statistics
            intermediate_kernel_regularizer (str | dict, optional): kernel regularizer of intermediate FC layers. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
            intermediate_bias_regularizer (str | dict, optional): bias regularizer of intermediate FC layers. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
            *args: positional arguments for Lstm model
            **kwargs: keyword arguments for Lstm model
        """
        super().__init__(lstm_units, dense_units, out_steps, num_features,
                         *args, **kwargs)
        if isinstance(intermediate_units, int):
            intermediate_units = [intermediate_units]
        self.intermediate_units = intermediate_units

        # dense layers regularizers
        if isinstance(intermediate_kernel_regularizer, dict):
            self.intermediate_kernel_regularizer = keras.regularizers.get(
                intermediate_kernel_regularizer)
        else:
            self.intermediate_kernel_regularizer = intermediate_kernel_regularizer
        if isinstance(intermediate_bias_regularizer, dict):
            self.intermediate_bias_regularizer = keras.regularizers.get(
                intermediate_bias_regularizer)
        else:
            self.intermediate_bias_regularizer = intermediate_bias_regularizer

        self.intermediate_fc_layers = []
        for layer_count, num_units in enumerate(self.lstm_units):
            num_output_intermediate_fc_units = num_units * \
                2 if not self.use_bidirectional else num_units * 4
            self.intermediate_fc_layers.append(keras.Sequential([
                keras.layers.Dense(units, activation='relu',
                                   kernel_regularizer=self.intermediate_kernel_regularizer,
                                   bias_regularizer=self.intermediate_bias_regularizer)
                for units in (self.intermediate_units + [num_output_intermediate_fc_units])
            ],
                name=f"intermediate_fc_{layer_count}"))

    def get_config(self):
        config = super().get_config()
        config.update({
            'intermediate_units': self.intermediate_units,
            'intermediate_kernel_regularizer': keras.regularizers.serialize(self.intermediate_kernel_regularizer),
            'intermediate_bias_regularizer': keras.regularizers.serialize(self.intermediate_bias_regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=False):
        """Forward pass of the model.

        Args:
            inputs: input tuple containing (past_observations, past_exogenous, future_exogenous, next_day_statistics)
            training: boolean, whether the model is in training mode

        Returns:
            tensor of next day forecasts
        """
        # features.shape => (batch, num_timesteps, num_features)
        # past_exogenous.shape => (batch, num_timesteps, num_past_exogenous_features)
        # future_exogenous.shape => (batch, num_timesteps, num_future_exogenous_features)
        # next_day_statistics.shape => (batch, num_statistics)
        features, past_exogenous, future_exogenous, next_day_statistics = inputs

        # encoder

        # x.shape => (batch, num_timesteps,
        #             num_features + num_exogenous_features)
        x = keras.ops.concatenate([features, past_exogenous], axis=2)

        states = []
        for lstm_encoder in self.lstm_encoder_layers:
            # x.shape => (batch, num_time_steps, lstm_units)
            # h.shape => (batch, lstm_units)
            # h.shape => (batch, lstm_units)
            if not self.use_bidirectional:
                x, h, c = lstm_encoder(x)
                states.append([h, c])
            else:
                # hf, cf = forward_states
                # hb, cb = backward_states
                x, hf, cf, hb, cb = lstm_encoder(x)
                states.append([hf, cf, hb, cb])

        if not self.use_bidirectional:
            _, h, c = self.last_lstm_encoder(x)  # type: ignore
            states.append([h, c])
        else:
            # hf, cf = forward_states
            # hb, cb = backward_states
            _, hf, cf, hb, cb = self.last_lstm_encoder(x)
            states.append([hf, cf, hb, cb])

        x = future_exogenous
        for lstm_decoder, init_state, intermediate_fc in zip(self.lstm_decoder_layers, states,
                                                             self.intermediate_fc_layers):
            # concatenate the states and next day statistics
            # init_state.shape => (batch, lstm_units * 2) or (batch, lstm_units * 4)
            a1 = init_state + [next_day_statistics]
            a2 = keras.ops.concatenate(a1, axis=1)
            init_state = intermediate_fc(a2)

            # get h and c states from the intermediate layer
            split_num = 2 if not self.use_bidirectional else 4
            init_state = keras.ops.split(init_state, split_num, axis=1)

            # x.shape => (batch, out_steps, lstm_units)
            x = lstm_decoder(x, initial_state=init_state)

        for dense_layer in self.dense_layers:
            # x.shape => (batch, out_steps, dense_layer.units)
            x = dense_layer(x)

        # x.shape => (batch, out_steps, num_features)
        x = self.dense_out(x)

        return x
