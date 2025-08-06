import keras
import numpy as np
import pandas as pd


@keras.saving.register_keras_serializable(package="lstm_pv_forecasting.model")
class Lstm(keras.Model):
    '''Recurrent LSTM encoder-decoder model.'''

    def __init__(self, lstm_units: int | list[int], dense_units: int | list[int],
                 out_steps: int, num_features: int,
                 use_bidirectional: bool = False,  # New parameter
                 dropout: float = 0.0,
                 output_activation_function: str = "linear",
                 output_init_bias: np.ndarray | None = None,
                 output_kernel_initializer: str | keras.initializers.Initializer | None = 'glorot_uniform',
                 dense_activation: str | None = 'relu',
                 dense_kernel_initializer: str | keras.initializers.Initializer | None = 'glorot_uniform',
                 dense_bias_initializer: str | keras.initializers.Initializer | None = 'zeros',
                 dense_kernel_regularizer: str | dict[str,
                                                      object] | None = None,
                 dense_bias_regularizer: str | dict[str,
                                                    object] | None = None,
                 **kwargs):
        """Create Lstm model. Exogenous variables are not optional.

        Args:
            lstm_units (Union[int, List[int]]): number of units in encoder and decoder LSTMs, or list of number of units per LSTM encoder and decoder layers
            dense_units (Union[int, List[int]]): Number of units in hidden layer, or list of number of units per hidden layer, of fully connected part 
            out_steps (int): number of steps to forecast
            num_features (int): number of feautures of the time series
            use_bidirectional (bool, optional): whether to use Bidirectional layers. Defaults to False.
            dropout (float, optional): dropout value used in the network. Defaults to 0.0.
            output_activation_function (str): activation function of output layer. Defaults to "linear".
            output_init_bias: (np.ndarray, optional): constant init value for the output layer bias. Defaults to None.
            output_kernel_initializer: (str, keras.initializers.Initializer, optional): kernel initializer of output layer. Defaults to 'glorot_uniform'.
            dense_activation (str, optional): activation function of FC layers. Defaults to 'relu'.
            dense_kernel_initializer (str, keras.initializers.Initializer, optional): kernel initializer of FC layers. Defaults to 'glorot_uniform'.
            dense_bias_initializer (str, keras.initializers.Initializer, optional): bias initializer of FC layers. Defaults to 'zeros'.
            dense_kernel_regularizer (str | dict, optional): kernel regularizer of FC layers. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
            dense_bias_regularizer (str | dict, optional): bias regularizer of FC layers. Defaults to None. If a dict is passed, it must be a valid keras serialized regularizer.
        """
        super().__init__(**kwargs)

        self.out_steps = out_steps
        self.num_features = num_features
        self.dropout = dropout
        self.output_activation_function = output_activation_function
        self.output_init_bias = output_init_bias
        self.output_kernel_initializer = output_kernel_initializer
        self.use_bidirectional = use_bidirectional  # Store the new parameter

        # dense layers regularizers
        if isinstance(dense_kernel_regularizer, dict):
            self.dense_kernel_regularizer = keras.regularizers.get(
                dense_kernel_regularizer)
        else:
            self.dense_kernel_regularizer = dense_kernel_regularizer
        if isinstance(dense_bias_regularizer, dict):
            self.dense_bias_regularizer = keras.regularizers.get(
                dense_bias_regularizer)
        else:
            self.dense_bias_regularizer = dense_bias_regularizer

        if isinstance(lstm_units, int):
            lstm_units = [lstm_units]
        assert isinstance(
            lstm_units, list), 'lstm_units must be either an int or a list of int'
        self.lstm_units = lstm_units

        self.lstm_encoder_layers = []
        self.lstm_decoder_layers = []

        for layer_count, num_units in enumerate(self.lstm_units):
            lstm_layer = keras.layers.LSTM(num_units, return_sequences=True,
                                           dropout=self.dropout, name=f'decoder{layer_count}')
            if self.use_bidirectional:
                lstm_layer = keras.layers.Bidirectional(
                    lstm_layer, merge_mode='sum', name=f'bidirectional_decoder{layer_count}')
            self.lstm_decoder_layers.append(lstm_layer)

        last_lstm_encoder_units = self.lstm_units[-1]

        last_lstm_layer = keras.layers.LSTM(last_lstm_encoder_units, return_sequences=False,
                                            return_state=True, dropout=self.dropout,
                                            name=f'encoder{len(lstm_units)}')
        if self.use_bidirectional:
            last_lstm_layer = keras.layers.Bidirectional(
                last_lstm_layer, merge_mode='sum', name=f'bidirectional_encoder{len(lstm_units)}')
        self.last_lstm_encoder = last_lstm_layer

        for layer_count, num_units in enumerate(self.lstm_units[0:-1]):
            lstm_layer = keras.layers.LSTM(num_units, return_sequences=True,
                                           return_state=True, dropout=self.dropout,
                                           name=f'encoder{layer_count}')
            if self.use_bidirectional:
                lstm_layer = keras.layers.Bidirectional(
                    lstm_layer, merge_mode='sum', name=f'bidirectional_encoder{layer_count}')
            self.lstm_encoder_layers.append(lstm_layer)

        # dense layers

        if isinstance(dense_units, int):
            dense_units = [dense_units]
        assert isinstance(
            dense_units, list), 'dense units must be either an int or a list of int'
        self.dense_units = dense_units

        self.dense_layers = []
        for layer_count, num_units in enumerate(dense_units):
            self.dense_layers.append(keras.layers.Dense(num_units, activation=dense_activation,
                                                        kernel_initializer=dense_kernel_initializer,
                                                        bias_initializer=dense_bias_initializer,
                                                        kernel_regularizer=self.dense_kernel_regularizer,
                                                        bias_regularizer=self.dense_bias_regularizer,
                                                        name=f"dense{layer_count}"))

        self.dense_out = keras.layers.Dense(
            self.num_features,
            activation=output_activation_function,
            kernel_initializer=self.output_kernel_initializer)

        if self.output_init_bias is not None:
            self.dense_out.bias_initializer = keras.initializers.Constant(
                self.output_init_bias)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "out_steps": self.out_steps,
            "num_features": self.num_features,
            "dropout": self.dropout,
            "output_activation_function": self.output_activation_function,
            "output_kernel_initializer": self.output_kernel_initializer,
            "dense_kernel_regularizer": self.dense_kernel_regularizer,
            "dense_bias_regularizer": self.dense_bias_regularizer,
            # Add the new parameter to the config
            "use_bidirectional": self.use_bidirectional
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

    def call(self, inputs, training=None):
        """ Forward pass of the model.

        Args:
            inputs: input tuple containing (past_observations, past_exogenous, future_exogenous)
            training: boolean, whether the model is in training mode

        Returns:
            tensor of next day forecasts
        """
        # features.shape => (batch, num_timesteps, num_features)
        # past_exogenous.shape => (batch, num_timesteps, num_past_exogenous_features)
        # future_exogenous.shape => (batch, num_timesteps, num_future_exogenous_features)
        features, past_exogenous, future_exogenous = inputs
        # x.shape => (batch, num_timesteps,
        #             num_features + num_exogenous_features)
        x = keras.ops.concatenate([features, past_exogenous], axis=2)

        states = []
        for lstm_encoder in self.lstm_encoder_layers:
            # x.shape => (batch, num_time_steps, lstm_units)
            # h.shape => (batch, lstm_units)
            # c.shape => (batch, lstm_units)
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
        for lstm_decoder, init_state in zip(self.lstm_decoder_layers, states):
            # x.shape => (batch, out_steps, lstm_units)
            x = lstm_decoder(x, initial_state=init_state)

        for dense_layer in self.dense_layers:
            # x.shape => (batch, out_steps, dense_layer.units)
            x = dense_layer(x)

        # x.shape => (batch, out_steps, num_features)
        x = self.dense_out(x)

        return x
