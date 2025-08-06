import keras
import numpy as np


@keras.saving.register_keras_serializable(package="lstm_pv_forecasting.model")
class Naive(keras.Model):
    def __init__(self, out_steps: int, exogenous: bool = False,
                 output_init_bias: np.ndarray | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_steps = out_steps
        self.exogenous = exogenous
        self.output_init_bias = output_init_bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_steps": self.out_steps,
            "exogenous": self.exogenous,
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
        if self.exogenous:
            features, past_exogenous, future_exogenous = inputs
        else:
            features = inputs

        return features[:, -self.out_steps:, :]
