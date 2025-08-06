import keras
import numpy as np


class NRMSE(keras.metrics.RootMeanSquaredError):
    """
    Normalized Root Mean Squared Error (NRMSE) metric.

    NRMSE is a metric used to evaluate the performance of regression models.
    It calculates the root mean squared error between the true and predicted values,
    and then normalizes it by dividing it by a normalization factor. 
    The normalization factor can be provided as an argument, or it will be calculated as the range of values in the true values seen so far.
    This metric is calculated over the entire dataset seen since the last reset.

    Args:
        name (str): Name of the metric (default: 'nrmse').
        dtype: Data type of the metric result.
        normalization_factor (float | None): Value to normalize the NRMSE values (default: None).

    Methods:
        update_state(y_true, y_pred, sample_weight=None): Updates the metric state with new observations.
        result(): Computes and returns the metric result.
        reset_state(): Resets the metric state.

    Example usage: 
        nrmse_metric = NRMSE()
        nrmse_metric.update_state(y_true, y_pred)
        result = nrmse_metric.result()
    """

    def __init__(self, name='nrmse', dtype=None, normalization_factor: float | None = None):
        super(NRMSE, self).__init__(name=name, dtype=dtype)
        self.normalization_factor = normalization_factor

        if self.normalization_factor is None:
            self.max_val = self.add_weight(
                name='max_val', initializer=keras.initializers.Constant(value=-np.inf))
            self.min_val = self.add_weight(
                name='min_val', initializer=keras.initializers.Constant(value=np.inf))

    # @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric by accumulating the true and predicted values. 
        If the normalization factor is not provided, it also keeps track of the maximum and minimum values seen so far.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each sample.

        Returns:
            None
        """
        super().update_state(y_true, y_pred, sample_weight)
        if self.normalization_factor is None:
            self.max_val.assign(keras.ops.maximum(
                self.max_val, keras.ops.amax(y_true)))
            self.min_val.assign(keras.ops.minimum(
                self.min_val, keras.ops.amin(y_true)))

    # @tf.function
    def result(self):
        """
        Calculates the NRMSE value.

        Returns:
            NRMSE value.
        """
        if self.normalization_factor is None:
            normalization_factor = keras.ops.subtract(
                self.max_val, self.min_val)
        else:
            normalization_factor = self.normalization_factor
        rmse = super().result()
        return keras.ops.divide(rmse, normalization_factor)

    def reset_state(self):
        """
        Resets the metric state.

        Returns:
            None
        """
        super().reset_state()
        if self.normalization_factor is None:
            self.max_val.assign(-np.inf)
            self.min_val.assign(np.inf)


class NMAX(keras.metrics.Metric):
    """
    Normalized Maximum error (NMAX) metric.

    This metric calculates the normalized maximum error between the true values (y_true) and the predicted values (y_pred).
    The result is the maximum error divided by a normalization factor.
    If the normalization factor is not provided, it calculates the range of values in y_true seen so far.
    This metric is calculated over the entire dataset seen since the last reset.

    Args:
        name (str): Name of the metric (default: 'nmax').
        dtype: Data type of the metric result (default: None).
        normalization_factor (float | None): Value to normalize the NMAX values (default: None).

    Methods:
        update_state(y_true, y_pred, sample_weight=None): Updates the metric state with new observations.
        result(): Computes and returns the metric result.
        reset_state(): Resets the metric state.

    Example usage:
        nmax_metric = NMAX()
        nmax_metric.update_state(y_true, y_pred)
        result = nmax_metric.result()
    """

    def __init__(self, name='nmax', dtype=None, normalization_factor: float | None = None):
        super(NMAX, self).__init__(name=name, dtype=dtype)
        self.normalization_factor = normalization_factor
        if normalization_factor is None:
            self.max_val = self.add_weight(
                name='max_val', initializer=keras.initializers.Constant(value=-np.inf))
            self.min_val = self.add_weight(
                name='min_val', initializer=keras.initializers.Constant(value=np.inf))
        self.max_error = self.add_weight(
            name='max_error', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric by accumulating the true and predicted values.
        If the normalization factor is not provided, it also keeps track of the range of values in y_true 

        Args:
            y_true: True values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each sample.

        Returns:    
            None
        """
        self.max_error.assign(keras.ops.maximum(self.max_error,
                                                keras.ops.amax
                                                (keras.ops.abs(keras.ops.subtract(y_true, y_pred)))))
        if self.normalization_factor is None:
            self.max_val.assign(keras.ops.maximum(
                self.max_val, keras.ops.amax(y_true)))
            self.min_val.assign(keras.ops.minimum(
                self.min_val, keras.ops.amin(y_true)))

    def result(self):
        """
        Calculates the NMAX value.

        Returns:
            NMAX value.
        """
        if self.normalization_factor is None:
            normalization_factor = keras.ops.subtract(
                self.max_val, self.min_val)
        else:
            normalization_factor = self.normalization_factor
        return self.max_error / normalization_factor

    def reset_state(self):
        """
        Resets the metric state.

        Returns:
            None
        """
        self.max_error.assign(0.0)
        self.max_val.assign(-np.inf)
        self.min_val.assign(np.inf)


class MeanWindowNRMSE(keras.metrics.Mean):
    def __init__(self, name='mean_window_nrmse',
                 normalization_factor: float | None = None, **kwargs):
        """
        Average Normalized Root Mean Squared Error (NRMSE) metric for a batch of windows.

        This metric calculates the NRMSE for each window in the batch, and then computes the mean NRMSE value across all windows.
        The NRMSE is calculated by dividing the RMSE of each window by the range of values in the true values of that window.
        The normalization factor can be provided as an argument, or it will be calculated as the range of values in the true values of each window.

        Args:
            name (str): Name of the metric (default: 'mean_window_nrmse').
            normalization_factor (float | None): Value to normalize the NRMSE values (default: None).
            **kwargs: Additional keyword arguments.
        """
        super(MeanWindowNRMSE, self).__init__(name=name, **kwargs)
        # Initialize any necessary variables or state here
        self.normalization_factor = normalization_factor

    # @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update the metric state with new observations
        # Calculate the metric value based on y_true and y_pred
        # Update any necessary variables or state

        # currently only supports num_features = 1
        assert y_true.shape[-1] == 1, "y_true must have shape (batch_size, window_size, 1)"

        # squared_error.shape = (batch_size, window_size, 1)
        squared_error = keras.ops.square(keras.ops.subtract(y_true, y_pred))
        # mean_squared_error.shape = (batch_size, 1)
        mean_squared_error = keras.ops.mean(squared_error, axis=1)

        # normalization_factor.shape = (batch_size, 1)
        if self.normalization_factor is None:
            # calculate range of values in y_true per each window in the batch
            max_val = keras.ops.amax(y_true, axis=1)
            min_val = keras.ops.amin(y_true, axis=1)
            normalization_factor = keras.ops.subtract(max_val, min_val)
        else:
            normalization_factor = self.normalization_factor

        nrsme_values = keras.ops.divide(keras.ops.sqrt(
            mean_squared_error), normalization_factor)

        batch_mean_nrmse_values = keras.ops.mean(nrsme_values, axis=0)
        return super().update_state(batch_mean_nrmse_values, sample_weight=sample_weight)
