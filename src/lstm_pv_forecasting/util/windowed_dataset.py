import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union


def make_windowed_dataset(data: pd.DataFrame, input_window_size: int, output_window_size: int,
                          batch_size: int = 32, label_columns: list[str] | None = None,
                          exogenous_columns: list[str] | tuple[list[str],
                                                               list[str]] | None = None,
                          next_period_stats: pd.DataFrame | None = None,
                          shift: int = 1, shuffle: bool = True,
                          shuffle_buffer_size: int | None = None) -> tf.data.Dataset:
    """Creates windowed tf.data.Dataset from data.
        The dataset returned contains batches of tuples (inputs, labels).
        If exogenous_columns is specified the dataset contains
        inputs as tuples (features, past_exogenous, future_exogenous).
        If next_period_stats is not None, the inputs also contains
        stats for the next period.

    Args:
        data (pd.DataFrame): Dataframe containig the time series.
        input_window_size (int): size of the input window.
        output_window_size (int): size of the output window
        batch_size (int, optional): Batch size of the dataset. Defaults to 32.
        label_columns (list[str], optional): list of the columns to use as target labels. If not specified all the features are used as targets. Defaults to None.
        exogenous_columns (Union[list[str], Tuple[list[str], list[str]]], optional): List of columns to use as exogenous variables or tuple of lists of columns. In this last case the firs list specifies past exogenous variables and the second future exogenous variables (must be a subset of past exogenous). If None no exogenous variables are set. Defaults to None.
        next_period_stats (pd.DataFrame, optional): Dataframe containing the next period stats. If it is not None, shift must be equal to output_window_size. Defaults to None.
        shift (int, optional): shift between consecutive windows of data in the timeseries. Defaults to 1.
        shuffle (bool, optional): wether to shuffle the windows or not. Defaults to True.
        shuffle_buffer_size (int, optional): size of the buffer used for shuffling. If None, it is set to batch_size * 8. Defaults to None.

    Returns:
        tf.data.Dataset: The returned dataset yields (batch_of_features, batch_of_observations). If ``exogenous_columns`` batch_of_features is just observations of the time series of lenght ``input_window_size``, otherwise it is a tuple (batch_of_features, batch_of_past_exogenous, batch_of_future_exogenous). If ``next_period_stats`` is not None, batch of features is a tuple that also contains period stats for a period corresponding to ``output_window_size``.
    """
    column_index = {name: i for (i, name) in enumerate(data.columns)}

    if (label_columns is None) and (exogenous_columns is not None):
        raise ValueError(
            "if label_columns is None, exogenous_columns must be None")

    if label_columns is not None:
        assert set(label_columns).issubset(
            data.columns), "label_columns must be a subset of columns in data"
        feature_columns = label_columns
    else:
        feature_columns = data.columns

    if exogenous_columns is not None:
        # if no distinction is pecified between past and future exogenous
        # use the same list for both
        if isinstance(exogenous_columns, list):
            exogenous_columns = (exogenous_columns, exogenous_columns)

        past_exogenous_columns, future_exogenous_columns = exogenous_columns
        assert set(past_exogenous_columns).issubset(set(data.columns)
                                                    ), "past_exogenous_columns must be a subset of columns in data"
        assert len(set(past_exogenous_columns).intersection(set(feature_columns))
                   ) == 0, "past_exogenous_columns must not contain columns in label_columns"
        assert set(future_exogenous_columns).issubset(set(past_exogenous_columns)
                                                      ), "future_exogenous_columns must be a subset of past_exogenous_columns"

    if next_period_stats is not None:
        # if shift is different from output_window_size
        # next_period_stats cannot be specified
        if shift != output_window_size:
            raise ValueError(
                "next_period_stats can be specified only if shift is equal to output_window_size")

    data_array = np.array(data, dtype=np.float32)

    # Not sure why, but this block of code does not give
    # good results when training on the resulting dataset
    # substituted with the call to timeseries_dataset_from_array
    #
    # dataset = tf.data.Dataset.from_tensor_slices(data)
    # tot_window_size = input_window_size + output_window_size
    # dataset = dataset.window(tot_window_size, shift=shift, drop_remainder=True)
    # dataset = dataset.flat_map(lambda x: x).batch(tot_window_size)
    # if shuffle:
    #     dataset = dataset.shuffle(32)
    # dataset = dataset.batch(batch_size)

    if next_period_stats is not None:
        # last window must be dropped if next_period_stats is specified
        data_array = data_array[:-output_window_size]

    dataset = keras.preprocessing.timeseries_dataset_from_array(
        data=data_array,
        targets=None,
        sequence_length=input_window_size+output_window_size,
        sequence_stride=shift,
        shuffle=False,
        batch_size=None)  # type: tf.data.Dataset

    def split(window: tf.Tensor) -> tuple[tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        # this function is used to split the window into inputs and labels
        # and to stack the exogenous variables if present
        # it is used in the map method of the dataset

        # inputs = tf.convert_to_tensor(window[:input_window_size, :])
        inputs = window[:input_window_size, :]
        inputs = tf.stack([
            inputs[:, column_index[name]] for name in feature_columns
        ], axis=-1)
        inputs.set_shape([input_window_size, None])  # type: ignore

        labels = window[-output_window_size:, :]
        labels = tf.stack([
            labels[:, column_index[name]] for name in feature_columns
        ], axis=-1)
        labels.set_shape([output_window_size, None])  # type: ignore

        if exogenous_columns is not None:
            past_exogenous_columns, future_exogenous_columns = exogenous_columns

            exogenous_past = window[:input_window_size, :]
            exogenous_past = tf.stack([
                exogenous_past[:, column_index[name]] for name in past_exogenous_columns
            ], axis=-1)
            exogenous_past.set_shape(  # type: ignore
                [input_window_size, None])

            exogenous_future = window[-output_window_size:, :]
            exogenous_future = tf.stack([
                exogenous_future[:, column_index[name]] for name in future_exogenous_columns
            ], axis=-1)
            exogenous_future.set_shape(  # type: ignore
                [output_window_size, None])
            inputs = inputs, exogenous_past, exogenous_future

        return inputs, labels

    @tf.function
    def split_with_next_period_stats(window: tf.Tensor, next_period_stats: tf.Tensor) -> tuple[tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        inputs, labels = split(window)
        if isinstance(inputs, tuple):
            inputs = inputs + (next_period_stats,)
        else:
            inputs = inputs, next_period_stats

        return inputs, labels

    # input_tensor_list = []
    # labels_tensor_list = []
    # for window_batch in dataset.as_numpy_iterator():
    #     input_tensor, labels_tensor = split(window=window_batch)
    #     input_tensor_list.append(input_tensor)
    #     labels_tensor_list.append(labels_tensor)

    # # temporary workaround to test performance in this case

    # # input_tensor_list = input_tensor_list[:-1]
    # # labels_tensor_list = labels_tensor_list[:-1]

    # if exogenous_columns is not None:
    #     input_tensor_list, exogenous_past_tensor_list, exogenous_future_tensor_list = zip(
    #         *input_tensor_list)
    #     input_tensor_list = list(input_tensor_list)
    #     exogenous_past_tensor_list = list(exogenous_past_tensor_list)
    #     exogenous_future_tensor_list = list(exogenous_future_tensor_list)

    #     result_dataset = tf.data.Dataset.from_tensor_slices(
    #         ((tf.stack(input_tensor_list), tf.stack(exogenous_past_tensor_list), tf.stack(exogenous_future_tensor_list)), tf.stack(labels_tensor_list)))

    # else:
    #     result_dataset = tf.data.Dataset.from_tensor_slices(
    #         (tf.stack(input_tensor_list), tf.stack(labels_tensor_list)))

    if next_period_stats is not None:
        # add the next period stats to the dataset
        next_period_stats_array = np.array(next_period_stats, dtype=np.float32)
        # drop the first row (stats of first window) of next_period_stats
        next_period_stats_array = next_period_stats_array[1:]
        next_period_stats_dataset = tf.data.Dataset.from_tensor_slices(
            next_period_stats_array)
        dataset = tf.data.Dataset.zip((dataset, next_period_stats_dataset))

        result_dataset = dataset.map(
            split_with_next_period_stats, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        result_dataset = dataset.map(
            split, num_parallel_calls=tf.data.AUTOTUNE)  # .prefetch(14).cache()

    if shuffle:
        if shuffle_buffer_size is None:
            # value as used in keras.utils.timeseries_dataset_from_array
            shuffle_buffer_size = batch_size * 8
        result_dataset = result_dataset.shuffle(shuffle_buffer_size)
    result_dataset = result_dataset.batch(batch_size)

    return result_dataset
