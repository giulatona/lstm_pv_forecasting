from operator import call
from typing import Any, Union

import keras
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as metrics

from lstm_pv_forecasting.util.windowed_dataset import make_windowed_dataset


def backtest(model: keras.Model, dataframe: pd.DataFrame,
             train_dataframe: pd.DataFrame,
             input_window_size: int,
             output_window_size: int,
             label_columns: list[str],
             exogenous_columns: list[str] | tuple[list[str],
                                                  list[str]] | None = None,
             next_period_stats: pd.DataFrame | None = None,
             scaler: Any = None, seasonality: int = 24,
             callbacks: list | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest a model on a dataframe.

    Args:
        model (keras.Model): trained model
        dataframe (pd.DataFrame): dataframe to backtest
        train_dataframe (pd.DataFrame): dataframe used to train the model
        input_window_size (int): size of the input window
        output_window_size (int): size of the output window
        label_columns (list[str]): list of the columns to use as target labels
        exogenous_columns (Union[list[str], tuple[list[str], list[str]], None], optional): list of columns to use as exogenous variables. If None no exogenous variables are set. Defaults to None.
        next_period_stats (pd.DataFrame, optional): dataframe containing the next period statistics. Defaults to None.
        scaler (Any, optional): scaler to use to descale the predictions. Defaults to None.
        seasonality (int, optional): seasonality of the data, used to calculate MASE metric. Defaults to 24.
        callbacks (list | None, optional): List of callbacks to use during evaluation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: tuple containing the result dataframe and the metrics
    """
    if callbacks is None:
        callbacks = []
    dataset = make_windowed_dataset(dataframe, input_window_size=input_window_size, output_window_size=output_window_size,
                                    batch_size=1, label_columns=label_columns,
                                    exogenous_columns=exogenous_columns,
                                    next_period_stats=next_period_stats,
                                    shift=output_window_size, shuffle=False)

    predictions = model.predict(dataset, callbacks=callbacks)
    predictions = np.concatenate(predictions[:, :, 0], axis=0)
    if scaler is not None:
        predictions = scaler.inverse_transform(
            np.expand_dims(predictions, axis=1))
        train_descaled = scaler.inverse_transform(
            train_dataframe.loc[:, label_columns])
    else:
        train_descaled = train_dataframe.loc[:, label_columns]

    if len(label_columns) > 1:
        raise NotImplementedError(
            "Normalization factor calculation on multivariate prediction not supported yet")
    normalization_factor = np.max(train_descaled) - np.min(train_descaled)

    predictions = np.clip(predictions, a_min=0, a_max=None)

    res_slice = slice(input_window_size, input_window_size + len(predictions))

    result = pd.DataFrame(scaler.inverse_transform(dataframe.loc[dataframe.index[res_slice], label_columns]),
                          index=dataframe.index[res_slice], columns=['observations'])
    result.loc[dataframe.index[res_slice], 'predictions'] = predictions

    result_metrics = {'mae': metrics.mean_absolute_error(result.observations,
                                                         result.predictions),
                      'mape': metrics.mean_absolute_percentage_error(result.observations,
                                                                     result.predictions),
                      'nrmse': nrmse(result.observations,
                                     result.predictions,
                                     normalization_factor=normalization_factor),
                      'nmax': nmax(result.observations,
                                   result.predictions,
                                   normalization_factor=normalization_factor),
                      'mase': mase(result.observations,
                                   result.predictions,
                                   train_descaled, seasonality=seasonality)
                      }

    return result, result_metrics


def calc_daily_metrics(result_df: pd.DataFrame, train_target: np.ndarray,
                       seasonality: int) -> pd.DataFrame:
    grouped = result_df.groupby(pd.Grouper(freq='d'))

    if train_target.shape[-1] > 1:
        raise NotImplementedError(
            "Normalization factor calculation on multivariate prediction not supported yet")
    normalization_factor = np.max(train_target) - np.min(train_target)

    res_metrics = {}
    for day, df in grouped:
        res_metrics.update({day:
                            {'mae': metrics.mean_absolute_error(df.observations,
                                                                df.predictions),
                             'mape': metrics.mean_absolute_percentage_error(df.observations,
                                                                            df.predictions),
                             'nrmse': nrmse(df.observations,
                                            df.predictions,
                                            normalization_factor=normalization_factor),
                             'nmax': nmax(df.observations,
                                          df.predictions,
                                          normalization_factor=normalization_factor),
                             'mase': mase(df.observations,
                                          df.predictions,
                                          train_target,
                                          seasonality=seasonality)
                             }})
    return pd.DataFrame(res_metrics).T


def nrmse(y_true, y_pred, normalization_factor: float | None = None):
    if normalization_factor is None:
        normalization_factor = np.max(y_true) - np.min(y_true)
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred)) / normalization_factor


def nmax(y_true, y_pred, normalization_factor: float | None = None):
    if normalization_factor is None:
        normalization_factor = np.max(y_true) - np.min(y_true)
    return metrics.max_error(y_true, y_pred) / normalization_factor


def mase(y_true, y_pred, y_train, seasonality: int):
    mae = metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
    return mae / np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
