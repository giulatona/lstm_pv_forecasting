import argparse
import logging
from math import e
import pathlib
import pickle

import dvclive
import keras
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

from lstm_pv_forecasting.model import lstm
from lstm_pv_forecasting.util import evaluation
from lstm_pv_forecasting.util import logging as util_logging
from lstm_pv_forecasting.util.load_config import load_config

logger = logging.getLogger(__name__)


def eval(data_dir: pathlib.Path, model_dir: pathlib.Path,
         input_window_size: int, output_window_size: int,
         window_shift: int,
         label_columns: list[str], past_exogenous_columns: list[str] | None,
         future_exogenous_columns: list[str] | None, num_fourier_terms: int,
         seasonality: int,
         plot_dir: pathlib.Path,
         use_next_period_stats: bool = False,
         random_seed: int = 1,
         eval_on_validation_set: bool = True):
    """Evaluates a model and saves the prediction results in data_dir.

    Args:
        data_dir (pathlib.Path): Directory containing the prepared datasets.
        model_dir (pathlib.Path): Directory containing the trained model.
        input_window_size (int): size of the input window.
        output_window_size (int): size of the output window.
        label_columns (list[str]): list of the columns to use as target labels.
        past_exogenous_columns (list[str] | None): list of columns to use as past exogenous variables. If None no past exogenous variables are set.
        future_exogenous_columns (list[str] | None): list of columns to use as future exogenous variables. If None no future exogenous variables are set.
        num_fourier_terms (int): number of fourier terms to use as (past and future) exogenous variables.
        seasonality (int): seasonality of the data, used to calculate MASE metric.
        plot_dir (pathlib.Path): Directory where to save the plots and the metrics.
        use_next_period_stats (bool, optional): Whether to use next period statistics or not. Defaults to False.
        random_seed (int, optional): Random seed. Defaults to 1.
        eval_on_validation_set (bool, optional): Whether to test on the validation set or on the test set. Defaults to True (use validation set).
    """

    logger.info("Eval stage")
    keras.utils.set_random_seed(random_seed)

    model_path = model_dir.joinpath("model.keras")
    if model_path.exists():
        model = keras.models.load_model(model_path)  # type: keras.Model | None
        if model is None:
            raise ValueError(f"Model file {model_path} is empty")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    predictions, daily_metrics = eval_model(data_dir, model, input_window_size, output_window_size,
                                            window_shift, label_columns, past_exogenous_columns,
                                            future_exogenous_columns, num_fourier_terms,
                                            seasonality,
                                            use_next_period_stats, random_seed,
                                            eval_on_validation_set)

    prediction_results_path = data_dir.joinpath("prediction_results.pkl")
    logger.info(f"saving prediction results in {prediction_results_path}")
    predictions.to_pickle(prediction_results_path)

    # plot backtesting prediction results
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({
        "font.family": "serif",   # specify font family here
        "font.serif": ["Times"],  # specify font here
        "font.size": 8,  # specify font size here
        "figure.figsize": [14 / 2.54, 7 / 2.54],
        "figure.dpi": 200.0
    })

    ax = predictions.plot()
    ax.set_ylabel("Power (kW)")
    fig = ax.get_figure()

    daily_metrics.reset_index().to_csv(plot_dir.joinpath(
        "daily_metrics.csv"), index=True, index_label="day_number")

    # saving plot and metrics through DVClive
    with dvclive.Live(str(plot_dir), cache_images=True, dvcyaml=None) as live:
        for metric_name, value in daily_metrics.mean().items():
            live.log_metric(f"test/{metric_name}", value, plot=False)
        # logging stdev, min and max nrsme
        live.log_metric("test/nrmse_min",
                        daily_metrics.nrmse.min(), plot=False)
        live.log_metric("test/nrmse_max",
                        daily_metrics.nrmse.max(), plot=False)
        live.log_metric("test/nrmse_std",
                        daily_metrics.nrmse.std(), plot=False)
        live.log_image("predictions.png", fig)


def eval_model(data_dir: pathlib.Path, model: keras.Model,
               input_window_size: int, output_window_size: int,
               window_shift: int,
               label_columns: list[str], past_exogenous_columns: list[str] | None,
               future_exogenous_columns: list[str] | None, num_fourier_terms: int,
               seasonality: int,
               use_next_period_stats: bool = False,
               random_seed: int = 1,
               eval_on_validation_set: bool = True,
               callbacks: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates a model and returns daily metrics.

    Args:
        data_dir (pathlib.Path): Directory containing the prepared datasets.
        model (keras.Model): Trained model.
        model_dir (pathlib.Path): Directory containing the trained model.
        input_window_size (int): size of the input window.
        output_window_size (int): size of the output window.
        label_columns (list[str]): list of the columns to use as target labels.
        past_exogenous_columns (list[str] | None): list of columns to use as past exogenous variables. If None no past exogenous variables are set.
        future_exogenous_columns (list[str] | None): list of columns to use as future exogenous variables. If None no future exogenous variables are set.
        num_fourier_terms (int): number of fourier terms to use as (past and future) exogenous variables.
        use_next_period_stats (bool, optional): Whether to use next period statistics or not. Defaults to False.
        seasonality (int): seasonality of the data, used to calculate MASE metric.
        random_seed (int, optional): Random seed. Defaults to 1.
        eval_on_validation_set (bool, optional): Whether to test on the validation set or on the test set. Defaults to True (use validation set).
        callbacks (list | None, optional): List of callbacks to use during evaluation. Defaults to None.

    Returns:
        pd.DataFrame: predictions
        pd.DataFrame: daily_metrics
    """
    keras.utils.set_random_seed(random_seed)

    logger.info("reading prepared dataset")
    train_df = pd.read_pickle(data_dir.joinpath("train.pkl"))
    if eval_on_validation_set:
        logger.info("reading validation set")
        df_to_eval = pd.read_pickle(data_dir.joinpath("val.pkl"))
    else:
        logger.info("reading test set")
        df_to_eval = pd.read_pickle(data_dir.joinpath("test.pkl"))

    with open(data_dir.joinpath("target_scaler.pkl"), 'rb') as f:
        target_scaler = pickle.load(f)

    fourier_terms = [f'{season} {term}{k}' for season in ['Year', 'Day']
                     for term in ['sin', 'cos'] for k in range(1, num_fourier_terms + 1)]

    if future_exogenous_columns is None:
        future_exogenous_columns = []
    if past_exogenous_columns is None:
        past_exogenous_columns = []

    backtest_exogenous_columns = (past_exogenous_columns + fourier_terms,
                                  future_exogenous_columns + fourier_terms)

    if use_next_period_stats:
        logger.info("reading next period statistics")
        if eval_on_validation_set:
            next_period_stats = pd.read_pickle(
                data_dir.joinpath("val_daily_stats_normalized.pkl"))
        else:
            next_period_stats = pd.read_pickle(
                data_dir.joinpath("test_daily_stats_normalized.pkl"))
    else:
        logger.info("next period statistics not used")
        next_period_stats = None

    predictions, metrics = evaluation.backtest(model,  # type: ignore
                                               df_to_eval, train_df, input_window_size=input_window_size,
                                               output_window_size=output_window_size, label_columns=label_columns,
                                               scaler=target_scaler,
                                               exogenous_columns=backtest_exogenous_columns,
                                               next_period_stats=next_period_stats,
                                               seasonality=seasonality,
                                               callbacks=callbacks)

    train_descaled = target_scaler.inverse_transform(
        train_df.loc[:, label_columns])

    daily_metrics = evaluation.calc_daily_metrics(
        result_df=predictions, train_target=train_descaled, seasonality=seasonality)

    return predictions, daily_metrics


if __name__ == "__main__":
    util_logging.setup_logging(logfilename_prefix="eval")

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_config(config_path=args.config)

    random_seed = params["base"]["random_seed"]

    data_dir = pathlib.Path(params["base"]["data_dir"])
    model_dir = pathlib.Path(params["base"]["model_dir"])
    plot_dir = pathlib.Path(params["eval"]["plot_dir"])
    seasonality = params["eval"]["seasonality"]

    use_next_period_stats = params["train"]["next_period_stats"]["next_period_stats"]["use_next_period_stats"]

    label_columns = params["train"]["model"]["label_columns"]
    past_exogenous_columns = params["train"]["model"]["past_exogenous_columns"]
    future_exogenous_columns = params["train"]["model"]["future_exogenous_columns"]

    eval_on_validation_set = params["eval"].get("eval_on_validation_set", True)

    eval(data_dir=data_dir, model_dir=model_dir,
         input_window_size=params["train"]["input_window_size"],
         output_window_size=params["train"]["output_window_size"],
         window_shift=params["train"]["window_shift"],
         label_columns=label_columns,
         past_exogenous_columns=past_exogenous_columns,
         future_exogenous_columns=future_exogenous_columns,
         num_fourier_terms=params["dataset"]["preprocess"]["num_fourier_terms"],
         seasonality=seasonality,
         plot_dir=plot_dir,
         use_next_period_stats=use_next_period_stats,
         random_seed=random_seed,
         eval_on_validation_set=eval_on_validation_set)
