import argparse
import importlib
import inspect
import logging
import pathlib
import time
from typing import Any, Literal, Sequence

from dvclive.keras import DVCLiveCallback
from dvclive.live import Live
import keras
from matplotlib.pylab import normal
import pandas as pd

from lstm_pv_forecasting.metrics.metrics import MeanWindowNRMSE
from lstm_pv_forecasting.metrics.metrics import NMAX
from lstm_pv_forecasting.metrics.metrics import NRMSE
from lstm_pv_forecasting.util import logging as util_logging
from lstm_pv_forecasting.util.load_config import load_config
from lstm_pv_forecasting.util.windowed_dataset import make_windowed_dataset

logger = logging.getLogger(__name__)

MODEL_NAMES = Literal["Lstm", "NARX", "Naive", "BlockLstm", "CnnLstm"]


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def train(data_dir_path: pathlib.Path, model_dir: pathlib.Path,
          model_name: MODEL_NAMES, batch_size: int,
          input_window_size: int, output_window_size: int,
          window_shift: int,
          label_columns: list[str], past_exogenous_columns: list[str] | None,
          future_exogenous_columns: list[str] | None, num_fourier_terms: int,
          epochs: int,
          loss_params: dict[str, Any],
          optimizer_params: dict[str, Any], random_seed: int = 1,
          early_stopping: bool = True,
          use_next_period_stats: bool = False,
          shuffle: bool = True,
          shuffle_buffer_size: int | None = None,
          **model_params):
    """Trains a model and saves it in model_dir.

    Args:
        data_dir_path (pathlib.Path): path to the directory containing the prepared datasets.
        model_dir (pathlib.Path): path to the directory where the model will be saved.
        model_name (MODEL_NAMES): name of the model architecture to train.
        batch_size (int): batch size.
        input_window_size (int): size of the input window.
        output_window_size (int): size of the output window.
        window_shift (int): number of steps to shift the window.
        label_columns (list[str]): list of the columns to use as target labels.
        past_exogenous_columns (list[str] | None): list of columns to use as past exogenous variables. If None no past exogenous variables are set.
        future_exogenous_columns (list[str] | None): list of columns to use as future exogenous variables. If None no future exogenous variables are set.
        num_fourier_terms (int): number of fourier terms to use as (past and future) exogenous variables.
        epochs (int): number of epochs to train.
        loss_params (dict[str, Any]): parameters for the loss function.
        optimizer_params (dict[str, Any]): parameters for the optimizer.
        random_seed (int, optional): random seed. Defaults to 1.
        early_stopping (bool, optional): wether to use early stopping or not. Defaults to True.
        use_next_period_stats (bool): wether to use next period stats or not. Defaults to False.
        shuffle (bool, optional): wether to shuffle the train dataset or not. Defaults to True.
        shuffle_buffer_size (int | None, optional): size of the buffer to use for shuffling the train dataset. If None the buffer size equals batch_size * 8. Defaults to None.
        **model_params: parameters for the model.
    """
    logger.info("Train stage")

    live = Live(dvcyaml=None)

    model, train_metrics, eval_metrics = train_model(data_dir_path, model_name, batch_size,
                                                     input_window_size, output_window_size, window_shift,
                                                     label_columns, past_exogenous_columns, future_exogenous_columns,
                                                     num_fourier_terms, epochs, loss_params, optimizer_params,
                                                     random_seed, early_stopping, use_next_period_stats, shuffle,
                                                     shuffle_buffer_size,
                                                     [DVCLiveCallback(
                                                         live=live)],
                                                     **model_params)

    live.summary["train"] = train_metrics
    live.summary["eval"] = eval_metrics

    live.make_summary()

    model_save_path = model_dir.joinpath("model.keras")
    logger.info(f"saving model in {model_save_path}")
    model.save(model_save_path)


def train_model(data_dir_path: pathlib.Path,
                model_name: MODEL_NAMES, batch_size: int,
                input_window_size: int, output_window_size: int,
                window_shift: int,
                label_columns: list[str], past_exogenous_columns: list[str] | None,
                future_exogenous_columns: list[str] | None, num_fourier_terms: int,
                epochs: int,
                loss_params: dict[str, Any],
                optimizer_params: dict[str, Any], random_seed: int = 1,
                early_stopping: bool = True,
                use_next_period_stats: bool = False,
                shuffle: bool = True,
                shuffle_buffer_size: int | None = None,
                fit_callbacks: Sequence[keras.callbacks.Callback] = [],
                **model_params) -> tuple[keras.Model, dict, dict]:
    """Trains a model and returns it.

    Args:
        data_dir_path (pathlib.Path): path to the directory containing the prepared datasets.
        model_name (MODEL_NAMES): name of the model architecture to train.
        batch_size (int): batch size.
        input_window_size (int): size of the input window.
        output_window_size (int): size of the output window.
        window_shift (int): number of steps to shift the window.
        label_columns (list[str]): list of the columns to use as target labels.
        past_exogenous_columns (list[str] | None): list of columns to use as past exogenous variables. If None no past exogenous variables are set.
        future_exogenous_columns (list[str] | None): list of columns to use as future exogenous variables. If None no future exogenous variables are set.
        num_fourier_terms (int): number of fourier terms to use as (past and future) exogenous variables.
        epochs (int): number of epochs to train.
        loss_params (dict[str, Any]): parameters for the loss function.
        optimizer_params (dict[str, Any]): parameters for the optimizer.
        random_seed (int, optional): random seed. Defaults to 1.
        early_stopping (bool, optional): wether to use early stopping or not. Defaults to True.
        use_next_period_stats (bool): wether to use next period stats or not. Defaults to False.
        shuffle (bool, optional): wether to shuffle the train dataset or not. Defaults to True.
        shuffle_buffer_size (int | None, optional): size of the buffer to use for shuffling the train dataset. If None the buffer size equals batch_size * 8. Defaults to None.
        fit_callbacks (Sequence[keras.callbacks.Callback], optional): list of callbacks to use during training. Defaults to [].
        **model_params: parameters for the model.

    Returns:
        keras.Model: trained model
        dict: train metrics
        dict: validation metrics
    """
    keras.utils.set_random_seed(random_seed)

    logger.info("reading prepared datasets")
    train_df = pd.read_pickle(data_dir_path.joinpath("train.pkl"))
    val_df = pd.read_pickle(data_dir_path.joinpath("val.pkl"))

    fourier_terms = [f'{season} {term}{k}' for season in ['Year', 'Day']
                     for term in ['sin', 'cos'] for k in range(1, num_fourier_terms + 1)]

    if future_exogenous_columns is None:
        future_exogenous_columns = []
    if past_exogenous_columns is None:
        past_exogenous_columns = []

    train_exogenous_columns = (past_exogenous_columns + fourier_terms,
                               future_exogenous_columns + fourier_terms)

    if use_next_period_stats:
        logger.info("reading next period stats")
        train_next_period_stats = pd.read_pickle(
            data_dir_path.joinpath("train_daily_stats_normalized.pkl"))
        val_next_period_stats = pd.read_pickle(
            data_dir_path.joinpath("val_daily_stats_normalized.pkl"))
    else:
        logger.info("next period stats not used")
        train_next_period_stats = None
        val_next_period_stats = None

    train_dataset = make_windowed_dataset(train_df, input_window_size=input_window_size,
                                          output_window_size=output_window_size,
                                          shift=window_shift,
                                          batch_size=batch_size, label_columns=label_columns,
                                          exogenous_columns=train_exogenous_columns,
                                          next_period_stats=train_next_period_stats,
                                          shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size)

    evaluation_train_dataset = train_dataset

    # if training in open loop mode, the output window size is set to the number of
    # steps to predict in one shot.
    if model_params.get("open_loop_training_mode", False):
        one_shot_steps = model_params.get("one_shot_steps", 1)
        train_dataset = make_windowed_dataset(train_df, input_window_size=input_window_size,
                                              output_window_size=one_shot_steps,
                                              shift=window_shift,
                                              batch_size=batch_size, label_columns=label_columns,
                                              exogenous_columns=train_exogenous_columns,
                                              shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size)

    logger.info("train dataset created, number of batches: %d",
                len(train_dataset))
    val_dataset = make_windowed_dataset(val_df, input_window_size=input_window_size,
                                        output_window_size=output_window_size,
                                        shift=window_shift,
                                        batch_size=batch_size, label_columns=label_columns,
                                        exogenous_columns=train_exogenous_columns,
                                        next_period_stats=val_next_period_stats,
                                        shuffle=False)

    # set model output initial bias
    train_mean = train_df.loc[:, label_columns].mean().values

    logger.info(f"training model {model_name}")

    model_class = get_class("lstm_pv_forecasting.model", model_name)

    model = model_class(output_init_bias=train_mean,
                        **model_params)  # type: keras.Model

    learning_rate = optimizer_params.pop("learning_rate")
    if isinstance(learning_rate, float):
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, **optimizer_params)
    elif isinstance(learning_rate, dict):
        learning_rate_name = learning_rate["name"]  # type: str
        learning_rate_class = getattr(
            keras.optimizers.schedules, learning_rate_name)
        learning_rate_args = inspect.signature(
            learning_rate_class).parameters.keys()
        learning_rate_params = {
            k: v for k, v in learning_rate.items() if k in learning_rate_args}
        learning_rate_instance = learning_rate_class(**learning_rate_params)
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_instance, **optimizer_params)
    else:
        raise ValueError("learning_rate must be float or dict")

    loss_class = getattr(keras.losses, loss_params["name"])
    loss_instance = loss_class(**loss_params)

    # train and validation datasets are normalized in [-1, 1] range
    normalization_factor = 2
    mean_window_nrmse = MeanWindowNRMSE(
        normalization_factor=normalization_factor)

    if model_name in ["NARX"]:
        jit_compile = True
    else:
        jit_compile = False
    model.compile(loss=loss_instance,
                  optimizer=optimizer,
                  metrics=[keras.metrics.MeanAbsoluteError(), NRMSE(), NMAX(),
                           mean_window_nrmse],
                  jit_compile=jit_compile)

    callbacks = []
    if fit_callbacks:
        callbacks.extend(fit_callbacks)

    if early_stopping:
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_mean_window_nrmse',
                                                                patience=15,
                                                                mode='min',
                                                                restore_best_weights=True,
                                                                start_from_epoch=10)
        callbacks.append(early_stopping_callback)
    else:
        # since early stopping is disabled we need to manually save the optimal checkpoint and restore it at the end
        checkpoint_dir = pathlib.Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_dir.joinpath("model.weights.h5"),
            monitor='val_loss', mode='min',
            save_freq='epoch',
            save_best_only=True, save_weights_only=True)
        callbacks.append(checkpoint_callback)

    # run a single batch to initialize the model
    # Note: val_dataset is used instead of train_dataset because
    # in case of open loop training (NARX model) the train_dataset is not compatible
    # with the normal forward pass of the model
    model.call(val_dataset.as_numpy_iterator().next()[0])

    if model.trainable_variables:
        logger.info("started training")
        start_time = time.time()
        model.fit(train_dataset.cache(), epochs=epochs,
                  validation_data=val_dataset.cache(),
                  callbacks=callbacks,
                  verbose=0)  # type: ignore
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"training completed, elapsed time {elapsed_time}")
        # restore the best checkpoint
        if not early_stopping:
            model.load_weights(checkpoint_dir.joinpath("model.weights.h5"))

    else:  # for example in the case of naive model
        logger.info(
            "no trainable variables found, skipping training. Running train just to get dummy metrics")
        model.fit(train_dataset, epochs=1,
                  validation_data=val_dataset,
                  callbacks=callbacks,
                  verbose=0)  # type: ignore

    # evaluate metrics on train and validation datasets
    logger.info("evaluating model best checkpoint")
    train_metrics = model.evaluate(
        evaluation_train_dataset, return_dict=True, verbose=0, batch_size=batch_size)
    val_metrics = model.evaluate(
        val_dataset, return_dict=True, verbose=0, batch_size=batch_size)

    return model, train_metrics, val_metrics


if __name__ == "__main__":
    util_logging.setup_logging(logfilename_prefix="train")

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_config(config_path=args.config)

    random_seed = params["base"]["random_seed"]
    num_fourier_terms = params["dataset"]["preprocess"]["num_fourier_terms"]
    data_dir = pathlib.Path(params["base"]["data_dir"])
    model_dir = pathlib.Path(params["base"]["model_dir"])
    early_stopping = params["train"]["early_stopping"]
    loss_params = params["train"]["loss"]
    optimizer_params = params["train"]["optimizer_params"]
    model_name = params["train"]["model"]["name"]
    model_params = params["train"]["model"]["model_params"]

    use_next_period_stats = params["train"]["next_period_stats"]["next_period_stats"]["use_next_period_stats"]

    label_columns = params["train"]["model"]["label_columns"]
    past_exogenous_columns = params["train"]["model"]["past_exogenous_columns"]
    future_exogenous_columns = params["train"]["model"]["future_exogenous_columns"]

    train(data_dir, model_dir,
          model_name=model_name,
          batch_size=params["train"]["batch_size"],
          input_window_size=params["train"]["input_window_size"],
          output_window_size=params["train"]["output_window_size"],
          window_shift=params["train"]["window_shift"],
          label_columns=label_columns,
          past_exogenous_columns=past_exogenous_columns,
          future_exogenous_columns=future_exogenous_columns,
          num_fourier_terms=params["dataset"]["preprocess"]["num_fourier_terms"],
          epochs=params["train"]["epochs"],
          use_next_period_stats=use_next_period_stats,
          loss_params=loss_params,
          optimizer_params=optimizer_params,
          random_seed=random_seed,
          early_stopping=early_stopping,
          shuffle=params["train"]["shuffle"],
          shuffle_buffer_size=params["train"]["shuffle_buffer_size"],
          **model_params)
