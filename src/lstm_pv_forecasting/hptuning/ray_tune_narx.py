import argparse
import logging
import pathlib

import keras
import numpy as np
import ray
from ray import tune
from ray.air.integrations.keras import ReportCheckpointCallback

from lstm_pv_forecasting.stages.eval import eval_model
from lstm_pv_forecasting.stages.train import train_model
from lstm_pv_forecasting.util.load_config import load_config
import lstm_pv_forecasting.util.logging as util_logging

logger = logging.getLogger(__name__)


def objective(config):
    data_dir_path = config["data_dir"]
    model_name = config["model_name"]
    label_columns = config["label_columns"]
    past_exogenous_columns = config["past_exogenous_columns"]
    future_exogenous_columns = config["future_exogenous_columns"]
    num_fourier_terms = config["num_fourier_terms"]
    random_seed = config["random_seed"]
    use_next_period_stats = config["use_next_period_stats"]
    early_stopping = config["early_stopping"]
    batch_size = config["batch_size"]
    input_window_size = config["input_window_size"]
    output_window_size = config["output_window_size"]
    window_shift = config["window_shift"]
    label_columns = config["label_columns"]
    past_exogenous_columns = config["past_exogenous_columns"]
    future_exogenous_columns = config["future_exogenous_columns"]
    num_fourier_terms = config["num_fourier_terms"]
    seasonality = config["seasonality"]
    epochs = config["epochs"]
    random_seed = config["random_seed"]
    early_stopping = config["early_stopping"]
    use_next_period_stats = config["use_next_period_stats"]
    shuffle = config["shuffle"]
    shuffle_buffer_size = config["shuffle_buffer_size"]

    loss_params = config["loss_params"]
    optimizer_params = config["optimizer_params"].copy()

    # TODO: these model params are specific for NARX model, need to make it generic
    model_params = config["model_params"]

    callbacks = [ReportCheckpointCallback(
        metrics={"nrmse": "val_mean_window_nrmse"},
        report_metrics_on="epoch_end",
        checkpoint_on=[])  # because ray.train.tensorflow.TensorflowCheckpoint seems to not work correctly with subclassed keras models
    ]
    # callbacks = []

    model, _, _ = train_model(data_dir_path=data_dir_path, model_name=model_name, batch_size=batch_size,
                              input_window_size=input_window_size, output_window_size=output_window_size,
                              window_shift=window_shift, label_columns=label_columns,
                              past_exogenous_columns=past_exogenous_columns,
                              future_exogenous_columns=future_exogenous_columns, num_fourier_terms=num_fourier_terms,
                              epochs=epochs, loss_params=loss_params, optimizer_params=optimizer_params,
                              random_seed=random_seed, early_stopping=early_stopping,
                              use_next_period_stats=use_next_period_stats, shuffle=shuffle,
                              shuffle_buffer_size=shuffle_buffer_size,
                              fit_callbacks=callbacks,
                              **model_params)

    _, metrics = eval_model(data_dir_path, model, input_window_size, output_window_size, window_shift,
                            label_columns, past_exogenous_columns, future_exogenous_columns, num_fourier_terms,
                            use_next_period_stats=use_next_period_stats,
                            seasonality=seasonality,
                            random_seed=random_seed)

    return metrics.mean().to_dict()


def get_search_space(params: dict, same_units: bool) -> dict:
    non_tunable_parameters = {
        "random_seed": params["base"]["random_seed"],
        "num_fourier_terms": params["dataset"]["preprocess"]["num_fourier_terms"],
        "data_dir": pathlib.Path(params["base"]["data_dir"]).resolve(),
        "model_dir": pathlib.Path(params["base"]["model_dir"]).resolve(),
        "early_stopping": params["train"]["early_stopping"],
        "model_name": params["train"]["model"]["name"],
        "use_next_period_stats": params["train"]["next_period_stats"]["next_period_stats"]["use_next_period_stats"],
        "label_columns": params["train"]["model"]["label_columns"],
        "past_exogenous_columns": params["train"]["model"]["past_exogenous_columns"],
        "future_exogenous_columns": params["train"]["model"]["future_exogenous_columns"],
        "loss_params": params["train"]["loss"],
        "shuffle": params["train"]["shuffle"],
        "shuffle_buffer_size": params["train"]["shuffle_buffer_size"],
        "batch_size": params["train"]["batch_size"],
        "input_window_size": params["train"]["input_window_size"],
        "output_window_size": params["train"]["output_window_size"],
        "window_shift": params["train"]["window_shift"],
        "epochs": params["train"]["epochs"],
        "batch_size": params["train"]["batch_size"],
        "seasonality": params["eval"]["seasonality"],
    }

    search_space = {
        "num_layers": tune.randint(1, 3),
        "model_params": {
            "delay": params["train"]["model"]["model_params"]["delay"],
            "out_steps": params["train"]["model"]["model_params"]["out_steps"],
            "num_features": params["train"]["model"]["model_params"]["num_features"],
            "exogenous": params["train"]["model"]["model_params"]["exogenous"],
            "one_shot_steps": params["train"]["model"]["model_params"]["one_shot_steps"],
            "open_loop_training_mode": params["train"]["model"]["model_params"]["open_loop_training_mode"],
            "activation": params["train"]["model"]["model_params"]["activation"],
            "kernel_initializer": params["train"]["model"]["model_params"]["kernel_initializer"],
            "bias_initializer": params["train"]["model"]["model_params"]["bias_initializer"],
            "kernel_regularizer": {
                "module": "keras.regularizers",
                "class_name": "L1L2",
                "config": {
                    "l1": tune.uniform(0.0, 0.1),
                    "l2": tune.uniform(0.0, 0.1)
                }
            },
            "bias_regularizer": {
                "module": "keras.regularizers",
                "class_name": "L1L2",
                "config": {
                    "l1": tune.uniform(0.0, 0.1),
                    "l2": tune.uniform(0.0, 0.1)
                }
            }
        },
        "optimizer_params": {
            "learning_rate": {
                "name": "ExponentialDecay",
                "initial_learning_rate": tune.loguniform(1e-5, 1e-2),
                "decay_steps": params["train"]["optimizer_params"]["learning_rate"]["decay_steps"],
                "decay_rate": tune.uniform(0.85, 0.99),
                "staircase": False
            }},

    }

    if same_units:
        search_space["num_units"] = tune.randint(8, 256)
        search_space["model_params"]["units"] = tune.sample_from(
            lambda config: [config["num_units"]] * config["num_layers"])
    else:
        search_space["model_params"]["units"] = tune.sample_from(
            lambda config: [np.random.randint(8, 256) for _ in range(config["num_layers"])])

    search_space.update(non_tunable_parameters)

    return search_space


def tune_hyper_params(params: dict, same_units: bool, num_samples: int):

    search_space = get_search_space(params, same_units=same_units)

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=params["train"]["epochs"],
        grace_period=10,
        reduction_factor=2)

    tune_config = tune.TuneConfig(num_samples=num_samples,
                                  metric="nrmse",
                                  mode="min",
                                  scheduler=scheduler,)
    tuner = tune.Tuner(tune.with_resources(objective,
                                           {"cpu": 8,
                                            "gpu": 0.25}),
                       param_space=search_space,
                       tune_config=tune_config)

    results = tuner.fit()
    logger.info("Best hyperparameters found were: {}".format(
        results.get_best_result(scope="all").config))

    df_results = results.get_dataframe(
        filter_metric="nrmse", filter_mode="min")
    df_results.sort_values("nrmse", ascending=True, inplace=True)

    df_results.rename(columns={"config/model_params/units": "units",
                               "config/optimizer_params/learning_rate/initial_learning_rate": "initial_learning_rate",
                               "config/optimizer_params/learning_rate/decay_rate": "decay_rate",
                               "config/model_params/kernel_regularizer/config/l1": "kernel_regularizer_l1",
                               "config/model_params/kernel_regularizer/config/l2": "kernel_regularizer_l2",
                               "config/model_params/bias_regularizer/config/l1": "bias_regularizer_l1",
                               "config/model_params/bias_regularizer/config/l2": "bias_regularizer_l2",
                               }, inplace=True)
    logger.info(f"Results: {df_results.loc[:, ["nrmse", "units",
                                               "initial_learning_rate", "decay_rate",
                                               "kernel_regularizer_l1", "kernel_regularizer_l2",
                                               "bias_regularizer_l1", "bias_regularizer_l2",]]}")


if __name__ == "__main__":
    util_logging.setup_logging(logfilename_prefix="ray_tune_narx")

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument(
        '--same_units', dest='same_units', required=False, action='store_true'
    )
    args_parser.add_argument(
        '--num_samples', dest='num_samples', required=False, type=int, default=200)
    args = args_parser.parse_args()

    logger.info(f"Starting hyperparameter tuning with Ray Tune")
    logger.info(f"Loading config from {args.config}")
    config_path = pathlib.Path(args.config).resolve()
    params = load_config(config_path=config_path)

    tune_hyper_params(params, same_units=args.same_units,
                      num_samples=args.num_samples)
