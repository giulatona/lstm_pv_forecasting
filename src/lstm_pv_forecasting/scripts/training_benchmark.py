import argparse
from datetime import datetime
import logging
import pathlib
import tempfile
import time

import keras
from keras.callbacks import Callback
import tensorflow as tf

from lstm_pv_forecasting.stages.train import train_model
from lstm_pv_forecasting.util import logging as util_logging
from lstm_pv_forecasting.util.load_config import load_config

logger = logging.getLogger(__name__)


class TimingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.train_begin_time = time.perf_counter()
        self.times = []
        self.total_training_time = None

    def on_train_end(self, logs=None):
        elapsed_time = time.perf_counter() - self.train_begin_time
        self.total_training_time = elapsed_time
        logger.info(f"Total training time: {elapsed_time:.6f} seconds")
        logger.info(
            f"Average epoch time: {sum(self.times) / len(self.times):.6f} seconds")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.epoch_time_start)


class PredictionTimingCallback(Callback):
    def on_predict_begin(self, logs=None):
        self.start_time_ns = time.perf_counter_ns()
        self.predict_batch_times_ns = []

    def on_predict_end(self, logs=None):
        self.end_time_ns = time.perf_counter_ns()
        self.elapsed = (self.end_time_ns - self.start_time_ns) / 1e9  # seconds

    def on_predict_batch_begin(self, batch, logs=None):
        self.batch_time_start_ns = time.perf_counter_ns()

    def on_predict_batch_end(self, batch, logs=None):
        batch_time_ns = time.perf_counter_ns() - self.batch_time_start_ns
        self.predict_batch_times_ns.append(batch_time_ns)

    @property
    def predict_batch_times(self):
        # Return batch times in seconds
        return [t / 1e9 for t in self.predict_batch_times_ns]


def main(device="gpu", verbose=False, profile=False, profile_batch=(2, 5), n_repeats=1):

    if profile:
        raise NotImplementedError("Profiling is not implemented yet.")

    """
    Benchmark training time using train_model from train.py, reading config from params.yaml.

    Args:
        device (str, optional): The device to use for training. Must be either "gpu" or "cpu". Defaults to "gpu".
        verbose (bool, optional): Whether to print training progress. Defaults to False.
        profile (bool, optional): Whether to enable profiling during training. Defaults to False.
        profile_batch (tuple, optional): The batch range to use for profiling. Defaults to (2, 5).

    Returns:
        None
    """
    # Load configuration from params.yaml
    config_path = pathlib.Path("params.yaml")
    params = load_config(config_path)

    # Extract relevant parameters as in train.py
    data_dir = pathlib.Path(params["base"]["data_dir"])
    model_name = params["train"]["model"]["name"]
    model_params = params["train"]["model"]["model_params"]
    batch_size = params["train"]["batch_size"]
    epochs = params["train"]["epochs"]
    input_window_size = params["train"]["input_window_size"]
    output_window_size = params["train"]["output_window_size"]
    window_shift = params["train"]["window_shift"]
    label_columns = params["train"]["model"]["label_columns"]
    past_exogenous_columns = params["train"]["model"]["past_exogenous_columns"]
    future_exogenous_columns = params["train"]["model"]["future_exogenous_columns"]
    num_fourier_terms = params["dataset"]["preprocess"]["num_fourier_terms"]
    loss_params = params["train"]["loss"]
    optimizer_params = params["train"]["optimizer_params"]
    random_seed = params["base"]["random_seed"]
    early_stopping = params["train"]["early_stopping"]
    use_next_period_stats = params["train"]["next_period_stats"]["next_period_stats"]["use_next_period_stats"]
    shuffle = params["train"].get("shuffle", True)
    shuffle_buffer_size = params["train"].get("shuffle_buffer_size", None)

    prediction_times = []
    total_training_times = []
    all_inference_batch_times = []  # Store all batch times across all runs
    all_epoch_times = []  # Store all epoch times across all runs

    for repeat in range(n_repeats):
        logger.info(f"=== Run {repeat + 1}/{n_repeats} ===")
        # Prepare callbacks
        fit_callbacks = []
        timing_callback = TimingCallback()
        fit_callbacks.append(timing_callback)

        if profile:
            tmpdirPath = pathlib.Path(
                tempfile.gettempdir()) / "tensorboard_logs"
            tmpdirPath.mkdir(parents=True, exist_ok=True)
            tmpdirPath = tmpdirPath.joinpath(
                f"lstm_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            print(f"TensorBoard logs will be stored in {tmpdirPath}")
            tboard_callback = keras.callbacks.TensorBoard(
                log_dir=str(tmpdirPath),
                histogram_freq=1,
                profile_batch=profile_batch
            )
            fit_callbacks.append(tboard_callback)

        # Set device context based on the device parameter
        if device == "gpu":
            device_spec = "/device:GPU:0"
        elif device == "cpu":
            device_spec = "/device:CPU:0"
        else:
            raise ValueError("device must be either 'gpu' or 'cpu'")

        # Start timing
        start_time = time.perf_counter()
        with tf.device(device_spec):
            copy_optimizer_params = optimizer_params.copy()
            model, _, _ = train_model(
                data_dir_path=data_dir,
                model_name=model_name,
                batch_size=batch_size,
                input_window_size=input_window_size,
                output_window_size=output_window_size,
                window_shift=window_shift,
                label_columns=label_columns,
                past_exogenous_columns=past_exogenous_columns,
                future_exogenous_columns=future_exogenous_columns,
                num_fourier_terms=num_fourier_terms,
                epochs=epochs,
                loss_params=loss_params,
                optimizer_params=copy_optimizer_params,
                random_seed=random_seed,
                early_stopping=early_stopping,
                use_next_period_stats=use_next_period_stats,
                shuffle=shuffle,
                shuffle_buffer_size=shuffle_buffer_size,
                fit_callbacks=fit_callbacks,
                **model_params
            )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(
            f"Outer training function elapsed time: {elapsed_time:.6f} seconds")
        all_epoch_times.extend(timing_callback.times)
        if hasattr(timing_callback, "total_training_time") and timing_callback.total_training_time is not None:
            total_training_times.append(timing_callback.total_training_time)
        else:
            total_training_times.append(elapsed_time)

        # Inference timing using PredictionTimingCallback
        logger.info("Running eval_model for inference timing...")
        try:
            from lstm_pv_forecasting.stages.eval import eval_model
            prediction_timing_callback = PredictionTimingCallback()
            _pred, _metrics = eval_model(
                data_dir,
                model,
                input_window_size,
                output_window_size,
                window_shift,
                label_columns,
                past_exogenous_columns,
                future_exogenous_columns,
                num_fourier_terms,
                params["eval"]["seasonality"],
                use_next_period_stats=use_next_period_stats,
                random_seed=random_seed,
                eval_on_validation_set=True,
                callbacks=[prediction_timing_callback]
            )
            if hasattr(prediction_timing_callback, "elapsed") and prediction_timing_callback.elapsed is not None:
                logger.info(
                    f"Inference (eval_model) completed in {prediction_timing_callback.elapsed:.6f} seconds")
                prediction_times.append(prediction_timing_callback.elapsed)
                # Store all batch times for this run
                all_inference_batch_times.extend(
                    prediction_timing_callback.predict_batch_times)
            else:
                logger.info("Inference timing not available")
                prediction_times.append(float('nan'))
        except Exception as e:
            logger.error(f"Could not run eval_model for inference timing: {e}")
            prediction_times.append(float('nan'))

    # Report statistics
    import numpy as np
    logger.info("=== Timing statistics over all runs ===")
    logger.info(
        f"Training total time: mean={np.nanmean(total_training_times):.6f} s, std={np.nanstd(total_training_times):.6f} s")
    if all_epoch_times:
        logger.info(
            f"Training epoch time: mean={np.nanmean(all_epoch_times):.6f} s, std={np.nanstd(all_epoch_times):.6f} s")
    logger.info(
        f"Inference time: mean={np.nanmean(prediction_times):.6f} s, std={np.nanstd(prediction_times):.6f} s")
    if all_inference_batch_times:
        logger.info(
            f"Inference time per batch: mean={np.nanmean(all_inference_batch_times):.6f} s, std={np.nanstd(all_inference_batch_times):.6f} s")

    # Display model summary from the last run
    print("\n=== Model Summary (last run) ===")
    try:
        model.summary()
    except Exception as e:
        print(f"Could not display model summary: {e}")


if __name__ == '__main__':
    util_logging.setup_logging(logfilename_prefix="train")
    parser = argparse.ArgumentParser(
        description='Benchmarks training time of lstm encoder-decoder using train_model.')
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu", help="device type")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="whether to store tensorboard logs for profiling")
    parser.add_argument("--profile_batch", type=int, nargs=2, metavar=('START', 'END'),
                        default=(2, 5), help="batch range to profile (default: 2 5)")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="number of times to repeat the training and evaluation")
    args = parser.parse_args()

    main(device=args.device, verbose=args.verbose,
         profile=args.profile, profile_batch=tuple(args.profile_batch), n_repeats=args.n_repeats)
