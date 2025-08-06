
# LSTM PV Forecasting



This repository provides the code for the paper:

> Giuseppe La Tona and Maria Carmela Di Piazza, "Enhanced Day-Ahead Solar Irradiance Forecasting Using a Modified LSTM Encoder-Decoder: A Deep Learning vs. Hybrid Machine Learning Comparison," *submitted to IEEE Access*, 2025.

**Authors Affiliation:**
Institute of Marine Engineering (INM), National Research Council, Palermo, Italy

This project is intended for research on solar radiance forecasting using artificial neural networks (ANNs). It compares several ANN architectures for solar forecasting, and in particular introduces a modified LSTM encoder-decoder model that supports static exogenous variables, enabling the use of next-day statistics from third-party weather forecasts.



## Table of Contents

- [Using a VS Code Devcontainer (Recommended)](#using-a-vs-code-devcontainer-recommended)
- [Installation](#installation)
- [Models Summary](#models-summary)
- [Datasets](#datasets)
- [Usage](#usage)
- [Experiment Configuration](#experiment-configuration)
- [Reproducing Published Results](#reproducing-published-results)
- [Tuning](#tuning)
- [Testing](#testing)
- [Performance benchmark](#performance-benchmark)
- [Using the `generate_plots.py` Script](#using-the-generate_plots.py-script)
- [Citation](#citation)


## Using a VS Code Devcontainer (Recommended)

As an alternative to installing dependencies locally, you can use this repository inside a [Visual Studio Code Devcontainer](https://code.visualstudio.com/docs/devcontainers/containers). This provides a fully configured, reproducible development environment with all required dependencies pre-installed.

### Quick Start

1. Make sure you have [Docker](https://www.docker.com/) and [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed.
2. Open the repository folder in VS Code.
3. When prompted, reopen the folder in the devcontainer, or use the command palette: `Dev Containers: Reopen in Container`.
4. The environment will be built automatically and you can start working immediately, with all dependencies available.

This approach is recommended for ensuring consistency and avoiding dependency conflicts on your local machine.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/giulatona/lstm_pv_forecasting.git
```

2. Navigate to the project directory:
```bash
cd lstm_pv_forecasting
```

3. Install the dependencies using pipenv in editable mode:
```bash
pipenv install --dev
```


## Models Summary

The following models are available in this repository:

- **DeepNextDayStat** (Modified LSTM Encoder-Decoder, *M-LSTM-ED* in the paper):
  The model proposed in the paper. An LSTM encoder-decoder with exogenous inputs architecture extended to include static exogenous variables 
  (e.g., next-day statistics from weather forecasts) concatenated before the decoder.

- **Lstm** (LSTM Encoder-Decoder, *LSTM-ED* in the paper):
  LSTM encoder-decoder with exogenous inputs for time series forecasting.

- **NARX** (*NARX* in the paper):
  Nonlinear AutoRegressive model with eXogenous inputs, a hybrid neural network for time series forecasting with exogenous variables.

- **Naive** (Naive seasonal, *Naive seasonal* in the paper):
  Baseline model that repeats the value from the same hour of the previous day (seasonal naive forecast).

Refer to the experiment configuration and the paper for details on how each model is used and evaluated.

## Datasets

This project uses two meteorological datasets from locations with different climatic characteristics: one from Palermo, Italy (SIAS), and one from Sceaux, France (PVGIS). Both datasets provide hourly solar irradiance and temperature measurements over multiple years.

### SIAS dataset (Palermo, Italy)

This dataset was provided by the Agrometeorological Information Service of Sicily (SIAS) and refers to a location in Palermo, southern Italy. It contains hourly global solar irradiation (MJ/m²) and hourly maximum and minimum temperatures measured from 2001-01-01 to 2008-12-31 (seven consecutive years). The mean hourly temperatures were calculated for use in this project.

**Download instructions:**
The SIAS dataset is publicly available from the Agrometeorological Information Service of Sicily (SIAS) at http://www.sias.regione.sicilia.it. Access requires user registration and agreement to the provider’s terms of use.

**Important:** After downloading, the SIAS dataset file must be saved to the path `data/sias_palermo_radiance.csv` as required by the configuration files. 

### PVGIS dataset (EUPVGIS_sceaux_2006-2016.csv)
This repository includes data from the Photovoltaic Geographical Information System (PVGIS), provided by the European Commission, Joint Research Centre (JRC).

The PVGIS dataset refers to a station located in Sceaux, France. It contains hourly global horizontal irradiance (GHI, W/m²) and hourly mean temperature measurements recorded over nine consecutive years (2006–2016).

This repository includes the following PVGIS data file, provided by the European Commission, Joint Research Centre (JRC):
- File: `data/EUPVGIS_sceaux_2006-2016.csv`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
  - https://creativecommons.org/licenses/by/4.0/
- Source: https://re.jrc.ec.europa.eu/pvg_tools/en/

Please cite as:
European Commission, Joint Research Centre (JRC), PVGIS Photovoltaic Geographical Information System (2025). https://re.jrc.ec.europa.eu/pvg_tools/en/

## Usage

This project used [DVC](https://dvc.org/) to run ML pipelines and to track experiments. To run an experiment:
```bash
pipenv run dvc exp run
```

The project uses [Hydra](https://hydra.cc/) for configuration. 
```bash
pipenv run dvc exp run --set-param 'param_name=param_value'
```

To change group defaults from command line groups names are separated using "/" instead of ".".
```bash
pipenv run dvc exp run --set-param 'train/model=lstm'
```

DVC experiments can be run without affecting your workspace by using the `--temp` flag. This will create a temporary directory for the experiment, ensuring that your current workspace remains unchanged.

To run an experiment in a temporary directory:
```bash
pipenv run dvc exp run --temp
```

Consequently DVC runs the experiment inside it's own temporary copy of the workspace. The same applies when using --jobs with exp run/queue. However, it is important to use relative paths in the code.

A DVC experiment is a versioned experiment that is tracked by DVC. Each experiment is associated with a Git reference, allowing you to easily reproduce, compare, and share your machine learning experiments. This integration with Git ensures that your experiments are version-controlled and can be managed alongside your codebase. See [Git Custom References for ML Experiments](https://datachain.ai/blog/experiment-refs) for further details.

Note: if dvc fails to refresh gdrive tocken remove default.json from subfolder of $HOME/.cache/pydrive2fs/

## Experiment Configuration

The main experiment parameters are defined in the YAML files inside the `config/` folder. The `params.yaml` file is generated automatically by [Hydra](https://hydra.cc/) and should not be edited directly. Key parameters include:

- **Model type:** Selectable via `train/model` the value must be set equal to the name of the yaml file inside train/model (e.g., `lstm`, `narx`, etc.) 
- **Input window size:** Number of past time steps used as input (`window_size`)
- **Forecast horizon:** Number of time steps to predict ahead (`forecast_horizon`)
- **Batch size:** Number of samples per training batch (`batch_size`)
- **Epochs:** Number of training epochs (`epochs`)
- **Learning rate:** Optimizer learning rate (`learning_rate`)
- **Features:** Input features used (irradiance, temperature, exogenous variables, etc.)
- **Static exogenous variables:** Next-day statistics from weather forecasts (enabled in modified LSTM model)
- **Dataset selection:** The dataset is selected in `config.yaml` (see the `dataset` field), and dataset-specific parameters are defined in the corresponding YAML file inside `config/dataset/`.
- **Random seed:** For reproducibility (`seed`)

You can modify these parameters in the YAML files under `config/`, or override them from the command line using DVC and Hydra, for example:
```bash
pipenv run dvc exp run --set-param 'train/model=lstm' --set-param 'train.epochs=50'
```

Refer to the configuration files in the `config/` folder for further details on each parameter.

### Example: Running Experiments with Different Datasets

To run an experiment using the SIAS (Palermo, Italy) dataset:
```bash
pipenv run dvc exp run --set-param 'dataset=sias_palermo' --set-param 'train/model=lstm'
```

To run an experiment using the PVGIS (Sceaux, France) dataset:
```bash
pipenv run dvc exp run --set-param 'dataset=eu_pvgis' --set-param 'train/model=lstm'
```

You can further customize the experiment by overriding other parameters as needed. See the `config.yaml` and files in `config/dataset/` for available dataset options and parameters.



## Reproducing Published Results

To reproduce the results published in the paper, follow these steps:

1. Ensure you have all datasets available as described in the [Datasets](#datasets) section.

2. Use the automated reproduction script `src/lstm_pv_forecasting/scripts/reproduce_paper_results.py` with the provided configuration files:
   - `paper_results_pvgis_config.yaml` - For PVGIS (Sceaux, France) dataset experiments
   - `paper_results_sias_config.yaml` - For SIAS (Palermo, Italy) dataset experiments

3. Run the reproduction script for each dataset:
   ```bash
   # Execute PVGIS experiments
   pipenv run python src/lstm_pv_forecasting/scripts/reproduce_paper_results.py --config paper_results_pvgis_config.yaml
   
   # Execute SIAS experiments
   pipenv run python src/lstm_pv_forecasting/scripts/reproduce_paper_results.py --config paper_results_sias_config.yaml
   ```

4. After completing the experiments, reproduce the same figures shown in the paper using the `generate_plots.py` script:
   ```bash
   # Generate plots for test results (PVGIS dataset)
   pipenv run python src/lstm_pv_forecasting/scripts/generate_plots.py --config test_results_pvgis_plot_config.yaml
   
   # Generate plots for test results (SIAS dataset)
   pipenv run python src/lstm_pv_forecasting/scripts/generate_plots.py --config test_results_sias_plot_config.yaml
   ```

The configuration files contain the exact experiment setups and git commit hashes used in the paper, and the plot configuration files are set up to generate the same visualizations presented in the publication.

If you encounter any issues during reproduction, please open an issue or contact the authors.

## Tuning

There is a different tuning script for each considered model architecture at this time. Before running it, the params.yaml file must be updated
running `dvc exp run`

```bash
pipenv run dvc exp run --set-param 'dataset=eu_pvgis' --set-param 'train/model=lstm'
pipenv run python src/lstm_pv_forecasting/hptuning/ray_tune_lstm.py --config params.yaml --num_samples 100
```

To run multiple trials in parallel execute:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

This is necessary also for running multiple dvc jobs in parallel.

### Analyzing Ray Tune Results

#### Using TensorBoard

Ray Tune integrates seamlessly with TensorBoard for visualizing experiment results. To start TensorBoard and visualize the results:

1. Navigate to the directory where your Ray Tune results are stored:
    ```bash
    cd ray_results_path
    ```

2. Start TensorBoard:
    ```bash
    tensorboard --logdir=./
    ```

3. Open your web browser and go to `http://localhost:6006` to view the TensorBoard dashboard.

#### Analyzing nohup.out Log

The `nohup.out` file captures the standard output and error streams of your Ray Tune experiments when run with `nohup`. To analyze the log:

1. Open the `nohup.out` file using a text editor or command-line tool:
    ```bash
    less nohup.out
    ```

2. Look for key metrics, errors, and other relevant information to understand the experiment's progress and results.

#### Restoring the Tuner Object

If you need to restore a Ray Tune Tuner object to resume or analyze an experiment, you can do so using the following steps:

1. Import the necessary modules:
    ```python
    from ray.tune import Tuner
    from lstm_pv_forecasting.hptuning.ray_tune_lstm import objective
    ```

2. Restore the Tuner object from the checkpoint directory:
    ```python
    tuner = Tuner.restore("path/to/ray_results", trainable=objective)
    ```

    (the path to ray results can be retrieved from the log)

3. You can now resume the experiment or analyze the results using the restored Tuner object:
    ```python
    results = tuner.get_results()
    ```

Check out [Analyzing Tune Experiment Results](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html#analyzing-tune-experiment-results) for further details.

By following these steps, you can effectively analyze your Ray Tune experiment results using TensorBoard, the `nohup.out` log, and restore the Tuner object if needed.

## Testing
Run tests using pytest

Check this [link](https://jasonstitt.com/perfect-python-live-test-coverage) for adding live update to vscode editor.

## Performance benchmark

To measure training time run benchmark

```bash
pipenv run python src/lstm_pv_forecasting/scripts/training_benchmark.py --device gpu --epochs 5 --batch_size 128 
```

To use Tensorboard profiler

```bash
pipenv run python src/lstm_pv_forecasting/scripts/training_benchmark.py --device gpu --epochs 5 --batch_size 128 --profile
```

Note: in case of errors due to not found libcupti add `{env_path}/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/` to `LD_LIBRARY_PATH`
```bash
source configure_cupti_lib_path.sh
```

## Using the `generate_plots.py` Script

The `generate_plots.py` script is used to generate line plots and scatter plots for solar irradiance predictions. The script reads configuration details from a YAML file and produces SVG plots in the `results_plot/` directory.

### Prerequisites
- Ensure you have all dependencies installed as specified in the `Pipfile`.
- The DVC experiments and data files must be properly set up in your workspace.

### Configuration
Create a YAML configuration file specifying the periods, experiments, times, and seasons for the plots. An example configuration file is provided as `plot_config_example.yaml` in the root of the workspace.

### Usage
Run the script using the following command:

```bash
python src/lstm_pv_forecasting/scripts/generate_plots.py --config <path_to_config_yaml>
```

Replace `<path_to_config_yaml>` with the path to your YAML configuration file.

### Example
To use the provided example configuration file:

```bash
python src/lstm_pv_forecasting/scripts/generate_plots.py --config plot_config_example.yaml
```

### Output
The script generates:
- Line plots for the specified periods, saved as `line_plot0.svg`, `line_plot1.svg`, etc., in the `results_plot/` directory.
- Scatter plots for each season, saved as `summer_scatter_plots.svg`, `fall_scatter_plots.svg`, etc., in the `results_plot/` directory.

## Citation

If you use this code or results in your research, please cite the following paper:

> Giuseppe La Tona and Maria Carmela Di Piazza, "Enhanced Day-Ahead Solar Irradiance Forecasting Using a Modified LSTM Encoder-Decoder: A Deep Learning vs. Hybrid Machine Learning Comparison," *submitted to IEEE Access*, 2025.