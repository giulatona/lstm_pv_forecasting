import argparse
import io

from dvc.api import DVCFileSystem
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from sklearn.metrics import r2_score
import yaml


def plot_scatter(predictions_df, experiments_to_plot, times_to_plot, season_period, output_file):
    start_date, end_date = season_period
    restricted_predictions_df = predictions_df.loc[start_date:end_date]

    # Filter experiments to plot to only those with scatter_plot set to True
    experiments_to_plot = {
        k: v for k, v in experiments_to_plot.items() if v.get("scatter_plot", False)}

    # Create a figure with subplots, one row per time of day
    fig, axes = plt.subplots(len(times_to_plot), len(
        experiments_to_plot), figsize=(18.2 / 2.54, 6 / 2.54 * len(times_to_plot)))

    # Define a color map for experiments
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / (len(experiments_to_plot) - 1))
              for i in range(len(experiments_to_plot))]

    # Calculate the same limits for both axes
    min_limit = float('inf')
    max_limit = float('-inf')
    for experiment in experiments_to_plot.keys():
        min_limit = min(min_limit, restricted_predictions_df["observations"].min(
        ), restricted_predictions_df[experiment].min())
        max_limit = max(max_limit, restricted_predictions_df["observations"].max(
        ), restricted_predictions_df[experiment].max())

    # Iterate over times and axes to plot scatter plots
    for ax_row, time_to_plot in zip(axes, times_to_plot):
        for ax, (experiment, spec), color in zip(ax_row, experiments_to_plot.items(), colors):
            # Get the time index for the experiment
            time_index = restricted_predictions_df.index[restricted_predictions_df.index.strftime(
                '%H:%M') == time_to_plot]
            restricted_predictions_df.loc[time_index].plot.scatter(
                x="observations", y=experiment, label=spec["label"], ax=ax, color=color)
            ax.set_ylabel("Predictions")
            ax.set_xlabel("Observations")
            ax.text(0.05, 0.88, f"Time: {time_to_plot}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')
            ax.legend(loc="upper left")
            # Calculate R^2
            r2 = r2_score(y_true=restricted_predictions_df.loc[time_index, "observations"],
                          y_pred=restricted_predictions_df.loc[time_index, experiment])
            ax.text(0.05, 0.80, f"R²: {r2:.2f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment='top')

            # Set axis limits
            ax.set_xlim(min_limit, max_limit)
            ax.set_ylim(min_limit, max_limit)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file, format='svg')


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate scatter plots for predictions.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    periods_to_plot = config['periods_to_plot']
    experiments_to_plot = config['experiments_to_plot']

    # Load predictions data
    results_list = []
    for exp_name, spec in experiments_to_plot.items():
        fs = DVCFileSystem(url=".", rev=exp_name)
        with fs.open("data/prediction_results.pkl", 'rb') as f:
            data = f.read()
            if isinstance(data, bytes):
                results = pd.read_pickle(io.BytesIO(data))
            else:
                raise ValueError(
                    "Expected bytes from file read, got something else.")
        results_list.append(results)

    predictions_df = pd.DataFrame({
        exp_name: result['predictions'] for exp_name, result in zip(experiments_to_plot.keys(), results_list)
    })

    # Align data indices to handle cases where an experiment has fewer data points
    min_index = predictions_df.index
    for result in results_list:
        min_index = min_index.intersection(result.index)

    # Trim all dataframes to the overlapping index
    predictions_df = predictions_df.loc[min_index]
    for exp_name, result in zip(experiments_to_plot.keys(), results_list):
        predictions_df[exp_name] = result.loc[min_index, 'predictions']

    # Load predictions data from extern experiments (from Matlab code)
    if "extern" in config:
        filename = config["extern"]["filename"]
        date_format = config["extern"]["date_format"]
        results = pd.read_csv(filename, index_col=0,
                              parse_dates=True, date_format=date_format)
        results = results.rename(columns={
            "radiance": "observations", "forecast": "predictions"})
        predictions_df["extern"] = results.loc[min_index, "predictions"]
        experiments_to_plot["extern"] = config["extern"]

    predictions_df.loc[:,
                       "observations"] = results_list[0].loc[min_index, 'observations']

    # Convert MJ/m² to W/m²
    predictions_df *= 1e6 / 3600

    plt.style.use(['science'])
    plt.rcParams.update({
        "font.family": "serif",   # specify font family here
        "font.serif": ["Times"],  # specify font here
        "font.size": 8,  # specify font size here
        "figure.figsize": [14 / 2.54, 7 / 2.54],
        "figure.dpi": 200.0
    })

    # line plots
    for count, period in enumerate(periods_to_plot):
        start = period["start"]
        end = period["end"]

        fig, ax = plt.subplots(figsize=(18.2 / 2.54, 7 / 2.54))

        # Define line styles and colors
        line_styles = ['-', '--', '-.', ':']
        cmap = plt.get_cmap('Accent')
        colors = [cmap(i) for i in range(len(cmap.colors))]

        # Track used colors to avoid repetition
        used_colors = {}

        # Add observations to the plot
        predictions_df.loc[start:end, "observations"].plot(
            ax=ax, linewidth=1.5, linestyle='-', color='black')

        # Highlight experiments with 'highlight' set to True
        for idx, (exp_name, exp_config) in enumerate(experiments_to_plot.items()):
            color = colors[idx % len(colors)]
            linestyle = '-'

            # Check if the color is already used
            if color in used_colors:
                linestyle = line_styles[used_colors[color] % len(line_styles)]
                used_colors[color] += 1
            else:
                used_colors[color] = 1

            if exp_config.get("highlight", False):
                predictions_df.loc[start:end, exp_name].plot(
                    ax=ax, linewidth=1.5, linestyle=linestyle, color=color)
            else:
                predictions_df.loc[start:end, exp_name].plot(
                    ax=ax, linewidth=0.75, linestyle=linestyle, color=color, alpha=0.8)

        ax.legend(
            ["observations"] + [exp["label"]
                                for exp in experiments_to_plot.values()],
            loc="upper center",  # Position the legend above the plot
            # Adjust the position (centered horizontally, below the plot)
            bbox_to_anchor=(0.8, -0.15),
            ncol=2,  # Number of columns in the legend
            frameon=False  # Optional: remove the legend box
        )
        ax.set_ylabel("Solar Irradiance (W/m²)")

        # Save the line plot
        fig.savefig(f"results_plot/line_plot{count}.svg", format='svg')

    # Generate plots for each season
    for season, period in config['seasons'].items():
        output_file = f"results_plot/{season}_scatter_plots.svg"
        plot_scatter(predictions_df, experiments_to_plot,
                     config['times_to_plot'], period, output_file)


if __name__ == "__main__":
    main()
