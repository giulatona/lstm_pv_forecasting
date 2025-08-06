import logging
import logging.config
import pathlib
import yaml


CONFIG_DIR = "./config"
LOG_DIR = "./logs"


def setup_logging(logfilename_prefix: str = ""):
    """Load logging configuration"""

    config_path = pathlib.Path(CONFIG_DIR).joinpath("logging.yaml").resolve()

    # Load the config file
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    log_dir_path = pathlib.Path(LOG_DIR).resolve()
    log_dir_path.mkdir(exist_ok=True)

    config["handlers"]["file"]["filename"] = log_dir_path.joinpath(
        f"{logfilename_prefix}.log")

    # Configure the logging module with the config file
    logging.config.dictConfig(config)
