import os
import yaml


def load_config(config_path: str | os.PathLike):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    return params
