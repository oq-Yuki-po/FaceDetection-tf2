from typing import Dict

import yaml


def load_yaml(path: str) -> Dict:
    with open(path, mode="r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config
