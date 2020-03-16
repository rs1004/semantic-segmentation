import json
import os
from pathlib import Path


def set_config():
    with open((Path(__file__).parent / 'config.json').resolve(), 'r') as f:
        envs = json.load(f)

    for k, v in envs.items():
        if isinstance(v, int):
            v = str(v)
        os.environ[k] = v
