"""
Config loader — reads YAML, returns a dot-accessible namespace.
"""
import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(i) for i in d]
    return d


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


def config_to_dict(ns) -> dict:
    if isinstance(ns, SimpleNamespace):
        return {k: config_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [config_to_dict(i) for i in ns]
    return ns
