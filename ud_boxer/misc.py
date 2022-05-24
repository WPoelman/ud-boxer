import json
import pickle
from os import PathLike
from pathlib import Path

__all__ = [
    "ensure_ext",
]


def ensure_ext(path: PathLike, extension: str) -> Path:
    """Make sure a path ends with a desired file extension."""
    return (
        Path(path)
        if str(path).endswith(extension)
        else Path(f"{path}{extension}")
    )


def load_pickle(path):
    with open(path, "rb") as f:
        content = pickle.load(f)
    return content


def load_json(path):
    with open(path, "rb") as f:
        content = json.load(f)
    return content
