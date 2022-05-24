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
