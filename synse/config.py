from pathlib import Path
from typing import Dict

from synse.base import BaseEnum

__all__ = [
    "Config",
]

PROJECT_ROOT = Path(__file__).parent.parent


class Config:
    GRS_PATH = Path(PROJECT_ROOT / "grew/main.grs").resolve()

    EDGE_MAPPINGS_PATH = Path(
        PROJECT_ROOT / "data/output/edge_mappings.json"
    ).resolve()

    SPLIT_DIR_PATH = Path(PROJECT_ROOT / "data/splits").resolve()

    class SUPPORTED_LANGUAGES(BaseEnum):
        """Supported languages, this is based on the PMB 4.0.0"""

        NL = "nl"
        DE = "de"
        IT = "it"
        EN = "en"

    class DATA_SPLIT(BaseEnum):
        """Data splits based on the PMB 4.0.0"""

        DEV = "dev"
        EVAL = "eval"
        TEST = "test"
        TRAIN = "train"

    class UD_SYSTEM(BaseEnum):
        """Supported UD parsers"""

        STANZA = "stanza"
        TRANKIT = "trankit"

    # Used to switch between stanza and trankit language identifiers
    UD_LANG_MAPPING = {
        "de": "german",
        "en": "english",
        "it": "italian",
        "nl": "dutch",
    }
