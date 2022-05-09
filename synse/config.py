from pathlib import Path
from typing import Dict

from synse.base import BaseEnum

__all__ = [
    "Config",
]

PROJECT_ROOT = Path(__file__).parent.parent


class Config:
    """Class to house project-wide constants and config items"""

    # -- Paths --
    GRS_PATH = Path(PROJECT_ROOT / "grew/main.grs").resolve()
    EDGE_MAPPINGS_PATH = Path(
        PROJECT_ROOT / "data/mappings/edge_mappings.json"
    ).resolve()
    LEMMA_SENSE_MAPPINGS_PATH = Path(
        PROJECT_ROOT / "data/mappings/lemma_sense_lookup_en_gold_train.pickle"
    ).resolve()
    LEMMA_POS_SENSE_MAPPINGS_PATH = Path(
        PROJECT_ROOT
        / "data/mappings/lemma_pos_sense_lookup_en_gold_train.pickle"
    ).resolve()
    SPLIT_DIR_PATH = Path(PROJECT_ROOT / "data/splits").resolve()
    EDGE_CLF_PATH = Path(
        PROJECT_ROOT / "data/edge_classifier/edge_clf.joblib"
    ).resolve()

    # -- Enums --
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

    # -- Defaults --
    DEFAULT_BOX_CONNECT = "CONTINUATION"
    DEFAULT_ROLE = "Agent"
    DEFAULT_TIME_ROLE = "EQU"
    DEFAULT_GENDER = "person.n.01"
