from pathlib import Path

import joblib

from ud_boxer.base import BaseEnum
from ud_boxer.misc import load_json

__all__ = [
    "Config",
]


class Config:
    """Class to house project-wide constants and config items"""

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
        EVAL = "eval"  # Eval is only available for English!
        TEST = "test"
        TRAIN = "train"
        ALL = "all"

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

    # -- Paths --
    PROJECT_ROOT = Path(__file__).parent.parent
    GRS_PATH = Path(PROJECT_ROOT / "grew/main.grs").resolve()
    DATA_DIR = PROJECT_ROOT / "data"
    MAPPINGS_DIR = DATA_DIR / "mappings"
    LOG_PATH = Path(DATA_DIR / "logs").resolve()
    # NOTE: seq2seq is currently only for english, so keep it that way
    SEQ2SEQ_DIR = Path(DATA_DIR / "results/seq2seq").resolve()

    @staticmethod
    def get_result_dir(lang: SUPPORTED_LANGUAGES, data_split: DATA_SPLIT):
        path = Path(
            Config.DATA_DIR / f"results/rewrite/{lang}/{data_split}"
        ).resolve()
        path.mkdir(exist_ok=True)
        return path

    @staticmethod
    def get_edge_mappings(lang: SUPPORTED_LANGUAGES):
        # TODO: this is currently language neutral since it only works with
        # UPOS and DEPRELs, maybe in the future language specific mappings can
        # be used or everything can be extracted across all training sets of
        # all languages.
        path = Path(
            Config.MAPPINGS_DIR / f"en_edge_mappings_train.json"
        ).resolve()
        return load_json(path)

    @staticmethod
    def get_lemma_sense(lang: SUPPORTED_LANGUAGES):
        path = Path(
            Config.DATA_DIR / f"mappings/{lang}_lemma_sense_lookup_train.json"
        ).resolve()
        return load_json(path)

    @staticmethod
    def get_lemma_pos_sense(lang: SUPPORTED_LANGUAGES):
        path = Path(
            Config.DATA_DIR
            / f"mappings/{lang}_lemma_pos_sense_lookup_train.json"
        ).resolve()
        return load_json(path)

    @staticmethod
    def get_edge_clf(lang: SUPPORTED_LANGUAGES):
        path = Path(
            Config.DATA_DIR / f"edge_classifier/{lang}_edge_clf.joblib"
        ).resolve()
        return joblib.load(path)

    @staticmethod
    def get_split_ids(lang: SUPPORTED_LANGUAGES, split: DATA_SPLIT):
        path = Path(Config.DATA_DIR / f"splits/{lang}_{split}.txt").resolve()
        return path.read_text().rstrip().split("\n")

    # -- Defaults --
    DEFAULT_BOX_CONNECT = "CONTINUATION"
    DEFAULT_ROLE = "Agent"
    DEFAULT_TIME_ROLE = "EQU"
    DEFAULT_GENDER = "person.n.01"
