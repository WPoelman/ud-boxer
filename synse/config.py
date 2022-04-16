from enum import Enum
from pathlib import Path
from typing import List


class Config:
    GRS_PATH = Path(Path(__file__).parent.parent / "grew/main.grs").resolve()

    EDGE_MAPPINGS_PATH = Path(
        Path(__file__).parent.parent / "data/output/edge_mappings.json"
    ).resolve()

    class SUPPORTED_LANGUAGES(str, Enum):
        """Supported languages, this is based on the PMB 4.0.0"""

        NL = "nl"
        DE = "de"
        IT = "it"
        EN = "en"

        @classmethod
        def all_values(cls) -> List[str]:
            return [i.value for i in cls]
