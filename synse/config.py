from pathlib import Path


class Config:
    GRS_PATH = Path(Path(__file__).parent.parent / "grew/main.grs").resolve()

    EDGE_MAPPINGS_PATH = Path(
        Path(__file__).parent.parent / "data/output/edge_mappings.json"
    ).resolve()
