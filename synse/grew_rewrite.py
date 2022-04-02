# Grew has no stubs & mixes types everywhere, no need to bother mypy with that.
# mypy: ignore-errors
from os import PathLike
from pathlib import Path
from typing import Any, List

import grew  # type: ignore
from synse.sbn import SBNGraph

DEFAULT_GRS_PATH = Path("../grew/main.grs").resolve()


class Grew:
    def __init__(self, grs_path=DEFAULT_GRS_PATH) -> None:
        grew.init()
        self.grs = grew.grs(grs_path)

    def run(self, conll_path: PathLike, strat: str = "main") -> List[SBNGraph]:
        grew_graph = grew.graph(conll_path)
        result = grew.run(self.grs, grew_graph, strat)
        return [SBNGraph().from_grew(res) for res in result]
