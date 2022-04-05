# Grew has no stubs & mixed types everywhere, no need to bother mypy with that.
# mypy: ignore-errors
from os import PathLike
from pathlib import Path
from typing import Any, List

from stanza.utils.conll import CoNLL

import grew
from synse.sbn import SBNGraph
from synse.sbn_spec import SBNError

DEFAULT_GRS_PATH = Path(
    Path(__file__).parent.parent / "grew/main.grs"
).resolve()


class Grew:
    def __init__(self, grs_path=DEFAULT_GRS_PATH) -> None:
        grew.init()
        self.grs = grew.grs(str(grs_path))

    def run(self, conll_path: PathLike, strat: str = "main") -> List[SBNGraph]:
        # This is not ideal since we need to deserialize the file 2x, once
        # here and once 'inside' GREW. It might be worth it to convert the
        # sentence(s) to a GREW graph(s) directly. We need to this though since
        # GREW throws an error when providing a conll-u file with > 1 sentence.
        sentences, _ = CoNLL.conll2dict(conll_path)

        if len(sentences) > 1:
            raise SBNError(
                "Cannot deal with conllu files that have > 1 sentence in them "
                "currently. This is something GREW cannot parse."
            )

        grew_graph = grew.graph(str(conll_path))
        result = grew.run(self.grs, grew_graph, strat)
        return [SBNGraph().from_grew(res) for res in result]
