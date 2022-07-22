import json
import subprocess
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from tqdm import tqdm

from ud_boxer.config import Config
from ud_boxer.sbn import SBNSource
from ud_boxer.sbn_spec import SBNError, get_base_id

__all__ = [
    "PMB",
    "pmb_generator",
    "smatch_score",
]


class PMB:
    def __init__(
        self,
        split: Config.DATA_SPLIT,
        language: Config.SUPPORTED_LANGUAGES,
    ):
        if split == Config.DATA_SPLIT.ALL:
            self.ids = set()
        else:
            self.ids = set(Config.get_split_ids(language, split))

    def generator(
        self,
        starting_path: PathLike,
        pattern: str,
        exclude: str = "predicted",
        disable_tqdm: bool = False,
        desc_tqdm: str = "",
    ) -> Generator[Path, None, None]:
        # Is this ideal? No not at all since were looping over the entire
        # dataset, even if we just need a couple of files from a split.
        # The alternative is also not great since then we end up with a very
        # fragmented PMB file structure. This is the most understandable and
        # clear way to deal with this issue in my opinion. Suggestions welcome!
        dont_filter = len(self.ids) == 0
        test = lambda p: True if dont_filter else get_base_id(p) in self.ids

        yield from (
            path
            for path in pmb_generator(
                starting_path,
                pattern,
                exclude,
                disable_tqdm,
                desc_tqdm,
            )
            if test(path)
        )


def pmb_generator(
    starting_path: PathLike,
    pattern: str,
    # By default we don't want to regenerate predicted output
    exclude: str = "predicted",
    disable_tqdm: bool = False,
    desc_tqdm: str = "",
) -> Generator[Path, None, None]:
    """Helper to glob over the pmb dataset"""
    path_glob = Path(starting_path).glob(pattern)
    return tqdm(
        (p for p in path_glob if exclude not in str(p)),
        disable=disable_tqdm,
        desc=desc_tqdm,
    )


_KEY_MAPPING = {
    "n": "input_graphs",
    "g": "gold_graphs_generated",
    "s": "evaluation_graphs_generated",
    "c": "correct_graphs",
    "p": "precision",
    "r": "recall",
    "f": "f1",
}
_RELEVANT_ITEMS = ["p", "r", "f"]


def smatch_score(gold: PathLike, test: PathLike) -> Dict[str, float]:
    """Use mtool to score two amr-like graphs using SMATCH"""
    try:
        # NOTE: this is not ideal, but mtool is quite esoteric in how it reads
        # in graphs, so it's quite hard to just plug two amr-like strings
        # in it. Maybe we can run this as a deamon to speed it up a bit or
        # put some time into creating a usable package to import for this use-
        # case.
        smatch_cmd = f"mtool --read amr --score smatch --gold {gold} {test}"
        response = subprocess.check_output(smatch_cmd, shell=True)
        decoded = json.loads(response)
    except subprocess.CalledProcessError as e:
        raise SBNError(
            f"Could not call mtool smatch with command '{smatch_cmd}'\n{e}"
        )

    clean_dict = {
        _KEY_MAPPING.get(k, k): v
        for k, v in decoded.items()
        if k in _RELEVANT_ITEMS
    }

    return clean_dict


def create_record(
    pmb_id: str,
    raw_sent: str,
    sbn_source: SBNSource = SBNSource.UNKNOWN,
    strict_scores: Dict[str, Any] = dict(),
    lenient_scores: Dict[str, Any] = dict(),
    sbn: Optional[str] = None,
    lenient_error: Optional[str] = None,
    strict_error: Optional[str] = None,
):
    return {
        "pmb_id": pmb_id,
        "source": sbn_source,
        "raw_sent": raw_sent,
        "sbn_str": sbn,
        "lenient_error": lenient_error,
        "strict_error": strict_error,
        **strict_scores,
        **{f"{k}_lenient": v for k, v in lenient_scores.items()},
    }
