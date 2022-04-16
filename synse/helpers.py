import json
import subprocess
from os import PathLike
from pathlib import Path
from typing import Dict, Generator

from tqdm import tqdm

from synse.sbn_spec import SBNError


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


KEY_MAPPING = {
    "n": "input_graphs",
    "g": "gold_graphs_generated",
    "s": "evaluation_graphs_generated",
    "c": "correct_graphs",
    "p": "precision",
    "r": "recall",
    "f": "f1",
}
RELEVANT_ITEMS = ["p", "r", "f"]


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
        KEY_MAPPING.get(k, k): v
        for k, v in decoded.items()
        if k in RELEVANT_ITEMS
    }

    return clean_dict
