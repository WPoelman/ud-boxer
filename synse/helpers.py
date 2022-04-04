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


def smatch_score(gold: PathLike, test: PathLike) -> Dict[str, float]:
    """Use mtool to score two amr-like graphs using SMATCH"""
    try:
        smatch_cmd = f"mtool --read amr --score smatch --gold {gold} {test}"
        response = subprocess.check_output(smatch_cmd, shell=True)
        decoded = json.loads(response)
        clean_dict = {KEY_MAPPING.get(k, k): v for k, v in decoded.items()}
    except subprocess.CalledProcessError as e:
        raise SBNError(
            f"Could not call mtool smatch with command '{smatch_cmd}'\n{e}"
        )

    return clean_dict
