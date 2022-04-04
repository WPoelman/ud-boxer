import json
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

from synse.grew_rewrite import Grew
from synse.sbn import SBNGraph
from synse.sbn_spec import SUPPORTED_LANGUAGES, SBNError
from synse.ud import UD_SYSTEM, UDGraph


def get_args() -> Namespace:
    parser = ArgumentParser()
    # Maybe also add option to glob folder (just as with the store_ud_parse
    # script). Maybe also provide a 'raw' sentence (raw sent -> UD parse -> SBN)
    # option.
    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        required=True,
        help="Path to start recursively search for sbn / ud files.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=SUPPORTED_LANGUAGES.EN.value,
        choices=SUPPORTED_LANGUAGES.all_values(),
        type=str,
        help="Language to use for ud pipelines.",
    )
    parser.add_argument(
        "-s",
        "--ud_system",
        default=UD_SYSTEM.STANZA.value,
        type=str,
        choices=UD_SYSTEM.all_values(),
        help="System pipeline to use for generating UD parses.",
    )
    parser.add_argument(
        "--lenient_pm",
        help="System pipeline to use for generating UD parses.",
    )

    return parser.parse_args()


SMATCH_PATH = (
    "/home/wessel/Documents/documents/study/1_thesis/libraries/mtool/main.py"
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


def get_smatch_score(gold, test):
    try:
        smatch_cmd = f"python {SMATCH_PATH}  --read amr --score smatch --gold {gold} {test}"
        response = subprocess.check_output(smatch_cmd, shell=True)
        decoded = json.loads(response)
        clean_dict = {KEY_MAPPING.get(k, k): v for k, v in decoded.items()}
    except subprocess.CalledProcessError:
        raise SBNError(f"Could not call smatch with command '{smatch_cmd}'")

    return clean_dict


def main():
    args = get_args()
    """
    Inference can be done on a UD parse. This will be read into a UDGraph,
    after this the stored transformations and mappings will be applied as
    much as possible and the result will be a .sbn file (possibly with
    a visualization).

    As an extra, it might be handy to log the applied transformations
    & mappings to see what led to certain decisions.
    """
    grew = Grew()
    desc_msg = "Running inference"
    path_glob = Path(args.starting_path).glob("**/en.drs.penman")
    good_ones = []

    for filepath in tqdm(path_glob, desc=desc_msg):
        ud_filepath = Path(
            filepath.parent / f"{args.language}.ud.{args.ud_system}.conll"
        )
        if not ud_filepath.exists():
            continue
        try:
            predicted_dir = Path(filepath.parent / "predicted")
            predicted_dir.mkdir(exist_ok=True)

            result = grew.run(ud_filepath)
            for i, res in enumerate(result):
                res.to_png(Path(predicted_dir / f"{i}_output.png"))
                res.to_sbn(Path(predicted_dir / f"{i}_output.sbn"))
                penman_path = res.to_penman(
                    Path(predicted_dir / f"{i}_output.penman")
                )
                scores = get_smatch_score(
                    filepath.parent / "en.drs.penman", penman_path
                )
                # print(scores)
                if scores["f1"] > 0.5:
                    good_ones.append(str(filepath))
                Path(predicted_dir / "scores.json").write_text(
                    json.dumps(scores, indent=2)
                )
        except Exception as e:
            print(e)
            continue
    print("\n".join(good_ones))


if __name__ == "__main__":
    main()
