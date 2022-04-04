import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from synse.grew_rewrite import Grew
from synse.helpers import pmb_generator, smatch_score
from synse.sbn_spec import SUPPORTED_LANGUAGES
from synse.ud import UD_SYSTEM

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


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

    good_ones = []
    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"

    for filepath in pmb_generator(
        args.starting_path, "**/en.drs.penman", desc_tqdm="Running inference"
    ):
        ud_filepath = Path(filepath.parent / ud_file_format)
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
                scores = smatch_score(
                    filepath.parent / "en.drs.penman", penman_path
                )
                Path(predicted_dir / "scores.json").write_text(
                    json.dumps(scores, indent=2)
                )

                if scores["f1"] > 0.5:
                    good_ones.append(str(filepath))
        except Exception as e:
            print(e)
            continue
    print("\n".join(good_ones))


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
