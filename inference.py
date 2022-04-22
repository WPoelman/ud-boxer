import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from synse.config import Config
from synse.grew_rewrite import Grew
from synse.ud import UDGraph, UDParser

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

GREW = Grew()


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--sentence",
        type=str,
        help="Sentence to transform to SBN.",
    )
    parser.add_argument(
        "--ud",
        type=str,
        help="Path to UD conll parse to transform to SBN.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=Config.SUPPORTED_LANGUAGES.EN.value,
        choices=Config.SUPPORTED_LANGUAGES.all_values(),
        type=str,
        help="Language to use for ud pipelines.",
    )
    parser.add_argument(
        "-s",
        "--ud_system",
        default=Config.UD_SYSTEM.STANZA.value,
        type=str,
        choices=Config.UD_SYSTEM.all_values(),
        help="System pipeline to use for generating UD parse.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Output directory to store results in, will be created if it "
        "does not exist.",
    )

    # Main options
    parser.add_argument(
        "--store_visualizations",
        action="store_true",
        help="Store png of prediction in 'output_dir'.",
    )
    parser.add_argument(
        "--store_penman",
        action="store_true",
        help="Store penman version of DRS of prediction in 'output_dir'.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    if not args.sentence and not args.ud:
        raise ValueError("Please provide either a sentence or UD conll file.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    if args.sentence:
        parser = UDParser(system=args.ud_system, language=args.language)
        ud_filepath = Path(
            output_dir / f"{args.language}.ud.{args.ud_system}.conll"
        )
        parser.parse(args.sentence, ud_filepath)
    elif args.ud:
        ud_filepath = Path(args.ud).resolve()

    res = GREW.run(ud_filepath)
    res.to_sbn(Path(output_dir / f"{args.language}.drs.sbn"))

    if args.store_visualizations:
        UDGraph().from_path(ud_filepath).to_png(
            output_dir / f"{args.language}.ud.{args.ud_system}.png"
        )
        res.to_png(Path(output_dir / f"{args.language}.drs.png"))

    if args.store_penman:
        res.to_penman(Path(output_dir / f"{args.language}.drs.penman"))
        res.to_penman(
            Path(output_dir / f"{args.language}.drs.lenient.penman"),
            split_sense=True,
        )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
