import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from ud_boxer.config import Config
from ud_boxer.grew_rewrite import Grew
from ud_boxer.ud import UDGraph, UDParser

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


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
        help="Language to use for UD pipelines.",
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
        help="Store Penman notation of DRS of prediction in 'output_dir'.",
    )
    parser.add_argument(
        "--store_dot",
        action="store_true",
        help="Store graphviz dot strings used for the visualization in "
        "'output_dir'.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    if not args.sentence and not args.ud:
        raise ValueError("Please provide either a sentence or UD conll file.")

    grew = Grew(language=args.language)

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

    res = grew.run(ud_filepath)
    res.to_sbn(output_dir / f"{args.language}.drs.sbn")

    ud_graph = UDGraph().from_path(ud_filepath)

    if args.store_visualizations:
        ud_graph.to_png(
            output_dir / f"{args.language}.ud.{args.ud_system}.png"
        )
        res.to_png(output_dir / f"{args.language}.drs.png")

    if args.store_dot:
        Path(output_dir / f"{args.language}.drs.dot").write_text(
            res.to_dot_str()
        )
        Path(
            output_dir / f"{args.language}.ud.{args.ud_system}.dot"
        ).write_text(ud_graph.to_dot_str())

    if args.store_penman:
        res.to_penman(output_dir / f"{args.language}.drs.penman")
        res.to_penman(
            output_dir / f"{args.language}.drs.lenient.penman",
            strict=False,
        )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
