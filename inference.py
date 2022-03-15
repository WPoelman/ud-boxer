import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from synse.graph.mapper import Converter
from synse.graph.rewrite import BoxRemover, NodeRemover, POSResolver
from synse.sbn.sbn import SBNGraph
from synse.sbn.sbn_spec import SUPPORTED_LANGUAGES
from synse.ud.ud import UDGraph


def get_args() -> Namespace:
    parser = ArgumentParser()
    # Maybe also add option to glob folder (just as with the store_ud_parse
    # script). Maybe also provide a 'raw' sentence (raw sent -> UD parse -> SBN)
    # option.
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="UD connl file to convert to SBN.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=SUPPORTED_LANGUAGES.EN.value,
        choices=SUPPORTED_LANGUAGES.all_values(),
        type=str,
        help="Language to use for pipeline.",
    )
    parser.add_argument(
        "-m",
        "--mappings_path",
        required=True,
        type=str,
        help="Extracted mappings file to use.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    """
    Inference can ben done on a UD parse. This will be read into a UDGraph,
    after this the stored transformations and mappings will be applied as
    much as possible and the result will be a .sbn file (possibly with
    a visualization).

    As an extra, it might be handy to log the applied transformations
    & mappings to see what led to certain decisions.
    """
    input_path = Path(args.input_path)
    S = SBNGraph().from_path(f"{input_path.parent}/en.drs.sbn")
    U = UDGraph().from_path(input_path)

    S.to_png("data/tmp_output/test_converter_reference.png")
    U.to_png("data/tmp_output/test_converter_ud_parse.png")

    with open(args.mappings_path) as f:
        mappings = json.load(f)
    converter = Converter([NodeRemover, BoxRemover, POSResolver], mappings)

    result = converter.convert(U)

    result.to_png("data/tmp_output/test_converter_converted_ud.png")


if __name__ == "__main__":
    main()
