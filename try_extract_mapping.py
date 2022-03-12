import json
import time
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path

from networkx.algorithms.isomorphism import DiGraphMatcher

from synse.graph import BaseGraph
from synse.graph.mapper import MapExtractor
from synse.graph.rewrite import NodeRemover, POSResolver
from synse.sbn import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNGraph
from synse.sbn.sbn import SBN_NODE_TYPE
from synse.sbn.sbn_spec import SUPPORTED_LANGUAGES
from synse.ud import UD_SYSTEM, UDGraph
from synse.ud.ud import UD_NODE_TYPE, Collector


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        default="data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold",
        help="Path to start recursively searching for sbn and conll files.",
    )
    parser.add_argument(
        "-e",
        "--error_file",
        type=str,
        default="logs/errors_comparisons.txt",
        help="File to write errors to.",
    )
    parser.add_argument(
        "-v",
        "--visualization",
        action="store_true",
        help="Show visualizations.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="data/output",
        help="Path to save output files to.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=SUPPORTED_LANGUAGES.all_values(),
        default=SUPPORTED_LANGUAGES.EN.value,
        help="Language to use (needed for UD parsing and more)",
    )
    parser.add_argument(
        "-u",
        "--ud_system",
        type=str,
        choices=UD_SYSTEM.all_values(),
        default=UD_SYSTEM.STANZA.value,
        help="UD parse to use.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    lang = args.language

    start = time.perf_counter()
    total, failed = 0, 0

    extractor = MapExtractor()

    for filepath in Path(args.starting_path).glob("**/*.sbn"):
        total += 1

        ud_filepath = Path(
            filepath.parent / f"{lang}.ud.{args.ud_system}.conll"
        )
        if not ud_filepath.exists():
            raise FileNotFoundError(f"No UD conll file for {filepath.parent}")

        try:
            U = UDGraph().from_path(ud_filepath)
            S = SBNGraph().from_path(filepath)
        except:
            # Ignore the empty sbn docs or whitespace ids, add SBNError to deal with this more gracefully?
            continue

        extractor.extract_mappings(U, S)
    with open("test_mappings.json", "w") as f:
        json.dump(
            {
                "edge-mappings": extractor.edge_mappings,
                "node-mappings": extractor.node_mappings,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    end = round(time.perf_counter() - start, 2)

    print(
        f"""

    Total files:             {total:>{6}}
    Parsed without errors:   {total - failed:>{6}}
    Parsed with errors:      {failed:>{6}}
     
    Took {end} seconds
    """
    )


if __name__ == "__main__":
    main()
