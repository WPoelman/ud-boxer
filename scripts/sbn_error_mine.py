import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from synse.sbn import SBNGraph, sbn_graphs_are_isomorphic
from thesis.synse.sbn.sbn import SBNError

logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        default="data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold",
        help="Path to start recursively searching for sbn files.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0

    for filepath in Path(args.starting_path).glob("**/*.sbn"):
        total += 1
        try:
            A = SBNGraph().from_path(filepath)
        except SBNError as e:
            logger.error(f"Unable to parse {filepath}\nReason: {e}")
            failed += 1
            continue

        try:
            save_path = Path(f"{filepath.parent}/en.test.sbn")
            A.to_sbn(save_path)
            B = SBNGraph().from_path(save_path)
            if sbn_graphs_are_isomorphic(A, B):
                save_path.unlink()
            else:
                raise AssertionError(
                    "Reconstructed graph and original are not the same"
                )

        except (SBNError, AssertionError) as e:
            logger.error(f"Unable to save {filepath}\nReason: {e}")
            failed += 1
            continue

    end = round(time.perf_counter() - start, 2)

    logger.info(
        f"""

    Total files:             {total:>{6}}
    Parsed without errors:   {total - failed:>{6}}
    Parsed with errors:      {failed:>{6}}
     
    Took {end} seconds
    """
    )


if __name__ == "__main__":
    main()
