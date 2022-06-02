import logging
from argparse import ArgumentParser, Namespace

from tqdm.contrib.logging import logging_redirect_tqdm

from ud_boxer.misc import ensure_ext
from ud_boxer.sbn import SBNGraph

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to SBN file.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./output.penman",
        help="Path to save penman output to.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    SBNGraph().from_path(args.input_path).to_penman(args.output_path)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
