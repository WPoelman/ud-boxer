import logging
from argparse import ArgumentParser, Namespace
import tempfile

from ud_boxer.helpers import smatch_score
from ud_boxer.sbn import SBNGraph

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--gold_sbn",
        type=str,
        required=True,
        help="Gold SBN file.",
    )
    parser.add_argument(
        "--predicted_sbn",
        type=str,
        required=True,
        help="Predicted SBN file to evaluate.",
    )
    parser.add_argument(
        "--is_single_line",
        action="store_true",
        help="Flag to indicate if predicted_sbn is in single line format.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    G = SBNGraph().from_path(args.gold_sbn)
    P = SBNGraph().from_path(args.predicted_sbn, is_single_line=args.is_single_line)

    with tempfile.NamedTemporaryFile("w") as gold_f, tempfile.NamedTemporaryFile("w") as pred_f:
        scores = smatch_score(G.to_penman(gold_f.name), P.to_penman(pred_f.name))

    print(scores)

if __name__ == "__main__":
    main()
