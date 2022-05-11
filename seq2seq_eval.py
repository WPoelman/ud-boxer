import logging
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

from synse.config import Config
from synse.grew_rewrite import Grew
from synse.helpers import PMB, smatch_score
from synse.sbn import SBNGraph
from synse.sbn_spec import get_base_id, get_doc_id
from synse.ud import UDGraph, UDParser

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        required=True,
        help="Path to start recursively search for SBN files.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to se2seq output file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Output directory to store results in, will be created if it "
        "does not exist.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=Config.SUPPORTED_LANGUAGES.EN.value,
        choices=Config.SUPPORTED_LANGUAGES.all_values(),
        type=str,
        help="Language to use for dataset.",
    )
    parser.add_argument(
        "--data_split",
        default=Config.DATA_SPLIT.TRAIN.value,
        choices=Config.DATA_SPLIT.all_values(),
        type=str,
        help="Data split to run inference on.",
    )
    return parser.parse_args()


def generate_result(args, sbn_line, gold_path):
    current_dir = gold_path.parent

    raw_sent = Path(current_dir / f"{args.language}.raw").read_text().rstrip()
    G = SBNGraph().from_string(sbn_line, is_single_line=True)

    with tempfile.NamedTemporaryFile("w") as f:
        scores = smatch_score(gold_path, G.to_penman(f.name))
        lenient_scores = smatch_score(
            current_dir / f"{args.language}.drs.lenient.penman",
            G.to_penman(f.name, lenient=True),
        )

    result_record = {
        "pmb_id": get_doc_id(args.language, gold_path),
        "raw_sent": raw_sent,
        **scores,
        **{f"{k}_lenient": v for k, v in lenient_scores.items()},
    }
    return result_record


def full_run(args, sbn_line, filepath):
    path = str(filepath)
    try:
        return generate_result(args, sbn_line, filepath), None
    except Exception as e:
        logger.error(f"{path}: {e}")
        return None, path


def main():
    args = get_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    dataset = {
        k: v
        for k, v in [
            item.split(",")
            for item in Path(args.input_file).read_text().split("\n")
        ]
    }

    pmb = PMB(args.data_split)
    result_records, error_records = [], []
    failed = 0
    for filepath in pmb.generator(
        args.starting_path,
        f"**/{args.language}.drs.penman",
        desc_tqdm="Gathering data",
    ):
        filepath = Path(filepath).resolve()
        base_id = get_base_id(filepath)
        sbn_line = dataset[base_id]
        score, err = full_run(args, sbn_line, filepath)

        if score:
            result_records.append(score)
        else:
            error_records.append({"pmb_id": base_id, "error": err})
            failed += 1

    df = pd.DataFrame().from_records(result_records)
    df_err = pd.DataFrame().from_records(error_records)

    df.to_csv(Path(args.output_dir) / "results.csv", index=False)
    df_err.to_csv(Path(args.output_dir) / "errors.csv", index=False)

    print(
        f"""
    
    PARSED DOCS:          {len(df)}
    FAILED DOCS:          {failed}
    TOTAL DOCS:           {len(df) + failed}

    AVERAGE F1 (strict):  {df["f1"].mean():.3} ({df["f1"].min():.3} - {df["f1"].max():.3})
    AVERAGE F1 (lenient): {df["f1_lenient"].mean():.3} ({df["f1_lenient"].min():.3} - {df["f1_lenient"].max():.3})
    """
    )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
