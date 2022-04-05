import concurrent.futures
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from synse.grew_rewrite import Grew
from synse.helpers import pmb_generator, rnd, smatch_score
from synse.sbn_spec import SUPPORTED_LANGUAGES, get_doc_id
from synse.ud import UD_SYSTEM

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# TODO: possibly do this in smatch_score by default
RELEVANT_COLS = ["precision", "recall", "f1"]
GREW = Grew()


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
        "-r",
        "--results_file",
        required=True,
        type=str,
        help="CSV file to write results and scores to.",
    )
    parser.add_argument(
        "-w",
        "--max_workers",
        default=16,
        help="Max concurrent workers used to run inference with. Be careful "
        "with setting this too high since grew might error (segfault) if hit "
        "too hard by too many concurrent tasks.",
    )
    return parser.parse_args()


def generate_result(args, ud_filepath):
    predicted_dir = Path(ud_filepath.parent / "predicted")
    predicted_dir.mkdir(exist_ok=True)
    raw_sent = (
        Path(ud_filepath.parent / f"{args.language}.raw").read_text().rstrip()
    )

    res = GREW.run(ud_filepath)
    # res.to_png(Path(predicted_dir / f"{i}_output.png"))
    # res.to_sbn(Path(predicted_dir / f"{i}_output.sbn"))

    penman_path = res.to_penman(Path(predicted_dir / f"output.penman"))
    scores = smatch_score(
        ud_filepath.parent / f"{args.language}.drs.penman",
        penman_path,
    )
    penman_lenient_path = res.to_penman(
        Path(predicted_dir / f"output.lenient.penman"),
        split_sense=True,
    )
    lenient_scores = smatch_score(
        ud_filepath.parent / f"{args.language}.drs.lenient.penman",
        penman_lenient_path,
    )

    # TODO: clean this up
    result = {
        "pmb_id": get_doc_id(args.language, filepath=ud_filepath),
        "raw_sent": raw_sent,
        **{k: v for k, v in scores.items() if k in RELEVANT_COLS},
        **{
            f"{k}_lenient": v
            for k, v in lenient_scores.items()
            if k in RELEVANT_COLS
        },
    }

    return result


def full_run(args, ud_filepath):
    try:
        return generate_result(args, ud_filepath)
    except Exception as e:
        logger.error(e)
        return None


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

    results_records = []
    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []
        for filepath in pmb_generator(
            args.starting_path,
            f"**/{args.language}.drs.penman",
            desc_tqdm="Submitting tasks",
        ):
            ud_filepath = Path(filepath.parent / ud_file_format)
            if not ud_filepath.exists():
                continue
            futures.append(executor.submit(full_run, args, ud_filepath))

        for res in tqdm(
            concurrent.futures.as_completed(futures), desc="Running inference"
        ):
            if result := res.result():
                results_records.append(result)
            else:
                failed += 1

    df = pd.DataFrame().from_records(results_records)
    df.to_csv(args.results_file, index=False)

    print(
        f"""
    
    PARSED DOCS:          {len(df)}
    FAILED DOCS:          {failed}
    TOTAL DOCS:           {len(df) + failed}

    AVERAGE F1 (strict):  {rnd(df["f1"].mean())} ({rnd(df["f1"].min())} - {rnd(df["f1"].max())})
    AVERAGE F1 (lenient): {rnd(df["f1_lenient"].mean())} ({rnd(df["f1_lenient"].min())} - {rnd(df["f1_lenient"].max())})
    """
    )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
