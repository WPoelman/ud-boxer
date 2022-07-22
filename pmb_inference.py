import concurrent.futures
import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ud_boxer.config import Config
from ud_boxer.grew_rewrite import Grew
from ud_boxer.helpers import PMB, create_record, smatch_score
from ud_boxer.misc import ensure_ext
from ud_boxer.sbn import SBNSource
from ud_boxer.sbn_spec import get_doc_id

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        required=True,
        help="Path to start recursively search for SBN & UD files.",
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
        "--ud_system",
        default=Config.UD_SYSTEM.STANZA.value,
        type=str,
        choices=Config.UD_SYSTEM.all_values(),
        help="UD system to use for generating parses.",
    )
    parser.add_argument(
        "--data_split",
        default=Config.DATA_SPLIT.TRAIN.value,
        choices=Config.DATA_SPLIT.all_values(),
        type=str,
        help="Data split to run inference on.",
    )
    parser.add_argument(
        "--sbn_source",
        default=SBNSource.GREW.value,
        type=str,
        choices=SBNSource.all_values(),
        help="Add flag to SBNGraph and results where this file came from.",
    )
    parser.add_argument(
        "-r",
        "--results_file",
        type=str,
        help="CSV file to write results and scores to.",
    )
    parser.add_argument(
        "-w",
        "--max_workers",
        default=16,
        help="Max concurrent workers used to run inference with. Be careful "
        "with setting this too high since mtool might error (segfault) if hit "
        "too hard by too many concurrent tasks.",
    )

    # Main options
    parser.add_argument(
        "--clear_previous",
        action="store_true",
        help="When visiting a directory, clear the previously predicted "
        "output if it's there.",
    )
    parser.add_argument(
        "--store_visualizations",
        action="store_true",
        help="Store png of prediction in 'predicted' directory.",
    )
    parser.add_argument(
        "--store_sbn",
        action="store_true",
        help="Store SBN of prediction in 'predicted' directory.",
    )
    return parser.parse_args()


def generate_result(args, ud_filepath):
    current_dir = ud_filepath.parent

    pred_dir = current_dir / "predicted"
    pred_dir.mkdir(exist_ok=True)

    if args.clear_previous:
        for item in pred_dir.iterdir():
            if item.is_file():
                item.unlink()

    G = GREW.run(ud_filepath)
    G.source = args.sbn_source  # Setter?
    if args.store_visualizations:
        G.to_png(pred_dir / "output.png")

    if args.store_sbn:
        G.to_sbn(pred_dir / "output.sbn")

    penman_path = G.to_penman(pred_dir / "output.penman")
    scores = smatch_score(
        current_dir / f"{args.language}.drs.penman",
        penman_path,
    )
    penman_lenient_path = G.to_penman(
        pred_dir / "output.lenient.penman",
        evaluate_sense=False,
    )
    lenient_scores = smatch_score(
        current_dir / f"{args.language}.drs.lenient.penman",
        penman_lenient_path,
    )
    return scores, lenient_scores, G.to_sbn_string()


def full_run(args, ud_filepath):
    raw_sent = (
        Path(ud_filepath.parent / f"{args.language}.raw").read_text().rstrip()
    )

    sbn, error = None, None
    scores, lenient_scores = dict(), dict()

    try:
        scores, lenient_scores, sbn = generate_result(args, ud_filepath)
    except Exception as e:
        error = str(e)
        logger.error(f"{ud_filepath}: {error}")

    # TODO: fix this with the new explicit strict/lenient names
    record = create_record(
        pmb_id=get_doc_id(args.language, ud_filepath),
        raw_sent=raw_sent,
        sbn_source=args.sbn_source,
        sbn=sbn,
        error=error,
        scores=scores,
        lenient_scores=lenient_scores,
    )
    return record


def main():
    args = get_args()

    # I know, I know...
    global GREW
    GREW = Grew(language=args.language)

    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"
    pmb = PMB(args.data_split, args.language)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []
        for filepath in pmb.generator(
            args.starting_path,
            f"**/{args.language}.drs.penman",
            desc_tqdm="Gathering data",
        ):
            ud_filepath = filepath.parent / ud_file_format
            if not ud_filepath.exists():
                continue
            futures.append(executor.submit(full_run, args, ud_filepath))

        result_records = [
            res.result()
            for res in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Running inference",
            )
        ]

    result_path = Config.get_result_dir(args.language, args.data_split)

    df = pd.DataFrame().from_records(result_records)
    if args.results_file:
        final_path = result_path / ensure_ext(args.results_file, ".csv").name
        df.to_csv(final_path, index=False)

    df["f1"] = df["f1"].fillna(0)
    df["f1_lenient"] = df["f1_lenient"].fillna(0)
    generation_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    overall_result_msg = f"""
    {generation_data}

    ARGS: {args}

    DATA SPLIT:           {args.data_split}
    PARSED DOCS:          {len(df[df['error'].isnull()])}
    FAILED DOCS:          {len(df[df['error'].notnull()])}
    TOTAL DOCS:           {len(df)}

    AVERAGE F1 (strict):  {df["f1"].mean():.3} ({df["f1"].min():.3} - {df["f1"].max():.3})
    AVERAGE F1 (lenient): {df["f1_lenient"].mean():.3} ({df["f1_lenient"].min():.3} - {df["f1_lenient"].max():.3})
    """

    with open(result_path / "overall.txt", "a") as f:
        f.write(f"{overall_result_msg}\n\n")

    print(overall_result_msg)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
