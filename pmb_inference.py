import concurrent.futures
import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from synse.config import Config
from synse.grew_rewrite import Grew
from synse.helpers import PMB, smatch_score
from synse.misc import ensure_ext
from synse.sbn_spec import get_doc_id

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

GREW = Grew()


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
        "-s",
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

    predicted_dir = Path(current_dir / "predicted")
    predicted_dir.mkdir(exist_ok=True)

    if args.clear_previous:
        for item in predicted_dir.iterdir():
            if item.is_file():
                item.unlink()

    raw_sent = Path(current_dir / f"{args.language}.raw").read_text().rstrip()

    res = GREW.run(ud_filepath)
    if args.store_visualizations:
        res.to_png(Path(predicted_dir / "output.png"))

    if args.store_sbn:
        res.to_sbn(Path(predicted_dir / "output.sbn"))

    penman_path = res.to_penman(Path(predicted_dir / "output.penman"))
    scores = smatch_score(
        current_dir / f"{args.language}.drs.penman",
        penman_path,
    )
    penman_lenient_path = res.to_penman(
        Path(predicted_dir / "output.lenient.penman"),
        strict=False,
    )
    lenient_scores = smatch_score(
        current_dir / f"{args.language}.drs.lenient.penman",
        penman_lenient_path,
    )

    result_record = {
        "pmb_id": get_doc_id(args.language, ud_filepath),
        "raw_sent": raw_sent,
        "sbn_str": res.to_sbn_string(),
        "error": None,
        **scores,
        **{f"{k}_lenient": v for k, v in lenient_scores.items()},
    }

    return result_record


def full_run(args, ud_filepath):
    try:
        return generate_result(args, ud_filepath), None
    except Exception as e:
        path = str(ud_filepath)
        logger.error(f"{path}: {e}")
        return None, str(e)


def main():
    args = get_args()

    results_records = []
    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"
    failed = 0
    files_with_errors = []

    pmb = PMB(args.data_split)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []
        for filepath in pmb.generator(
            args.starting_path,
            f"**/{args.language}.drs.penman",
            desc_tqdm="Gathering data",
        ):
            ud_filepath = Path(filepath.parent / ud_file_format)
            if not ud_filepath.exists():
                continue
            futures.append(executor.submit(full_run, args, ud_filepath))

        for res in tqdm(
            concurrent.futures.as_completed(futures), desc="Running inference"
        ):
            result, err = res.result()
            if result:
                results_records.append(result)
            else:
                results_records.append(
                    {
                        "pmb_id": get_doc_id(args.language, ud_filepath),
                        "error": err,
                    }
                )
                failed += 1

    result_path = Path(Config.RESULT_DIR / args.data_split)

    df = pd.DataFrame().from_records(results_records)
    if args.results_file:
        final_path = result_path / ensure_ext(args.results_file, ".csv").name
        df.to_csv(final_path, index=False)

    if files_with_errors:
        Path(Config.LOG_PATH / "paths_with_errors.txt").write_text(
            "\n".join(files_with_errors)
        )

    df["f1"] = df["f1"].fillna(0)
    df["f1_lenient"] = df["f1_lenient"].fillna(0)

    overall_result_msg = f"""
    {datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}

    DATA SPLIT:           {args.data_split}
    PARSED DOCS:          {len(df)}
    FAILED DOCS:          {failed}
    TOTAL DOCS:           {len(df) + failed}

    AVERAGE F1 (strict):  {df["f1"].mean():.3} ({df["f1"].min():.3} - {df["f1"].max():.3})
    AVERAGE F1 (lenient): {df["f1_lenient"].mean():.3} ({df["f1_lenient"].min():.3} - {df["f1_lenient"].max():.3})
    """

    with open(result_path / "overall.txt", "a") as f:
        f.write(f"{overall_result_msg}\n\n")

    print(overall_result_msg)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
