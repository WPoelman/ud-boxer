import concurrent.futures
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from synse.grew_rewrite import Grew
from synse.helpers import pmb_generator, smatch_score
from synse.sbn_spec import SUPPORTED_LANGUAGES, get_doc_id
from synse.ud import UD_SYSTEM

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
        res.to_png(Path(predicted_dir / f"output.png"))

    if args.store_sbn:
        res.to_sbn(Path(predicted_dir / f"output.sbn"))

    penman_path = res.to_penman(Path(predicted_dir / f"output.penman"))
    scores = smatch_score(
        current_dir / f"{args.language}.drs.penman",
        penman_path,
    )
    penman_lenient_path = res.to_penman(
        Path(predicted_dir / f"output.lenient.penman"),
        split_sense=True,
    )
    lenient_scores = smatch_score(
        current_dir / f"{args.language}.drs.lenient.penman",
        penman_lenient_path,
    )

    result_record = {
        "pmb_id": get_doc_id(args.language, ud_filepath),
        "raw_sent": raw_sent,
        **scores,
        **{f"{k}_lenient": v for k, v in lenient_scores.items()},
    }

    return result_record


def full_run(args, ud_filepath):
    try:
        return generate_result(args, ud_filepath), str(ud_filepath)
    except Exception as e:
        logger.error(e)
        return None, str(ud_filepath)


def main():
    args = get_args()

    results_records = []
    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"
    failed = 0
    files_with_errors = []

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
            result, path = res.result()
            if result:
                results_records.append(result)
            else:
                files_with_errors.append(path)
                failed += 1

    df = pd.DataFrame().from_records(results_records)
    if args.results_file:
        df.to_csv(args.results_file, index=False)

    if files_with_errors:
        Path("paths_with_errors.txt").write_text("\n".join(files_with_errors))

    print(
        f"""
    
    PARSED DOCS:          {len(df)}
    FAILED DOCS:          {failed}
    TOTAL DOCS:           {len(df) + failed}

    AVERAGE F1 (strict):  {df["f1"].mean():.2} ({df["f1"].min():.2} - {df["f1"].max():.2})
    AVERAGE F1 (lenient): {df["f1_lenient"].mean():.2} ({df["f1_lenient"].min():.2} - {df["f1_lenient"].max():.2})
    """
    )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()