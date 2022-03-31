import json
import logging
import time
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from networkx.algorithms.isomorphism import DiGraphMatcher
from tqdm import tqdm

from synse.graph.mapper import MapExtractor
from synse.sbn import SBNGraph
from synse.sbn.sbn import SBNError, sbn_graphs_are_isomorphic
from synse.sbn.sbn_spec import SUPPORTED_LANGUAGES
from synse.ud import UD_LANG_DICT, UD_SYSTEM, UDGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()

    # Variables
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
        "-o",
        "--output_path",
        type=str,
        default="data/output",
        help="Path to save output files to.",
    )

    # Main options
    parser.add_argument(
        "--store_ud_parses",
        action="store_true",
        help="Creates and stores ud parses in the pmb dataset folders.",
    )
    parser.add_argument(
        "--extract_mappings",
        action="store_true",
        help="Extract and store mappings from U -> S.",
    )
    parser.add_argument(
        "--error_mine",
        action="store_true",
        help="See if sbn and ud parsing and storing work as expected.",
    )
    parser.add_argument(
        "--store_visualizations",
        action="store_true",
        help="Store SBN and UD visualizations in the dataset example folders.",
    )
    parser.add_argument(
        "--store_amr",
        action="store_true",
        help="Store SBN to AMR conversion (NOTE: tests it for now, not actually stores).",
    )

    return parser.parse_args()


def store_ud_parses(args):
    if args.ud_system == UD_SYSTEM.STANZA:
        from stanza import Pipeline, download
        from stanza.utils.conll import CoNLL

        # No need for very heavy NER / sentiment etc models currently
        processors = "tokenize,mwt,pos,lemma,depparse"
        download(args.language, processors=processors)
        pipeline = Pipeline(lang=args.language, processors=processors)

    elif args.ud_system == UD_SYSTEM.TRANKIT:
        from trankit import Pipeline, trankit2conllu

        pipeline = Pipeline(UD_LANG_DICT[args.language])

    file_format = f"{args.language}.ud.{args.ud_system}.conll"

    desc_msg = "Storing UD parses "
    path_glob = Path(args.starting_path).glob("**/*.raw")

    for filepath in tqdm(path_glob, desc=desc_msg):
        try:
            result = pipeline(filepath.read_text())
            out_file = Path(filepath.parent / file_format)

            if args.ud_system == UD_SYSTEM.STANZA:
                CoNLL.write_doc2conll(result, out_file)
            elif args.ud_system == UD_SYSTEM.TRANKIT:
                out_file.write_text(trankit2conllu(result))
        except Exception as e:
            logger.error(
                f"Unable to generate ud for {filepath}\nReason: {e}\n"
            )


def extract_mappings(args):
    extractor = MapExtractor()

    desc_msg = "Extracting mappings"
    path_glob = Path(args.starting_path).glob("**/*.sbn")

    for filepath in tqdm(path_glob, desc=desc_msg):
        ud_filepath = Path(
            filepath.parent / f"{args.language}.ud.{args.ud_system}.conll"
        )
        if not ud_filepath.exists():
            logger.error(
                f"Cannot extract mappings, no UD conll file for {filepath.parent}"
            )
            continue

        try:
            S = SBNGraph().from_path(filepath)
        except SBNError as e:
            # Ignore the empty sbn docs or whitespace ids
            logger.error(f"Cannot parse {filepath} reason: {e}")
            continue

        U = UDGraph().from_path(ud_filepath)

        extractor.extract_mappings(U, S)

    return
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extractor.dump_mappings(Path(args.output_path) / f"mappings-{date}.json")


def error_mine(args):
    desc_msg = "Mining errors"
    path_glob = Path(args.starting_path).glob("**/*.sbn")

    for filepath in tqdm(path_glob, desc=desc_msg):
        try:
            A = SBNGraph().from_path(filepath)
        except SBNError as e:
            logger.error(f"Unable to parse sbn: {filepath}\nReason: {e}")
            continue

        try:
            save_path = Path(f"{filepath.parent}/en.test.sbn")
            A.to_sbn(save_path)
            B = SBNGraph().from_path(save_path)
            if sbn_graphs_are_isomorphic(A, B):
                save_path.unlink()
            else:
                raise SBNError(
                    f"Reconstructed graph and original are not the same: {filepath}"
                )

        except SBNError as e:
            logger.error(f"Unable to save {filepath}\nReason: {e}")


def store_visualizations(args):
    desc_msg = "Creating visualizations"
    path_glob = Path(args.starting_path).glob("**/*.sbn")

    for filepath in tqdm(path_glob, desc=desc_msg):
        viz_dir = Path(filepath.parent / "viz_experiment")
        viz_dir.mkdir(exist_ok=True)

        ud_filepath = Path(
            filepath.parent / f"{args.language}.ud.{args.ud_system}.conll"
        )

        SBNGraph().from_path(filepath).to_png(
            str(Path(viz_dir / f"{filepath.stem}.png").resolve())
        )
        if not ud_filepath.exists():
            logger.warning(
                f"Skipping {filepath} UD visualization, no ud parse available"
            )
            continue

        UDGraph().from_path(ud_filepath).to_png(
            str(Path(viz_dir / f"{ud_filepath.stem}.png").resolve())
        )


def store_amr(args):
    desc_msg = "Testing AMR files"
    path_glob = Path(args.starting_path).glob("**/*.sbn")

    for filepath in tqdm(path_glob, desc=desc_msg):
        SBNGraph().from_path(filepath).to_amr(
            Path(filepath.parent / f"{filepath.stem}.amr").resolve()
        )


def main():
    args = get_args()

    start = time.perf_counter()

    # Maybe combine some of these since most of them loop through the same
    # files, but then again, currently the steps nice and separated.
    if args.store_ud_parses:
        store_ud_parses(args)

    if args.extract_mappings:
        extract_mappings(args)

    if args.error_mine:
        error_mine(args)

    if args.store_visualizations:
        store_visualizations(args)

    if args.store_amr:
        store_amr(args)

    logging.info(f"Took {round(time.perf_counter() - start, 2)} seconds")


if __name__ == "__main__":
    main()
