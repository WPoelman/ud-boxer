import logging
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from synse.helpers import pmb_generator
from synse.sbn import SBNError, SBNGraph, sbn_graphs_are_isomorphic
from synse.sbn_spec import SUPPORTED_LANGUAGES
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
    parser.add_argument(
        "--lenient_penman",
        action="store_true",
        help="Indicates whether or not to split senses into their components "
        "for more lenient Penman (AMR-like) scoring. When left default, scoring "
        "the generated penmen output will indirectly also target the word sense "
        "disambiguation performance included in it.",
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
        "--store_penman",
        action="store_true",
        help="Store SBN as Penman (AMR-like).",
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

    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"

    for filepath in pmb_generator(
        args.starting_path, "**/*.raw", desc_tqdm="Storing UD parses "
    ):
        try:
            result = pipeline(filepath.read_text())
            out_file = Path(filepath.parent / ud_file_format)

            if args.ud_system == UD_SYSTEM.STANZA:
                CoNLL.write_doc2conll(result, out_file)
            elif args.ud_system == UD_SYSTEM.TRANKIT:
                out_file.write_text(trankit2conllu(result))
        except Exception as e:
            logger.error(
                f"Unable to generate ud for {filepath}\nReason: {e}\n"
            )


def extract_mappings(args):
    raise NotImplementedError(
        "The mapping needs to be redone with the introduction of Grew, can't "
        "use this currently"
    )
    # extractor = MapExtractor()

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

        # extractor.extract_mappings(U, S)

    return
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extractor.dump_mappings(Path(args.output_path) / f"mappings-{date}.json")


def error_mine(args):
    for filepath in pmb_generator(
        args.starting_path, "**/*.sbn", desc_tqdm="Mining errors "
    ):
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
    for filepath in pmb_generator(
        args.starting_path, "**/*.sbn", desc_tqdm="Creating visualizations "
    ):
        viz_dir = Path(filepath.parent / "viz")
        viz_dir.mkdir(exist_ok=True)

        try:
            SBNGraph().from_path(filepath).to_png(
                str(Path(viz_dir / f"{filepath.stem}.png").resolve())
            )
            ud_filepath = Path(
                filepath.parent / f"{args.language}.ud.{args.ud_system}.conll"
            )
            if not ud_filepath.exists():
                logger.warning(
                    f"Skipping {filepath} UD visualization, no ud parse available"
                )
                continue

            UDGraph().from_path(ud_filepath).to_png(
                str(Path(viz_dir / f"{ud_filepath.stem}.png").resolve())
            )
        except Exception as e:
            print(f"Failed: {filepath}")


def store_penman(args):
    for filepath in pmb_generator(
        args.starting_path, "**/*.sbn", desc_tqdm="Testing Penman files "
    ):
        try:
            SBNGraph().from_path(filepath).to_penman(
                Path(filepath.parent / f"{filepath.stem}.penman").resolve(),
                args.lenient_penman,
            )
        except SBNError as e:
            logger.warning(e)


def collect_cyclic_graphs(args):
    paths = []
    for filepath in pmb_generator(
        args.starting_path, "**/*.sbn", desc_tqdm="Finding cyclic sbn graphs "
    ):
        S = SBNGraph().from_path(filepath)
        if not S.is_dag:
            paths.append(str(filepath))
    Path("cyclic paths.txt").write_text("\n".join(paths))


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
        collect_cyclic_graphs(args)

    if args.store_visualizations:
        store_visualizations(args)

    if args.store_penman:
        store_penman(args)

    logger.info(f"Took {round(time.perf_counter() - start, 2)} seconds")


if __name__ == "__main__":
    main()
