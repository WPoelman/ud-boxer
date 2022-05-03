import logging
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from synse.config import Config
from synse.grew_rewrite import Grew
from synse.helpers import pmb_generator
from synse.mapper import MapExtractor
from synse.sbn import SBNError, SBNGraph, sbn_graphs_are_isomorphic
from synse.sbn_spec import get_doc_id
from synse.ud import UDGraph, UDParser

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
        default=Config.SUPPORTED_LANGUAGES.EN.value,
        choices=Config.SUPPORTED_LANGUAGES.all_values(),
        type=str,
        help="Language to use for ud pipelines.",
    )
    parser.add_argument(
        "-s",
        "--ud_system",
        default=Config.UD_SYSTEM.STANZA.value,
        type=str,
        choices=Config.UD_SYSTEM.all_values(),
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
        "--search_dataset",
        action="store_true",
        help="Search the dataset for some stuff.",
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
    parser.add_argument(
        "--lenient_penman",
        action="store_true",
        help="Indicates whether or not to split senses into their components "
        "for more lenient Penman (AMR-like) scoring. When left default, scoring "
        "the generated penmen output will indirectly also target the word sense "
        "disambiguation performance included in it.",
    )

    return parser.parse_args()


def store_ud_parses(args):
    parser = UDParser(system=args.ud_system, language=args.language)
    ud_file_format = f"{args.language}.ud.{args.ud_system}.conll"

    for filepath in pmb_generator(
        args.starting_path, "**/*.raw", desc_tqdm="Storing UD parses "
    ):
        try:
            parser.parse_path(filepath, Path(filepath.parent / ud_file_format))
        except Exception as e:
            logger.error(
                f"Unable to generate ud for {filepath}\nReason: {e}\n"
            )


def search_dataset(args):
    """This function loops over the dataset to collect some info"""

    for system in Config.UD_SYSTEM.all_values():
        results = []
        for filepath in pmb_generator(
            args.starting_path,
            f"**/*.ud.{system}.conll",
            desc_tqdm="Searching multi-sentence UD parses ",
        ):
            sentences = Path(filepath).read_text().rstrip().split("\n\n")
            if len(sentences) > 1:
                results.append(str(filepath))
        Path(f"multi_sentence_conll_files_{system}.txt").write_text(
            "\n".join(results)
        )


def extract_mappings(args):
    extractor = MapExtractor()
    grew = Grew()

    for filepath in pmb_generator(
        args.starting_path,
        f"**/*.sbn",
        desc_tqdm="Extracting mappings",
    ):
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
            T = grew.run(ud_filepath)
        except SBNError as e:
            # Ignore the empty sbn docs or whitespace ids
            logger.error(f"Cannot parse {filepath} reason: {e}")
            continue
        except Exception as e:
            logger.error(f"Error parsing {filepath} reason: {e}")
            continue

        doc_id = get_doc_id(args.language, ud_filepath)

        extractor.extract(S, T, doc_id)
    extractor.export_csv("test_mappings.csv")
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
    name = ".lenient.penman" if args.lenient_penman else ".penman"
    for filepath in pmb_generator(
        args.starting_path, "**/*.sbn", desc_tqdm="Generating Penman files "
    ):
        try:
            SBNGraph().from_path(filepath).to_penman(
                Path(filepath.parent / f"{filepath.stem}{name}").resolve(),
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
    Path("cyclic_paths.txt").write_text("\n".join(paths))


def main():
    args = get_args()

    start = time.perf_counter()

    if args.store_ud_parses:
        store_ud_parses(args)

    if args.search_dataset:
        search_dataset(args)

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
    with logging_redirect_tqdm():
        main()
