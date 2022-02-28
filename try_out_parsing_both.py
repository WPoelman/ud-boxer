import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from synse.sbn import SBNGraph
from synse.ud import UDGraph


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        default="data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold",
        help="Path to start recursively searching for sbn and conll files.",
    )
    parser.add_argument(
        "-e",
        "--error_file",
        type=str,
        default="logs/errors_comparisons.txt",
        help="File to write errors to.",
    )
    parser.add_argument(
        "-v",
        "--visualization",
        action="store_true",
        help="Show visualizations.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="data/output",
        help="Path to save output files to.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0
    errors = []

    same_no_nodes = 0
    same_no_edges = 0
    same_no_nodes_and_edges = 0

    output_path = Path(args.output_path)
    for filepath in Path(args.starting_path).glob("**/*.sbn"):
        total += 1
        try:
            # TODO: find better way to handle language
            current_lang = filepath.stem.split(".")[0]

            # TODO: add option to select stanza or trankit here as well
            ud_filepath = Path(
                filepath.parent / f"{current_lang}.ud.stanza.conll"
            )
            if not ud_filepath.exists():
                raise FileNotFoundError(
                    f"No UD conll file for {filepath.parent}"
                )

            # TODO: try to extract the pmb id from the path and give it to the
            # graphs to use in exports.
            S = SBNGraph().from_path(filepath)
            U = UDGraph().from_path(ud_filepath)

            if args.visualization:
                # TODO: add output directory to args and use id
                S.show(
                    str(Path(output_path / f"{filepath.stem}.png").resolve())
                )
                U.show(
                    str(
                        Path(output_path / f"{ud_filepath.stem}.png").resolve()
                    )
                )

            total += 1

            same_nodes = len(S) == len(U)
            same_edges = len(S.edges) == len(U.edges)

            if same_nodes:
                same_no_nodes += 1
            if same_edges:
                same_no_edges += 1
            if same_nodes and same_edges:
                same_no_nodes_and_edges += 1
        except Exception as e:
            error_msg = f"Unable to parse {filepath}\nReason: {e}\n"
            print(error_msg)
            errors.append(error_msg)
            failed += 1

    end = round(time.perf_counter() - start, 2)

    errors and Path(args.error_file).write_text("\n\n".join(errors))

    print(
        f"""

    Total files:             {total:>{6}}
    Parsed without errors:   {total - failed:>{6}}
    Parsed with errors:      {failed:>{6}}
     
    same_no_nodes:           {same_no_nodes:>{6}}
    same_no_edges:           {same_no_edges:>{6}}
    same_no_nodes_and_edges: {same_no_nodes_and_edges:>{6}}

    Took {end} seconds
    """
    )


if __name__ == "__main__":
    main()
