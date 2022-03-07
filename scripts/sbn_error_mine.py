import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from synse.sbn import SBNGraph, sbn_graphs_are_isomorphic


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        default="data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold",
        help="Path to start recursively searching for sbn files.",
    )
    parser.add_argument(
        "-e",
        "--error_file",
        type=str,
        default="../logs/sbn_errors.txt",
        help="File to write errors to.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0
    errors = []

    for filepath in Path(args.starting_path).glob("**/*.sbn"):
        total += 1
        try:
            A = SBNGraph().from_path(filepath)

        except Exception as e:
            error_msg = f"Unable to parse {filepath}\nReason: {e}"
            errors.append(error_msg)
            print(error_msg)
            failed += 1
            continue

        try:
            save_path = Path(f"{filepath.parent}/en.test.sbn")
            A.to_sbn(save_path)
            B = SBNGraph().from_path(save_path)
            if sbn_graphs_are_isomorphic(A, B):
                save_path.unlink()
            else:
                raise AssertionError(
                    "Reconstructed graph and original are not the same"
                )

        except Exception as e:
            error_msg = f"Unable to save {filepath}\nReason: {e}"
            errors.append(error_msg)
            print(error_msg)
            failed += 1
            continue

    end = round(time.perf_counter() - start, 2)

    errors and Path(args.error_file).write_text("\n\n".join(errors))

    print(
        f"""

    Total files:             {total:>{6}}
    Parsed without errors:   {total - failed:>{6}}
    Parsed with errors:      {failed:>{6}}
     
    Took {end} seconds
    """
    )


if __name__ == "__main__":
    main()