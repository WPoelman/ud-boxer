import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from synse.graph.sbn import SBNGraph


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--starting_path', type=str,
        default='data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold',
        help='Path to start recursively searching for sbn files.'
    )
    return parser.parse_args()


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0
    for filepath in Path(args.starting_path).glob('**/*.sbn'):
        with open(filepath) as f:
            total += 1
            try:
                G = SBNGraph(sbn_string=f.read())
            except Exception as e:
                print(f'Unable to parse {filepath}\nReason: {e}\n')
                failed += 1

            # TODO: Add UD parse in JSON and SBN in JSON formats to
            # current folder

    end = round(time.perf_counter() - start, 2)

    print(f'''

    Total files:            {total}
    Parsed without errors:  {total - failed}
    Parsed with errors:     {failed}
    Took {end} seconds

    ''')


if __name__ == '__main__':
    main()
