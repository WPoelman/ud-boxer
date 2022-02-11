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

    parser.add_argument(
        '-e', '--error_file', type=str,
        default='errors.txt',
        help='File to write errors to.'
    )
    
    return parser.parse_args()


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0
    errors = []

    for filepath in Path(args.starting_path).glob('**/*.sbn'):
        with open(filepath) as f:
            total += 1
            try:
                G = SBNGraph().from_string(f.read())
            except Exception as e:
                error_msg = f'Unable to parse {filepath}\nReason: {e}\n'
                errors.append(error_msg)
                print(error_msg)

                failed += 1

            # TODO: Add UD parse in JSON and SBN in JSON formats to
            # current folder

    end = round(time.perf_counter() - start, 2)

    with open(args.error_file, 'w') as f:
        f.write('\n\n'.join(errors))

    print(f'''

    Total files:            {total:>{6}}
    Parsed without errors:  {total - failed:>{6}}
    Parsed with errors:     {failed:>{6}}
    Took {end} seconds
    ''')


if __name__ == '__main__':
    main()
