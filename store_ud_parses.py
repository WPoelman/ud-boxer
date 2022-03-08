import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

from synse.sbn.sbn_spec import SUPPORTED_LANGUAGES
from synse.ud import UD_LANG_DICT, UD_SYSTEM


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
        default="../logs/errors_ud.txt",
        help="File to write errors to.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=SUPPORTED_LANGUAGES.EN.value,
        choices=SUPPORTED_LANGUAGES.all_values(),
        type=str,
        help="Language to use for pipeline.",
    )
    parser.add_argument(
        "-s",
        "--ud_system",
        default=UD_SYSTEM.STANZA.value,
        type=str,
        choices=UD_SYSTEM.all_values(),
        help="System pipeline to use for generating UD parses.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0
    errors = []

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

    for file in tqdm(list(Path(args.starting_path).glob("**/*.raw"))):
        total += 1
        try:
            result = pipeline(file.read_text())
            out_file = Path(file.parent / file_format)

            if args.ud_system == UD_SYSTEM.STANZA:
                CoNLL.write_doc2conll(result, out_file)
            elif args.ud_system == UD_SYSTEM.TRANKIT:
                out_file.write_text(trankit2conllu(result))
        except Exception as e:
            error_msg = f"Unable to generate ud for {file}\nReason: {e}\n"
            errors.append(error_msg)
            print(error_msg)

            failed += 1

    end = round(time.perf_counter() - start, 2)

    errors and Path(args.error_file).write_text("\n\n".join(errors))

    print(
        f"""

    Total files:            {total:>{6}}
    UD without errors:      {total - failed:>{6}}
    UD with errors:         {failed:>{6}}
    Took {end} seconds
    """
    )


if __name__ == "__main__":
    main()
