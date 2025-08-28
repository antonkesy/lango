from lark import Lark, ParseTree
import os
from pathlib import Path


def parse_lark(
    path: Path, grammar: Path, prelude_dir: Path, file_extension: str
) -> ParseTree:
    parser = Lark.open(
        str(grammar),
        parser="lalr",
    )

    prelude_content = ""

    if os.path.exists(prelude_dir):
        for filename in sorted(os.listdir(prelude_dir)):
            if filename.endswith(f".{file_extension}"):
                prelude_file_path = os.path.join(prelude_dir, filename)
                try:
                    with open(prelude_file_path, "r") as prelude_file:
                        prelude_content += prelude_file.read() + "\n"
                except FileNotFoundError:
                    pass

    with open(path) as f:
        main_content = f.read()

    return parser.parse(main_content + prelude_content)
