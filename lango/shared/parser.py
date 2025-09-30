import os
from pathlib import Path
from typing import Set

from lark import Lark, ParseTree


def parse_lark(
    path: Path,
    grammar: Path,
    prelude_dir: Path,
    file_extension: str,
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

    with open(f"./build/main.{file_extension}", "w") as f:
        f.write(main_content + prelude_content)
    # Prevent user files from redefining prelude symbols
    import re

    def _extract_top_level_names(src: str) -> Set[str]:
        # Capture likely top-level symbol definitions.
        # This intentionally favors simple, conservative matching.
        pattern = re.compile(r"(?m)^[ \t]*([A-Za-z_][\w']*)\s*(?:::|\(|=)")
        return set(pattern.findall(src))

    prelude_names = _extract_top_level_names(prelude_content)
    main_names = _extract_top_level_names(main_content)
    # Filter out language keywords and common non-symbol tokens
    reserved_keywords = {
        "inst",
        "data",
        "let",
        "do",
        "if",
        "case",
        "module",
        "where",
        "import",
        "type",
        "precedence",
    }
    prelude_names = {n for n in prelude_names if n not in reserved_keywords}
    main_names = {n for n in main_names if n not in reserved_keywords}
    conflicts = prelude_names & main_names
    if conflicts:
        raise ValueError(
            f"Prelude defines names {sorted(conflicts)}; user file must not redefine prelude symbols",
        )

    # Keep original parsing order (main then prelude) for backward compatibility
    return parser.parse(main_content + prelude_content)
