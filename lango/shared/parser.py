"""Parsing: the shared grammar (``lango.lark``) plus a language grammar,
with the language's prelude appended to (or, for System O, prepended to)
the user program."""

import re
from functools import cache
from pathlib import Path

from lark import Lark, ParseTree

SHARED_GRAMMAR = Path(__file__).with_name("lango.lark")

# Keywords that the top-level name scan below must not mistake for definitions.
_RESERVED = frozenset(
    {
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
    },
)
_TOP_LEVEL_NAME = re.compile(r"(?m)^[ \t]*([A-Za-z_][\w']*)\s*(?:::|\(|=)")


@cache
def _parser(grammar: Path) -> Lark:
    return Lark(SHARED_GRAMMAR.read_text() + "\n" + grammar.read_text(), parser="lalr")


def _read_prelude(prelude_dir: Path, extension: str) -> str:
    return "".join(
        path.read_text() + "\n" for path in sorted(prelude_dir.glob(f"*.{extension}"))
    )


def _top_level_names(source: str) -> set[str]:
    """Likely top-level definitions (a conservative textual approximation)."""
    return set(_TOP_LEVEL_NAME.findall(source)) - _RESERVED


def _check_prelude_conflicts(prelude: str, program: str) -> None:
    conflicts = _top_level_names(prelude) & _top_level_names(program)
    if conflicts:
        raise ValueError(
            f"Prelude defines names {sorted(conflicts)}; user file must not redefine prelude symbols",
        )


def parse_lark(
    path: Path,
    grammar: Path,
    prelude_dir: Path,
    file_extension: str,
    prelude_first: bool = False,
    check_prelude_conflicts: bool = True,
) -> ParseTree:
    prelude = _read_prelude(prelude_dir, file_extension)
    program = path.read_text()
    if check_prelude_conflicts:
        _check_prelude_conflicts(prelude, program)
    # System O scopes declarations sequentially, so its prelude must precede
    # the user program.
    source = prelude + program if prelude_first else program + prelude
    return _parser(grammar).parse(source)
