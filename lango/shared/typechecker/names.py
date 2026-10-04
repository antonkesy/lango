"""Readable names for type variables: ``a`` ... ``z``, ``a1`` ... ``z1``, ``a2`` ..."""

from itertools import count
from typing import Iterator


def var_name(index: int) -> str:
    letter = chr(ord("a") + index % 26)
    suffix = index // 26
    return letter if suffix == 0 else f"{letter}{suffix}"


def var_names() -> Iterator[str]:
    return map(var_name, count())
