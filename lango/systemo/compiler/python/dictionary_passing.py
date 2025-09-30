"""SystemO -> Python using the dictionary passing transform (Section 4)."""

from lango.shared.ast.nodes import Program
from lango.systemo.compiler.python.codegen import Strategy, generate


def compile_program(program: Program) -> str:
    return generate(program, Strategy.DICTIONARY_PASSING)
