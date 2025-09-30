"""SystemO -> Python by monomorphization: the dictionary passing transform
with all dictionaries resolved at compile time into specialized copies."""

from lango.shared.ast.nodes import Program
from lango.systemo.compiler.python.codegen import Strategy, generate


def compile_program(program: Program) -> str:
    return generate(program, Strategy.MONOMORPHIZATION)
