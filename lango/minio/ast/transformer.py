from typing import Any

from lark import Tree

from lango.shared.ast.nodes import (
    AddOperation,
    AndOperation,
    ConcatOperation,
    DivOperation,
    EqualOperation,
    GreaterEqualOperation,
    GreaterThanOperation,
    IndexOperation,
    LessEqualOperation,
    LessThanOperation,
    LetStatement,
    LiteralPattern,
    MulOperation,
    NegOperation,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    PowFloatOperation,
    PowIntOperation,
    Program,
    SubOperation,
)
from lango.shared.ast.transformer import Items, SharedTransformer, text


class ASTTransformer(SharedTransformer):
    # MiniO keeps escape sequences in string literals; they are interpreted
    # when the string is printed.

    def literal_pattern(self, items: Items) -> LiteralPattern:
        # MiniO literal patterns hold the plain value.
        return LiteralPattern(items[0].value)

    # --- Operators ----------------------------------------------------------------
    def add(self, items: Items) -> AddOperation:
        return AddOperation(items[0], items[1])

    def sub(self, items: Items) -> SubOperation:
        return SubOperation(items[0], items[1])

    def mul(self, items: Items) -> MulOperation:
        return MulOperation(items[0], items[1])

    def div(self, items: Items) -> DivOperation:
        return DivOperation(items[0], items[1])

    def pow_int(self, items: Items) -> PowIntOperation:
        return PowIntOperation(items[0], items[1])

    def pow_float(self, items: Items) -> PowFloatOperation:
        return PowFloatOperation(items[0], items[1])

    def eq(self, items: Items) -> EqualOperation:
        return EqualOperation(items[0], items[1])

    def neq(self, items: Items) -> NotEqualOperation:
        return NotEqualOperation(items[0], items[1])

    def lt(self, items: Items) -> LessThanOperation:
        return LessThanOperation(items[0], items[1])

    def lteq(self, items: Items) -> LessEqualOperation:
        return LessEqualOperation(items[0], items[1])

    def gt(self, items: Items) -> GreaterThanOperation:
        return GreaterThanOperation(items[0], items[1])

    def gteq(self, items: Items) -> GreaterEqualOperation:
        return GreaterEqualOperation(items[0], items[1])

    def and_op(self, items: Items) -> AndOperation:
        return AndOperation(items[0], items[1])

    def or_op(self, items: Items) -> OrOperation:
        return OrOperation(items[0], items[1])

    def not_op(self, items: Items) -> NotOperation:
        return NotOperation(items[0])

    def neg(self, items: Items) -> NegOperation:
        return NegOperation(items[0])

    def concat(self, items: Items) -> ConcatOperation:
        return ConcatOperation(items[0], items[1])

    def index(self, items: Items) -> IndexOperation:
        return IndexOperation(items[0], items[1])

    # --- Do blocks ----------------------------------------------------------------
    def let_block(self, items: Items) -> list[LetStatement]:
        return items

    def let_binding(self, items: Items) -> LetStatement:
        return LetStatement(text(items[0]), items[1])


def transform_parse_tree(tree: Tree) -> Program:
    return ASTTransformer().transform(tree)
