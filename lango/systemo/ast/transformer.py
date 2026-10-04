from typing import Any

from lark import Tree

from lango.shared.ast.nodes import (
    Associativity,
    ConstrainedType,
    FunctionApplication,
    FunctionDefinition,
    IfElse,
    InstanceDeclaration,
    LetStatement,
    ListType,
    PrecedenceDeclaration,
    Program,
    SymbolicOperation,
    TypeConstraint,
    Variable,
)
from lango.shared.ast.transformer import Items, SharedTransformer, text

# Prefix minus is syntactic sugar for an application of the overloaded
# function ``negate`` (System O overloads functions, not syntax).
NEGATE = "negate"


def _unescape(text: str) -> str:
    return text.encode("utf-8").decode("unicode_escape")


class ASTTransformer(SharedTransformer):
    def literal_text(self, quoted: str) -> str:
        return _unescape(quoted[1:-1])

    # --- Names --------------------------------------------------------------
    def symbolic_var(self, items: Items) -> Variable:
        return Variable(text(items[0]))

    def operator_name(self, items: Items) -> str:
        return text(items[0])

    def inst_operator_name(self, items: Items) -> str:
        return text(items[0])

    def symbolic_operator(self, items: Items) -> str:
        return text(items[0])

    # --- Operators ----------------------------------------------------------
    def infix_op(self, items: Items) -> SymbolicOperation:
        left, operator, right = items
        return SymbolicOperation(operator, [left, right])

    def unary_minus(self, items: Items) -> FunctionApplication:
        return FunctionApplication(Variable(NEGATE), items[0])

    # --- Control flow -------------------------------------------------------
    def if_else_extended(self, items: Items) -> IfElse:
        condition, then_expr, elsif_clauses, else_expr = items
        result = else_expr
        for elsif_cond, elsif_then in reversed(elsif_clauses):
            result = IfElse(elsif_cond, elsif_then, result)
        return IfElse(condition, then_expr, result)

    def elsif_clauses(self, items: Items) -> list[tuple]:
        return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]

    def assignment_list(self, items: Items) -> list[LetStatement]:
        return items

    def assignment(self, items: Items) -> LetStatement:
        return LetStatement(text(items[0]), items[1])

    # --- Types --------------------------------------------------------------
    def list_type(self, items: Items) -> ListType:
        return ListType(items[0])

    def constraint(self, items: Items) -> TypeConstraint:
        return TypeConstraint(items[0], items[1])

    def constraints(self, items: Items) -> list[TypeConstraint]:
        return items

    def constrained_type(self, items: Items) -> ConstrainedType:
        return ConstrainedType(items[0], items[1])

    def type_scheme(self, items: Items) -> ConstrainedType:
        return ConstrainedType([], items[0])

    # --- Top-level declarations ---------------------------------------------
    def inst_func_def(self, items: Items) -> FunctionDefinition:
        return self.func_def(items)

    def inst_decl(self, items: Items) -> InstanceDeclaration:
        name = text(items[0])
        scheme = items[1]
        if not isinstance(scheme, ConstrainedType):
            scheme = ConstrainedType([], scheme)
        clauses: list[FunctionDefinition] = items[2:]
        for clause in clauses:
            if clause.function_name != name:
                raise ValueError(
                    f"Instance declaration for '{name}' defines '{clause.function_name}'",
                )
        return InstanceDeclaration(name, scheme, clauses)

    def _precedence(
        self,
        items: Items,
        associativity: Associativity,
    ) -> PrecedenceDeclaration:
        return PrecedenceDeclaration(
            int(text(items[0])),
            associativity,
            text(items[1]),
        )

    def infixl_decl(self, items: Items) -> PrecedenceDeclaration:
        return self._precedence(items, Associativity.LEFT)

    def infixr_decl(self, items: Items) -> PrecedenceDeclaration:
        return self._precedence(items, Associativity.RIGHT)

    def infix_decl(self, items: Items) -> PrecedenceDeclaration:
        return self._precedence(items, Associativity.NONE)


def transform_parse_tree(tree: Tree) -> Program:
    return ASTTransformer().transform(tree)
