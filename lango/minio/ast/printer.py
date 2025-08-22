"""
AST printer for displaying type-annotated AST nodes.
"""

from typing import TYPE_CHECKING, Optional

from lango.minio.ast.nodes import (
    AddOperation,
    ASTNode,
    BoolLiteral,
    DivOperation,
    DoBlock,
    Expression,
    FloatLiteral,
    FunctionApplication,
    GroupedExpression,
    IfElse,
    IntLiteral,
    LetStatement,
    ListLiteral,
    MulOperation,
    Program,
    StringLiteral,
    SubOperation,
    Variable,
)

if TYPE_CHECKING:
    from lango.minio.typechecker.minio_types import Type


def format_type(ty: Optional["Type"]) -> str:
    """Format a type for display."""
    if ty is None:
        return "<?>"
    return str(ty)


def print_annotated_ast(ast: Program, indent: int = 0) -> None:
    _print_node(ast, indent)


def _print_node(node: ASTNode, indent: int = 0) -> None:
    prefix = "  " * indent
    type_str = format_type(getattr(node, "ty", None))

    if isinstance(node, Program):
        print(f"{prefix}Program :: {type_str}")
        for stmt in node.statements:
            _print_node(stmt, indent + 1)

    elif isinstance(node, IntLiteral):
        print(f"{prefix}IntLiteral({node.value}) :: {type_str}")

    elif isinstance(node, FloatLiteral):
        print(f"{prefix}FloatLiteral({node.value}) :: {type_str}")

    elif isinstance(node, StringLiteral):
        print(f'{prefix}StringLiteral("{node.value}") :: {type_str}')

    elif isinstance(node, BoolLiteral):
        print(f"{prefix}BoolLiteral({node.value}) :: {type_str}")

    elif isinstance(node, Variable):
        print(f"{prefix}Variable({node.name}) :: {type_str}")

    elif isinstance(node, AddOperation):
        print(f"{prefix}AddOperation :: {type_str}")
        print(f"{prefix}  left:")
        _print_node(node.left, indent + 2)
        print(f"{prefix}  right:")
        _print_node(node.right, indent + 2)

    elif isinstance(node, SubOperation):
        print(f"{prefix}SubOperation :: {type_str}")
        print(f"{prefix}  left:")
        _print_node(node.left, indent + 2)
        print(f"{prefix}  right:")
        _print_node(node.right, indent + 2)

    elif isinstance(node, MulOperation):
        print(f"{prefix}MulOperation :: {type_str}")
        print(f"{prefix}  left:")
        _print_node(node.left, indent + 2)
        print(f"{prefix}  right:")
        _print_node(node.right, indent + 2)

    elif isinstance(node, DivOperation):
        print(f"{prefix}DivOperation :: {type_str}")
        print(f"{prefix}  left:")
        _print_node(node.left, indent + 2)
        print(f"{prefix}  right:")
        _print_node(node.right, indent + 2)

    elif isinstance(node, FunctionApplication):
        print(f"{prefix}FunctionApplication :: {type_str}")
        print(f"{prefix}  function:")
        _print_node(node.function, indent + 2)
        print(f"{prefix}  argument:")
        _print_node(node.argument, indent + 2)

    elif isinstance(node, IfElse):
        print(f"{prefix}IfElse :: {type_str}")
        print(f"{prefix}  condition:")
        _print_node(node.condition, indent + 2)
        print(f"{prefix}  then:")
        _print_node(node.then_expr, indent + 2)
        print(f"{prefix}  else:")
        _print_node(node.else_expr, indent + 2)

    elif isinstance(node, DoBlock):
        print(f"{prefix}DoBlock :: {type_str}")
        for stmt in node.statements:
            _print_node(stmt, indent + 1)

    elif isinstance(node, LetStatement):
        print(f"{prefix}LetStatement({node.variable}) :: {type_str}")
        print(f"{prefix}  value:")
        _print_node(node.value, indent + 2)

    elif isinstance(node, ListLiteral):
        print(f"{prefix}ListLiteral :: {type_str}")
        for elem in node.elements:
            _print_node(elem, indent + 1)

    elif isinstance(node, GroupedExpression):
        print(f"{prefix}GroupedExpression :: {type_str}")
        _print_node(node.expression, indent + 1)

    else:
        # Generic fallback for other node types
        print(f"{prefix}{type(node).__name__} :: {type_str}")
        # Try to print any child nodes
        for attr_name, attr_value in vars(node).items():
            if attr_name == "ty":
                continue
            if isinstance(attr_value, ASTNode):
                print(f"{prefix}  {attr_name}:")
                _print_node(attr_value, indent + 2)
            elif (
                isinstance(attr_value, list)
                and attr_value
                and isinstance(attr_value[0], ASTNode)
            ):
                print(f"{prefix}  {attr_name}:")
                for item in attr_value:
                    _print_node(item, indent + 2)
