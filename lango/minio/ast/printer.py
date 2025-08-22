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

    match node:
        case Program(statements=statements):
            print(f"{prefix}Program :: {type_str}")
            for stmt in statements:
                _print_node(stmt, indent + 1)

        case IntLiteral(value=value):
            print(f"{prefix}IntLiteral({value}) :: {type_str}")

        case FloatLiteral(value=value):
            print(f"{prefix}FloatLiteral({value}) :: {type_str}")

        case StringLiteral(value=value):
            print(f'{prefix}StringLiteral("{value}") :: {type_str}')

        case BoolLiteral(value=value):
            print(f"{prefix}BoolLiteral({value}) :: {type_str}")

        case Variable(name=name):
            print(f"{prefix}Variable({name}) :: {type_str}")

        case AddOperation(left=left, right=right):
            print(f"{prefix}AddOperation :: {type_str}")
            print(f"{prefix}  left:")
            _print_node(left, indent + 2)
            print(f"{prefix}  right:")
            _print_node(right, indent + 2)

        case SubOperation(left=left, right=right):
            print(f"{prefix}SubOperation :: {type_str}")
            print(f"{prefix}  left:")
            _print_node(left, indent + 2)
            print(f"{prefix}  right:")
            _print_node(right, indent + 2)

        case MulOperation(left=left, right=right):
            print(f"{prefix}MulOperation :: {type_str}")
            print(f"{prefix}  left:")
            _print_node(left, indent + 2)
            print(f"{prefix}  right:")
            _print_node(right, indent + 2)

        case DivOperation(left=left, right=right):
            print(f"{prefix}DivOperation :: {type_str}")
            print(f"{prefix}  left:")
            _print_node(left, indent + 2)
            print(f"{prefix}  right:")
            _print_node(right, indent + 2)

        case FunctionApplication(function=function, argument=argument):
            print(f"{prefix}FunctionApplication :: {type_str}")
            print(f"{prefix}  function:")
            _print_node(function, indent + 2)
            print(f"{prefix}  argument:")
            _print_node(argument, indent + 2)

        case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
            print(f"{prefix}IfElse :: {type_str}")
            print(f"{prefix}  condition:")
            _print_node(condition, indent + 2)
            print(f"{prefix}  then:")
            _print_node(then_expr, indent + 2)
            print(f"{prefix}  else:")
            _print_node(else_expr, indent + 2)

        case DoBlock(statements=statements):
            print(f"{prefix}DoBlock :: {type_str}")
            for stmt in statements:
                _print_node(stmt, indent + 1)

        case LetStatement(variable=variable, value=value):
            print(f"{prefix}LetStatement({variable}) :: {type_str}")
            print(f"{prefix}  value:")
            _print_node(value, indent + 2)

        case ListLiteral(elements=elements):
            print(f"{prefix}ListLiteral :: {type_str}")
            for elem in elements:
                _print_node(elem, indent + 1)

        case GroupedExpression(expression=expression):
            print(f"{prefix}GroupedExpression :: {type_str}")
            _print_node(expression, indent + 1)

        case _:
            # Generic fallback for other node types
            print(f"{prefix}{type(node).__name__} :: {type_str}")
            # Try to print any child nodes
            for attr_name, attr_value in vars(node).items():
                if attr_name == "ty":
                    continue
                match attr_value:
                    case ASTNode():
                        print(f"{prefix}  {attr_name}:")
                        _print_node(attr_value, indent + 2)
                    case [ASTNode() as first, *rest]:
                        print(f"{prefix}  {attr_name}:")
                        for item in attr_value:
                            _print_node(item, indent + 2)
                    case _:
                        pass
