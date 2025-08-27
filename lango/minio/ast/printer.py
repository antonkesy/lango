from typing import Optional

from rich.console import Console

from lango.minio.ast.nodes import (
    AddOperation,
    AndOperation,
    ArrowType,
    ASTNode,
    BoolLiteral,
    ConcatOperation,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DivOperation,
    DoBlock,
    EqualOperation,
    Field,
    FieldAssignment,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GreaterEqualOperation,
    GreaterThanOperation,
    GroupedExpression,
    GroupedType,
    IfElse,
    IndexOperation,
    IntLiteral,
    LessEqualOperation,
    LessThanOperation,
    LetStatement,
    ListLiteral,
    LiteralPattern,
    MulOperation,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    PowFloatOperation,
    PowIntOperation,
    Program,
    RecordConstructor,
    StringLiteral,
    SubOperation,
    TypeApplication,
    TypeConstructor,
    TypeParameter,
    TypeVariable,
    Variable,
    VariablePattern,
)
from lango.minio.typechecker.minio_types import Type


def format_type(
    ty: Optional["Type"],
    show_types: bool = True,
    compact: bool = False,
) -> str:
    """Format a type for display."""
    if not show_types:
        return ""
    if ty is None:
        return " :: <?>" if not compact else ""
    return f" :: {ty}" if not compact else f": {ty}"


def print_annotated_ast(
    ast: Program,
    indent: int = 0,
    show_types: bool = True,
    compact: bool = False,
    max_depth: Optional[int] = None,
) -> None:
    """
    Print an AST with various formatting options.

    Args:
        ast: The AST to print
        indent: Starting indentation level
        show_types: Whether to show type annotations
        compact: Whether to use compact formatting
        max_depth: Maximum depth to print (None for unlimited)
    """
    _print_node(ast, indent, show_types, compact, max_depth, 0)


def _print_node(
    node: ASTNode,
    indent: int = 0,
    show_types: bool = True,
    compact: bool = False,
    max_depth: Optional[int] = None,
    current_depth: int = 0,
) -> None:
    if max_depth is not None and current_depth >= max_depth:
        print("  " * indent + "...")
        return

    prefix = "  " * indent
    type_str = format_type(getattr(node, "ty", None), show_types, compact)
    indent_step = 0 if compact else 1

    match node:
        case Program(statements=statements):
            print(f"{prefix}Program{type_str}")
            for stmt in statements:
                _print_node(
                    stmt,
                    indent + indent_step,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        # Literals
        case IntLiteral(value=value):
            print(f"{prefix}IntLiteral({value}){type_str}")

        case FloatLiteral(value=value):
            print(f"{prefix}FloatLiteral({value}){type_str}")

        case StringLiteral(value=value):
            print(f'{prefix}StringLiteral("{value}"){type_str}')

        case BoolLiteral(value=value):
            print(f"{prefix}BoolLiteral({value}){type_str}")

        case NegativeInt(value=value):
            print(f"{prefix}NegativeInt({value}){type_str}")

        case NegativeFloat(value=value):
            print(f"{prefix}NegativeFloat({value}){type_str}")

        case ListLiteral(elements=elements):
            print(f"{prefix}ListLiteral{type_str}")
            for elem in elements:
                _print_node(
                    elem,
                    indent + indent_step,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        # Variables and Constructors
        case Variable(name=name):
            print(f"{prefix}Variable({name}){type_str}")

        case Constructor(name=name):
            print(f"{prefix}Constructor({name}){type_str}")

        # Arithmetic Operations
        case AddOperation(left=left, right=right):
            print(f"{prefix}AddOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case SubOperation(left=left, right=right):
            print(f"{prefix}SubOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case MulOperation(left=left, right=right):
            print(f"{prefix}MulOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case DivOperation(left=left, right=right):
            print(f"{prefix}DivOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case PowIntOperation(left=left, right=right):
            print(f"{prefix}PowIntOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case PowFloatOperation(left=left, right=right):
            print(f"{prefix}PowFloatOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        # Comparison Operations
        case EqualOperation(left=left, right=right):
            print(f"{prefix}EqualOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case NotEqualOperation(left=left, right=right):
            print(f"{prefix}NotEqualOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case LessThanOperation(left=left, right=right):
            print(f"{prefix}LessThanOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case LessEqualOperation(left=left, right=right):
            print(f"{prefix}LessEqualOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case GreaterThanOperation(left=left, right=right):
            print(f"{prefix}GreaterThanOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case GreaterEqualOperation(left=left, right=right):
            print(f"{prefix}GreaterEqualOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        # Logical Operations
        case AndOperation(left=left, right=right):
            print(f"{prefix}AndOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case OrOperation(left=left, right=right):
            print(f"{prefix}OrOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case NotOperation(operand=operand):
            print(f"{prefix}NotOperation{type_str}")
            _print_node(
                operand,
                indent + indent_step,
                show_types,
                compact,
                max_depth,
                current_depth + 1,
            )

        # String/List Operations
        case ConcatOperation(left=left, right=right):
            print(f"{prefix}ConcatOperation{type_str}")
            if not compact:
                print(f"{prefix}  left:")
                _print_node(
                    left,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  right:")
                _print_node(
                    right,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    left,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    right,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case IndexOperation(list_expr=list_expr, index_expr=index_expr):
            print(f"{prefix}IndexOperation{type_str}")
            if not compact:
                print(f"{prefix}  list:")
                _print_node(
                    list_expr,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  index:")
                _print_node(
                    index_expr,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    list_expr,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    index_expr,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case FunctionApplication(function=function, argument=argument):
            print(f"{prefix}FunctionApplication{type_str}")
            if not compact:
                print(f"{prefix}  function:")
                _print_node(
                    function,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  argument:")
                _print_node(
                    argument,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    function,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    argument,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
            print(f"{prefix}IfElse{type_str}")
            if not compact:
                print(f"{prefix}  condition:")
                _print_node(
                    condition,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  then:")
                _print_node(
                    then_expr,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  else:")
                _print_node(
                    else_expr,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    condition,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    then_expr,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    else_expr,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case DoBlock(statements=statements):
            print(f"{prefix}DoBlock{type_str}")
            for stmt in statements:
                _print_node(
                    stmt,
                    indent + indent_step,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case LetStatement(variable=variable, value=value):
            print(f"{prefix}LetStatement({variable}){type_str}")
            if not compact:
                print(f"{prefix}  value:")
                _print_node(
                    value,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    value,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case GroupedExpression(expression=expression):
            print(f"{prefix}GroupedExpression{type_str}")
            _print_node(
                expression,
                indent + indent_step,
                show_types,
                compact,
                max_depth,
                current_depth + 1,
            )

        # Constructor Expressions
        case ConstructorExpression(constructor_name=constructor_name, fields=fields):
            print(f"{prefix}ConstructorExpression({constructor_name}){type_str}")
            for field_assignment in fields:
                _print_node(
                    field_assignment,
                    indent + indent_step,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case FieldAssignment(field_name=field_name, value=value):
            print(f"{prefix}FieldAssignment({field_name}){type_str}")
            _print_node(
                value,
                indent + indent_step,
                show_types,
                compact,
                max_depth,
                current_depth + 1,
            )

        # Type System
        case TypeConstructor(name=name):
            print(f"{prefix}TypeConstructor({name}){type_str}")

        case TypeVariable(name=name):
            print(f"{prefix}TypeVariable({name}){type_str}")

        case ArrowType(from_type=from_type, to_type=to_type):
            print(f"{prefix}ArrowType{type_str}")
            if not compact:
                print(f"{prefix}  from:")
                _print_node(
                    from_type,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  to:")
                _print_node(
                    to_type,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    from_type,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    to_type,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case TypeApplication(constructor=constructor, argument=argument):
            print(f"{prefix}TypeApplication{type_str}")
            if not compact:
                print(f"{prefix}  constructor:")
                _print_node(
                    constructor,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  argument:")
                _print_node(
                    argument,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    constructor,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    argument,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case GroupedType(type_expr=type_expr):
            print(f"{prefix}GroupedType{type_str}")
            _print_node(
                type_expr,
                indent + indent_step,
                show_types,
                compact,
                max_depth,
                current_depth + 1,
            )

        # Patterns
        case ConstructorPattern(constructor=constructor, patterns=patterns):
            print(f"{prefix}ConstructorPattern({constructor}){type_str}")
            if patterns:
                if not compact:
                    print(f"{prefix}  patterns:")
                    for pattern in patterns:
                        _print_node(
                            pattern,
                            indent + 2,
                            show_types,
                            compact,
                            max_depth,
                            current_depth + 1,
                        )
                else:
                    for pattern in patterns:
                        _print_node(
                            pattern,
                            indent + 1,
                            show_types,
                            compact,
                            max_depth,
                            current_depth + 1,
                        )

        case ConsPattern(head=head, tail=tail):
            print(f"{prefix}ConsPattern{type_str}")
            if not compact:
                print(f"{prefix}  head:")
                _print_node(
                    head,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                print(f"{prefix}  tail:")
                _print_node(
                    tail,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    head,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
                _print_node(
                    tail,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case VariablePattern(name=name):
            print(f"{prefix}VariablePattern({name}){type_str}")

        case LiteralPattern(value=value):
            print(f"{prefix}LiteralPattern({value}){type_str}")

        case NegativeIntPattern(value=value):
            print(f"{prefix}NegativeIntPattern({value}){type_str}")

        case NegativeFloatPattern(value=value):
            print(f"{prefix}NegativeFloatPattern({value}){type_str}")

        # Data Declarations
        case DataDeclaration(
            type_name=type_name,
            type_params=type_params,
            constructors=constructors,
        ):
            print(f"{prefix}DataDeclaration({type_name}){type_str}")
            if type_params and not compact:
                print(f"{prefix}  type_params:")
                for param in type_params:
                    _print_node(
                        param,
                        indent + 2,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
            elif type_params:
                for param in type_params:
                    _print_node(
                        param,
                        indent + 1,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )

            if constructors and not compact:
                print(f"{prefix}  constructors:")
                for data_constructor in constructors:
                    _print_node(
                        data_constructor,
                        indent + 2,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
            elif constructors:
                for data_constructor in constructors:
                    _print_node(
                        data_constructor,
                        indent + 1,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )

        case DataConstructor(
            name=name,
            record_constructor=record_constructor,
            type_atoms=type_atoms,
        ):
            print(f"{prefix}DataConstructor({name}){type_str}")
            if record_constructor:
                if not compact:
                    print(f"{prefix}  record:")
                    _print_node(
                        record_constructor,
                        indent + 2,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
                else:
                    _print_node(
                        record_constructor,
                        indent + 1,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
            elif type_atoms:
                if not compact:
                    print(f"{prefix}  type_atoms:")
                    for atom in type_atoms:
                        _print_node(
                            atom,
                            indent + 2,
                            show_types,
                            compact,
                            max_depth,
                            current_depth + 1,
                        )
                else:
                    for atom in type_atoms:
                        _print_node(
                            atom,
                            indent + 1,
                            show_types,
                            compact,
                            max_depth,
                            current_depth + 1,
                        )

        case RecordConstructor(fields=fields):
            print(f"{prefix}RecordConstructor{type_str}")
            for field in fields:
                _print_node(
                    field,
                    indent + indent_step,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case Field(name=name, field_type=field_type):
            print(f"{prefix}Field({name}){type_str}")
            _print_node(
                field_type,
                indent + indent_step,
                show_types,
                compact,
                max_depth,
                current_depth + 1,
            )

        case TypeParameter(name=name):
            print(f"{prefix}TypeParameter({name}){type_str}")

        case FunctionDefinition(
            function_name=function_name,
            patterns=patterns,
            body=body,
        ):
            print(f"{prefix}FunctionDefinition({function_name}){type_str}")
            if patterns and not compact:
                print(f"{prefix}  patterns:")
                for pattern in patterns:
                    _print_node(
                        pattern,
                        indent + 2,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
            elif patterns:
                for pattern in patterns:
                    _print_node(
                        pattern,
                        indent + 1,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )

            if not compact:
                print(f"{prefix}  body:")
                _print_node(
                    body,
                    indent + 2,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )
            else:
                _print_node(
                    body,
                    indent + 1,
                    show_types,
                    compact,
                    max_depth,
                    current_depth + 1,
                )

        case _:
            # Generic fallback for other node types
            node_name = type(node).__name__
            print(f"{prefix}{node_name}{type_str}")
            # Try to print any child nodes
            for attr_name, attr_value in vars(node).items():
                if attr_name == "ty":
                    continue
                match attr_value:
                    case ASTNode():
                        if not compact:
                            print(f"{prefix}  {attr_name}:")
                            _print_node(
                                attr_value,
                                indent + 2,
                                show_types,
                                compact,
                                max_depth,
                                current_depth + 1,
                            )
                        else:
                            _print_node(
                                attr_value,
                                indent + 1,
                                show_types,
                                compact,
                                max_depth,
                                current_depth + 1,
                            )
                    case [first, *rest]:
                        if first is not None and isinstance(first, ASTNode):
                            if not compact:
                                print(f"{prefix}  {attr_name}:")
                                for item in attr_value:
                                    _print_node(
                                        item,
                                        indent + 2,
                                        show_types,
                                        compact,
                                        max_depth,
                                        current_depth + 1,
                                    )
                            else:
                                for item in attr_value:
                                    _print_node(
                                        item,
                                        indent + 1,
                                        show_types,
                                        compact,
                                        max_depth,
                                        current_depth + 1,
                                    )
                    case _:
                        pass


def print_ast_compact(ast: Program, show_types: bool = False) -> None:
    """Print AST in compact format without type annotations."""
    print_annotated_ast(ast, show_types=show_types, compact=True)


def print_ast_summary(ast: Program, max_depth: int = 3) -> None:
    """Print AST summary with limited depth."""
    print_annotated_ast(ast, show_types=False, max_depth=max_depth)


def print_ast_types_only(ast: Program) -> None:
    """Print AST showing only type information."""
    print_annotated_ast(ast, show_types=True, compact=True)


def print_ast_colored(
    ast: Program,
    show_types: bool = True,
    compact: bool = False,
) -> None:
    """Print AST with colors using Rich (if available)."""
    if not RICH_AVAILABLE:
        print("Rich not available, falling back to plain text")
        print_annotated_ast(ast, show_types=show_types, compact=compact)
        return

    try:
        from rich.console import Console

        console = Console()
        _print_node_colored(ast, console, show_types=show_types, compact=compact)
    except ImportError:
        print_annotated_ast(ast, show_types=show_types, compact=compact)


def _print_node_colored(
    node: ASTNode,
    console: "Console",
    indent: int = 0,
    show_types: bool = True,
    compact: bool = False,
    max_depth: Optional[int] = None,
    current_depth: int = 0,
) -> None:
    """Print a node with Rich colors."""
    if not RICH_AVAILABLE:
        return

    if max_depth is not None and current_depth >= max_depth:
        console.print("  " * indent + "...", style="dim")
        return

    try:
        from rich.text import Text
    except ImportError:
        return

    prefix = "  " * indent
    type_str = format_type(getattr(node, "ty", None), show_types, compact)
    indent_step = 0 if compact else 1

    # Define color scheme
    node_colors = {
        "literal": "cyan",
        "variable": "green",
        "constructor": "blue",
        "operation": "yellow",
        "control": "magenta",
        "type": "red",
        "pattern": "bright_blue",
        "declaration": "bright_magenta",
    }

    def get_node_style(node_type: str) -> str:
        if "Literal" in node_type:
            return node_colors["literal"]
        elif "Variable" in node_type or "Constructor" in node_type:
            return node_colors["variable"]
        elif "Operation" in node_type:
            return node_colors["operation"]
        elif "If" in node_type or "Do" in node_type or "Let" in node_type:
            return node_colors["control"]
        elif "Type" in node_type:
            return node_colors["type"]
        elif "Pattern" in node_type:
            return node_colors["pattern"]
        elif "Declaration" in node_type or "Definition" in node_type:
            return node_colors["declaration"]
        else:
            return "white"

    node_name = type(node).__name__
    style = get_node_style(node_name)

    # Create rich text with colors
    text = Text()
    text.append(prefix, style="dim")
    text.append(node_name, style=style)

    # Add node-specific info with safe attribute access
    if hasattr(node, "name") and getattr(node, "name", None):
        text.append(f"({getattr(node, 'name')})", style="bright_white")
    elif hasattr(node, "function_name") and getattr(node, "function_name", None):
        text.append(f"({getattr(node, 'function_name')})", style="bright_white")
    elif hasattr(node, "type_name") and getattr(node, "type_name", None):
        text.append(f"({getattr(node, 'type_name')})", style="bright_white")
    elif hasattr(node, "variable") and getattr(node, "variable", None):
        text.append(f"({getattr(node, 'variable')})", style="bright_white")
    elif hasattr(node, "value"):
        value = getattr(node, "value", None)
        if isinstance(value, str):
            text.append(f'("{value}")', style="bright_white")
        elif value is not None:
            text.append(f"({value})", style="bright_white")

    # Add type annotation
    if show_types and type_str:
        text.append(type_str, style="dim red")

    console.print(text)

    # Print children recursively
    for attr_name, attr_value in vars(node).items():
        if attr_name in [
            "ty",
            "name",
            "function_name",
            "type_name",
            "variable",
            "value",
        ]:
            continue

        match attr_value:
            case ASTNode():
                if not compact and attr_name not in ["expression", "body"]:
                    console.print(f"{prefix}  {attr_name}:", style="dim")
                    _print_node_colored(
                        attr_value,
                        console,
                        indent + 2,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
                else:
                    _print_node_colored(
                        attr_value,
                        console,
                        indent + indent_step,
                        show_types,
                        compact,
                        max_depth,
                        current_depth + 1,
                    )
            case list() if attr_value and isinstance(attr_value[0], ASTNode):
                if not compact:
                    console.print(f"{prefix}  {attr_name}:", style="dim")
                    for item in attr_value:
                        _print_node_colored(
                            item,
                            console,
                            indent + 2,
                            show_types,
                            compact,
                            max_depth,
                            current_depth + 1,
                        )
                else:
                    for item in attr_value:
                        _print_node_colored(
                            item,
                            console,
                            indent + indent_step,
                            show_types,
                            compact,
                            max_depth,
                            current_depth + 1,
                        )
