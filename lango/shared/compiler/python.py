"""
Shared Python code generation utilities for both systemo and minio compilers.
"""

from typing import Any, List, Optional


def build_record_pattern_match(
    value_expr: str,
    constructor: str,
    field_name: str,
    var_name: str,
    body: Any,
    compile_expression_func,
) -> str:
    """Build a readable pattern match for record constructors."""
    lines = [
        f"match {value_expr}:",
        f"        case {constructor}() if hasattr({value_expr}, 'fields') and '{field_name}' in {value_expr}.fields:",
        f"            {var_name} = {value_expr}.fields['{field_name}']",
        f"            return {compile_expression_func(body)}",
        "        case _:",
        "            raise ValueError('Pattern match failed')",
    ]
    return "\n".join(lines)


def build_positional_pattern_match(
    value_expr: str,
    constructor: str,
    var_name: str,
    body: Any,
    compile_expression_func,
    arg_index: int = 0,
) -> str:
    """Build a readable pattern match for positional constructors."""
    lines = [
        f"match {value_expr}:",
        f"        case {constructor}():",
        f"            {var_name} = {value_expr}.arg_{arg_index}",
        f"            return {compile_expression_func(body)}",
        "        case _:",
        "            raise ValueError('Pattern match failed')",
    ]
    return "\n".join(lines)


def build_multi_arg_pattern_match(
    value_expr: str,
    constructor: str,
    assignments: List[str],
    body: Any,
    compile_expression_func,
) -> str:
    """Build a readable pattern match for constructors with multiple arguments."""
    lines = [f"match {value_expr}:", f"        case {constructor}():"]
    for assignment in assignments:
        lines.append(f"            {assignment}")
    lines.extend(
        [
            f"            return {compile_expression_func(body)}",
            "        case _:",
            "            raise ValueError('Pattern match failed')",
        ],
    )
    return "\n".join(lines)


def build_literal_pattern_match(
    value_expr: str,
    value: Any,
    body: Any,
    compile_expression_func,
    compile_literal_value_func,
) -> str:
    """Build a readable pattern match for literal values."""
    lines = [
        f"if {value_expr} == {compile_literal_value_func(value)}:",
        f"        return {compile_expression_func(body)}",
        "    else:",
        "        raise ValueError('Pattern match failed')",
    ]
    return "\n".join(lines)


def build_cons_pattern_match(
    value_expr: str,
    head_var: Optional[str],
    tail_var: Optional[str],
    body: Any,
    compile_expression_func,
) -> str:
    """Build a readable pattern match for cons patterns (x:xs)."""
    assignments = []
    if head_var:
        assignments.append(f"        {head_var} = {value_expr}[0]")
    if tail_var:
        assignments.append(f"        {tail_var} = {value_expr}[1:]")

    lines = [f"if len({value_expr}) > 0:"]
    lines.extend(assignments)
    lines.extend(
        [
            f"        return {compile_expression_func(body)}",
            "    else:",
            "        raise ValueError('Pattern match failed')",
        ],
    )
    return "\n".join(lines)


def build_tuple_pattern_match(
    value_expr: str,
    tuple_vars: List[str],
    body: Any,
    compile_expression_func,
) -> str:
    """Build a readable pattern match for tuple patterns."""
    assignments = []
    for i, var in enumerate(tuple_vars):
        if not var.startswith("_tuple_elem_"):
            assignments.append(f"        {var} = {value_expr}[{i}]")

    lines = [f"if len({value_expr}) == {len(tuple_vars)}:"]
    lines.extend(assignments)
    lines.extend(
        [
            f"        return {compile_expression_func(body)}",
            "    else:",
            "        raise ValueError('Pattern match failed')",
        ],
    )
    return "\n".join(lines)


def build_list_pattern_match(
    value_expr: str,
    list_vars: List[str],
    body: Any,
    compile_expression_func,
) -> str:
    """Build a readable pattern match for list patterns."""
    assignments = []
    for i, var in enumerate(list_vars):
        if not var.startswith("_list_elem_"):
            assignments.append(f"        {var} = {value_expr}[{i}]")

    lines = [f"if len({value_expr}) == {len(list_vars)}:"]
    lines.extend(assignments)
    lines.extend(
        [
            f"        return {compile_expression_func(body)}",
            "    else:",
            "        raise ValueError('Pattern match failed')",
        ],
    )
    return "\n".join(lines)


def build_simple_pattern_match(
    value_expr: str,
    constructor: str,
    body: Any,
    compile_expression_func,
) -> str:
    """Build a readable pattern match for constructors with no arguments."""
    lines = [
        f"match {value_expr}:",
        f"        case {constructor}():",
        f"            return {compile_expression_func(body)}",
        "        case _:",
        "            raise ValueError('Pattern match failed')",
    ]
    return "\n".join(lines)


def compile_literal_value(value: Any) -> str:
    """Compile a literal value to Python code."""
    match value:
        case str():
            return f'"{value}"'
        case bool():
            return str(value)
        case list():
            return "[]"  # Handle empty list explicitly
        case _:
            return str(value)


def is_expression_systemo(stmt: Any) -> bool:
    """Check if a statement is an expression that can be compiled (systemo version)."""
    # Import here to avoid circular imports
    from lango.systemo.ast.nodes import (
        BoolLiteral,
        CharLiteral,
        Constructor,
        ConstructorExpression,
        DoBlock,
        FloatLiteral,
        FunctionApplication,
        GroupedExpression,
        IfElse,
        IntLiteral,
        ListLiteral,
        StringLiteral,
        SymbolicOperation,
        Variable,
    )

    match stmt:
        case (
            IntLiteral()
            | FloatLiteral()
            | StringLiteral()
            | CharLiteral()
            | BoolLiteral()
            | ListLiteral()
            | Variable()
            | Constructor()
            | SymbolicOperation()
            | FunctionApplication()
            | ConstructorExpression()
            | DoBlock()
            | GroupedExpression()
            | IfElse()
        ):
            return True
        case _:
            return False


def is_expression_minio(stmt: Any) -> bool:
    """Check if a statement is an expression that can be compiled (minio version)."""
    # Import here to avoid circular imports
    from lango.minio.ast.nodes import (
        AddOperation,
        AndOperation,
        BoolLiteral,
        CharLiteral,
        ConcatOperation,
        Constructor,
        ConstructorExpression,
        DivOperation,
        DoBlock,
        EqualOperation,
        FloatLiteral,
        FunctionApplication,
        GreaterEqualOperation,
        GreaterThanOperation,
        GroupedExpression,
        IfElse,
        IndexOperation,
        IntLiteral,
        LessEqualOperation,
        LessThanOperation,
        ListLiteral,
        MulOperation,
        NegOperation,
        NotEqualOperation,
        NotOperation,
        OrOperation,
        StringLiteral,
        SubOperation,
        Variable,
    )

    match stmt:
        case (
            IntLiteral()
            | FloatLiteral()
            | StringLiteral()
            | CharLiteral()
            | BoolLiteral()
            | ListLiteral()
            | Variable()
            | Constructor()
            | AddOperation()
            | SubOperation()
            | MulOperation()
            | DivOperation()
            | EqualOperation()
            | NotEqualOperation()
            | LessThanOperation()
            | LessEqualOperation()
            | GreaterThanOperation()
            | GreaterEqualOperation()
            | ConcatOperation()
            | AndOperation()
            | OrOperation()
            | NotOperation()
            | NegOperation()
            | IndexOperation()
            | IfElse()
            | FunctionApplication()
            | ConstructorExpression()
            | DoBlock()
            | GroupedExpression()
        ):
            return True
        case _:
            return False


# Keep the generic one for backward compatibility
def is_expression(stmt: Any) -> bool:
    """Check if a statement is an expression that can be compiled."""
    return is_expression_systemo(stmt)


def compile_expression_safe(stmt: Any, compile_expression_func) -> str:
    """Safely compile an expression statement."""
    return (
        compile_expression_func(stmt)
        if is_expression_systemo(stmt) or is_expression_minio(stmt)
        else "None"
    )
