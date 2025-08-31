from typing import Dict, Optional

from lango.shared.typechecker.lango_types import (
    DataType,
    FunctionType,
    TupleType,
    Type,
    TypeApp,
    TypeCon,
    TypeVar,
)
from lango.systemo.ast.nodes import (
    ArrowType,
    FunctionDefinition,
    GroupedType,
    InstanceDeclaration,
    ListType,
    Program,
)
from lango.systemo.ast.nodes import TupleType as ASTTupleType
from lango.systemo.ast.nodes import (
    TypeApplication,
    TypeConstructor,
    TypeVariable,
)


def parse_type_expr(node) -> Optional[Type]:
    """Convert a TypeExpression to a Type."""
    if node is None:
        return None

    match node:
        case TypeConstructor(name=type_name):
            match type_name:
                case "Int":
                    return TypeCon("Int")
                case "String":
                    return TypeCon("String")
                case "Float":
                    return TypeCon("Float")
                case "Bool":
                    return TypeCon("Bool")
                case _:
                    return DataType(type_name, [])
        case TypeVariable(name=name):
            return TypeVar(name)
        case ArrowType(from_type=from_type, to_type=to_type):
            from_type_parsed = parse_type_expr(from_type)
            to_type_parsed = parse_type_expr(to_type)
            if from_type_parsed is None or to_type_parsed is None:
                return None
            return FunctionType(from_type_parsed, to_type_parsed)
        case TypeApplication(constructor=constructor, argument=argument):
            constructor_type = parse_type_expr(constructor)
            argument_type = parse_type_expr(argument)
            if constructor_type is None or argument_type is None:
                return None
            match constructor_type:
                case DataType(name=name, type_args=type_args):
                    new_args = type_args + [argument_type]
                    return DataType(name, new_args)
                case _:
                    return TypeApp(constructor_type, argument_type)
        case ListType(element_type=element_type):
            element_type_parsed = parse_type_expr(element_type)
            if element_type_parsed is None:
                return None
            return TypeApp(TypeCon("List"), element_type_parsed)
        case ASTTupleType(element_types=element_types):
            element_types_parsed = []
            for elem_type in element_types:
                parsed = parse_type_expr(elem_type)
                if parsed is None:
                    return None
                element_types_parsed.append(parsed)
            return TupleType(element_types_parsed)
        case GroupedType(type_expr=type_expr):
            return parse_type_expr(type_expr)
        case _:
            return None


def collect_all_functions(
    program: Program,
) -> Dict[str, Dict[str, FunctionDefinition]]:
    """Collect all functions from the program, including both regular and overloaded functions."""
    functions: Dict[str, Dict[str, FunctionDefinition]] = {}

    for stmt in program.statements:
        if isinstance(stmt, FunctionDefinition):
            func_name = stmt.function_name
            func_type = stmt.ty

            # For functions without explicit types, use "no type" as the key
            if func_type is None:
                type_key = ""
            elif isinstance(func_type, FunctionType):
                type_key = str(func_type)
            else:
                # For non-function types, convert to string
                type_key = str(func_type) if func_type else ""

            if func_name not in functions:
                functions[func_name] = {}
            functions[func_name][type_key] = stmt

        elif isinstance(stmt, InstanceDeclaration):
            func_name = stmt.instance_name
            func_type_expr = stmt.type_signature

            if func_type_expr is None:
                type_key = "no type annotation"
            else:
                # Convert TypeExpression to Type
                func_type = parse_type_expr(func_type_expr)
                if func_type is None:
                    type_key = "invalid type"
                else:
                    type_key = str(func_type)

            if func_name not in functions:
                functions[func_name] = {}
            functions[func_name][type_key] = stmt.function_definition
    return functions
