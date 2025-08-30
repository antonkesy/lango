from typing import Any, Dict, List, Optional, Set

from lango.systemo.ast.nodes import (
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataConstructor,
    DataDeclaration,
    DoBlock,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    InstanceDeclaration,
    IntLiteral,
    LetStatement,
    ListLiteral,
    ListPattern,
    LiteralPattern,
    NegativeFloat,
    NegativeInt,
    Pattern,
    Program,
    StringLiteral,
    SymbolicOperation,
    TupleLiteral,
    TuplePattern,
    Variable,
    VariablePattern,
    ArrowType,
    TypeConstructor,
    TypeVariable,
    TypeApplication,
    GroupedType,
    ListType,
)
from lango.systemo.ast.nodes import TupleType as ASTTupleType

from lango.shared.typechecker.lango_types import (
    DataType,
    FunctionType,
    Type,
    TypeApp,
    TypeCon,
    TypeVar,
    TupleType,
)

from dataclasses import dataclass


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


@dataclass
class TypedFunction:
    pattern: List[Pattern]


@dataclass
class TopLevelFunction:
    name: str
    overloads: dict[str, TypedFunction]  # Use string representation of types as keys


def collect_overloaded_functions(program: Program) -> Dict[str, TopLevelFunction]:
    overloaded_functions: Dict[str, TopLevelFunction] = {}

    for stmt in program.statements:
        if isinstance(stmt, FunctionDefinition):
            func_name = stmt.function_name
            func_type = stmt.ty
            if func_type is None:
                continue
            # Only accept FunctionType objects
            if not isinstance(func_type, FunctionType):
                raise NotImplementedError("TODO")
                continue
            typed_func = TypedFunction(pattern=stmt.patterns)
            if func_name not in overloaded_functions:
                overloaded_functions[func_name] = TopLevelFunction(
                    name=func_name, overloads={}
                )
            overloaded_functions[func_name].overloads[str(func_type)] = typed_func
        elif isinstance(stmt, InstanceDeclaration):
            func_name = stmt.instance_name
            func_type_expr = stmt.type_signature
            if func_type_expr is None:
                continue
            # Convert TypeExpression to Type
            func_type = parse_type_expr(func_type_expr)
            if func_type is None:
                continue
            # Only accept FunctionType objects
            if not isinstance(func_type, FunctionType):
                continue
            typed_func = TypedFunction(pattern=stmt.function_definition.patterns)
            if func_name not in overloaded_functions:
                overloaded_functions[func_name] = TopLevelFunction(
                    name=func_name, overloads={}
                )
            overloaded_functions[func_name].overloads[str(func_type)] = typed_func
    return overloaded_functions
