from typing import Dict

from lango.systemo.ast.nodes import (
    BoolLiteral,
    Constructor,
    DataDeclaration,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    IntLiteral,
    ListLiteral,
    NegativeFloat,
    NegativeInt,
    Program,
    Statement,
    StringLiteral,
    SymbolicOperation,
    Variable,
    VariablePattern,
)


class TypeChecker:
    def __init__(self) -> None:
        self.var_types: Dict[str, str] = {}  # Simple string types for now
        self.function_types: Dict[str, str] = {}

    def check_program(self, ast: Program) -> bool:
        try:
            for stmt in ast.statements:
                self.check_statement(stmt)
            return True
        except Exception as e:
            print(f"Type checking failed: {e}")
            return False

    def check_statement(self, stmt: Statement) -> None:
        match stmt:
            case FunctionDefinition():
                # Set up parameter types (assume they're polymorphic for now)
                old_vars = self.var_types.copy()

                # Add parameters to scope - for simplicity, assume they can be any numeric type
                for pattern in stmt.patterns:
                    match pattern:
                        case VariablePattern():
                            self.var_types[pattern.name] = (
                                "Unknown"  # Will be inferred from usage
                            )

                # Try to infer body type
                try:
                    body_type = self.infer_expression(stmt.body)
                    self.function_types[stmt.function_name] = body_type
                except Exception:
                    # If inference fails, try with numeric assumptions
                    for pattern in stmt.patterns:
                        match pattern:
                            case VariablePattern():
                                self.var_types[pattern.name] = (
                                    "Int"  # Assume Int for arithmetic
                                )
                    try:
                        body_type = self.infer_expression(stmt.body)
                        self.function_types[stmt.function_name] = body_type
                    except Exception:
                        # Still failed, just register as unknown
                        self.function_types[stmt.function_name] = "Unknown"

                # Restore old variable scope
                self.var_types = old_vars

            case DataDeclaration():
                # Register constructors (simplified)
                for constructor in stmt.constructors:
                    self.function_types[constructor.name] = (
                        f"Constructor({stmt.type_name})"
                    )

    def infer_expression(self, expr: Expression) -> str:
        # Literals
        match expr:
            case IntLiteral() | NegativeInt():
                return "Int"
            case FloatLiteral() | NegativeFloat():
                return "Float"
            case StringLiteral():
                return "String"
            case BoolLiteral():
                return "Bool"
            case ListLiteral():
                if not expr.elements:
                    return "List[?]"
                element_type = self.infer_expression(expr.elements[0])
                # Check all elements have same type
                for elem in expr.elements[1:]:
                    elem_type = self.infer_expression(elem)
                    if elem_type != element_type:
                        raise TypeError(
                            f"List elements have different types: {element_type} vs {elem_type}",
                        )
                return f"List[{element_type}]"

            # Variables
            case Variable(name=var_name):
                if var_name in self.var_types:
                    return self.var_types[var_name]
                elif var_name in self.function_types:
                    return self.function_types[var_name]
                else:
                    # Assume it's a builtin or will be defined later
                    return "Unknown"

            case Constructor(name=constructor_name):
                if constructor_name in self.function_types:
                    return self.function_types[constructor_name]
                else:
                    return f"Constructor({constructor_name})"

            # Control flow
            case IfElse(
                condition=condition,
                then_expr=then_branch,
                else_expr=else_branch,
            ):
                cond_type = self.infer_expression(condition)
                if cond_type != "Bool":
                    raise TypeError(f"If condition must be Bool, got {cond_type}")

                then_type = self.infer_expression(then_branch)
                else_type = self.infer_expression(else_branch)

                if then_type != else_type:
                    raise TypeError(
                        f"If branches have different types: {then_type} vs {else_type}",
                    )

                return then_type

            # Function application
            case FunctionApplication(function=func_expr, argument=arg_expr):
                func_type = self.infer_expression(func_expr)
                arg_type = self.infer_expression(arg_expr)

                # Simplified: just return some result type
                if func_type.startswith("Constructor("):
                    return func_type
                else:
                    return "Unknown"  # Would need proper function type tracking

            # Grouping
            case GroupedExpression(expression=inner_expr):
                return self.infer_expression(inner_expr)

            case _:
                return "Unknown"


def type_check_ast(ast: Program) -> Dict[str, str]:
    checker = TypeChecker()
    if checker.check_program(ast):
        result = {}
        result.update(checker.var_types)
        result.update(checker.function_types)
        return result
    else:
        return {}
