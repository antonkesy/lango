"""
Simplified AST-based type checker that provides basic type checking for custom AST nodes.
This is a minimal implementation to demonstrate the concept.
"""

from typing import Dict

from lango.minio.ast_nodes import (
    AddOperation,
    AndOperation,
    BoolLiteral,
    ConcatOperation,
    Constructor,
    DataDeclaration,
    DivOperation,
    EqualOperation,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
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
    NegativeFloat,
    NegativeInt,
    NotEqualOperation,
    NotOperation,
    OrOperation,
    PowFloatOperation,
    PowIntOperation,
    Program,
    Statement,
    StringLiteral,
    SubOperation,
    Variable,
    VariablePattern,
)


class SimpleTypeChecker:
    """Simplified type checker for AST nodes."""

    def __init__(self) -> None:
        self.var_types: Dict[str, str] = {}  # Simple string types for now
        self.function_types: Dict[str, str] = {}

    def check_program(self, ast: Program) -> bool:
        """Type check an entire program."""
        try:
            for stmt in ast.statements:
                self.check_statement(stmt)
            return True
        except Exception as e:
            print(f"Type checking failed: {e}")
            return False

    def check_statement(self, stmt: Statement) -> None:
        """Check a single statement."""
        if isinstance(stmt, FunctionDefinition):
            # Set up parameter types (assume they're polymorphic for now)
            old_vars = self.var_types.copy()

            # Add parameters to scope - for simplicity, assume they can be any numeric type
            for pattern in stmt.patterns:
                if isinstance(pattern, VariablePattern):
                    self.var_types[pattern.name] = (
                        "Unknown"  # Will be inferred from usage
                    )

            # Try to infer body type
            try:
                body_type = self.infer_expression(stmt.body)
                self.function_types[stmt.function_name] = body_type
            except Exception as e:
                # If inference fails, try with numeric assumptions
                for pattern in stmt.patterns:
                    if isinstance(pattern, VariablePattern):
                        self.var_types[pattern.name] = (
                            "Int"  # Assume Int for arithmetic
                        )
                try:
                    body_type = self.infer_expression(stmt.body)
                    self.function_types[stmt.function_name] = body_type
                except:
                    # Still failed, just register as unknown
                    self.function_types[stmt.function_name] = "Unknown"

            # Restore old variable scope
            self.var_types = old_vars

        elif isinstance(stmt, DataDeclaration):
            # Register constructors (simplified)
            for constructor in stmt.constructors:
                self.function_types[constructor.name] = f"Constructor({stmt.type_name})"

    def infer_expression(self, expr: Expression) -> str:
        """Infer the type of an expression (simplified string-based types)."""

        # Literals
        if isinstance(expr, IntLiteral) or isinstance(expr, NegativeInt):
            return "Int"
        elif isinstance(expr, FloatLiteral) or isinstance(expr, NegativeFloat):
            return "Float"
        elif isinstance(expr, StringLiteral):
            return "String"
        elif isinstance(expr, BoolLiteral):
            return "Bool"
        elif isinstance(expr, ListLiteral):
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
        elif isinstance(expr, Variable):
            if expr.name in self.var_types:
                return self.var_types[expr.name]
            elif expr.name in self.function_types:
                return self.function_types[expr.name]
            else:
                # Assume it's a builtin or will be defined later
                return "Unknown"

        elif isinstance(expr, Constructor):
            if expr.name in self.function_types:
                return self.function_types[expr.name]
            else:
                return f"Constructor({expr.name})"

        # Arithmetic operations
        elif isinstance(expr, (AddOperation, SubOperation, MulOperation, DivOperation)):
            left_type = self.infer_expression(expr.left)
            right_type = self.infer_expression(expr.right)

            # Handle unknown types by trying to infer from context
            if left_type == "Unknown" and isinstance(expr.left, Variable):
                # If we're doing arithmetic, assume it's numeric
                if expr.left.name in self.var_types:
                    self.var_types[expr.left.name] = "Int"
                    left_type = "Int"

            if right_type == "Unknown" and isinstance(expr.right, Variable):
                # If we're doing arithmetic, assume it's numeric
                if expr.right.name in self.var_types:
                    self.var_types[expr.right.name] = "Int"
                    right_type = "Int"

            if left_type in ["Int", "Float"] and right_type in ["Int", "Float"]:
                # If either is Float, result is Float
                if left_type == "Float" or right_type == "Float":
                    return "Float"
                else:
                    return "Int"
            else:
                raise TypeError(
                    f"Arithmetic operation requires numeric types, got {left_type} and {right_type}",
                )

        elif isinstance(expr, (PowIntOperation, PowFloatOperation)):
            left_type = self.infer_expression(expr.left)
            right_type = self.infer_expression(expr.right)

            if left_type in ["Int", "Float"] and right_type in ["Int", "Float"]:
                return "Float" if isinstance(expr, PowFloatOperation) else "Int"
            else:
                raise TypeError(
                    f"Power operation requires numeric types, got {left_type} and {right_type}",
                )

        # Comparison operations
        elif isinstance(
            expr,
            (
                EqualOperation,
                NotEqualOperation,
                LessThanOperation,
                LessEqualOperation,
                GreaterThanOperation,
                GreaterEqualOperation,
            ),
        ):
            left_type = self.infer_expression(expr.left)
            right_type = self.infer_expression(expr.right)
            # For now, allow any comparison as long as types match
            if (
                left_type != right_type
                and left_type != "Unknown"
                and right_type != "Unknown"
            ):
                raise TypeError(f"Cannot compare {left_type} with {right_type}")
            return "Bool"

        # Logical operations
        elif isinstance(expr, (AndOperation, OrOperation)):
            left_type = self.infer_expression(expr.left)
            right_type = self.infer_expression(expr.right)
            if left_type != "Bool" or right_type != "Bool":
                raise TypeError(
                    f"Logical operation requires Bool operands, got {left_type} and {right_type}",
                )
            return "Bool"

        elif isinstance(expr, NotOperation):
            operand_type = self.infer_expression(expr.operand)
            if operand_type != "Bool":
                raise TypeError(
                    f"NOT operation requires Bool operand, got {operand_type}",
                )
            return "Bool"

        # String/List operations
        elif isinstance(expr, ConcatOperation):
            left_type = self.infer_expression(expr.left)
            right_type = self.infer_expression(expr.right)
            if left_type == right_type:
                return left_type
            else:
                raise TypeError(f"Cannot concatenate {left_type} with {right_type}")

        elif isinstance(expr, IndexOperation):
            list_type = self.infer_expression(expr.list_expr)
            index_type = self.infer_expression(expr.index_expr)

            if index_type != "Int":
                raise TypeError(f"List index must be Int, got {index_type}")

            if list_type.startswith("List[") and list_type.endswith("]"):
                element_type = list_type[5:-1]  # Extract element type
                return element_type
            else:
                raise TypeError(f"Index operation requires List, got {list_type}")

        # Control flow
        elif isinstance(expr, IfElse):
            cond_type = self.infer_expression(expr.condition)
            if cond_type != "Bool":
                raise TypeError(f"If condition must be Bool, got {cond_type}")

            then_type = self.infer_expression(expr.then_expr)
            else_type = self.infer_expression(expr.else_expr)

            if then_type != else_type:
                raise TypeError(
                    f"If branches have different types: {then_type} vs {else_type}",
                )

            return then_type

        # Function application
        elif isinstance(expr, FunctionApplication):
            func_type = self.infer_expression(expr.function)
            arg_type = self.infer_expression(expr.argument)

            # Simplified: just return some result type
            if func_type.startswith("Constructor("):
                return func_type
            else:
                return "Unknown"  # Would need proper function type tracking

        # Grouping
        elif isinstance(expr, GroupedExpression):
            return self.infer_expression(expr.expression)

        else:
            return "Unknown"


def type_check_ast(ast: Program) -> Dict[str, str]:
    """Simple type check that returns a dictionary of inferred types."""
    checker = SimpleTypeChecker()
    if checker.check_program(ast):
        # Return combined type information
        result = {}
        result.update(checker.var_types)
        result.update(checker.function_types)
        return result
    else:
        return {}
