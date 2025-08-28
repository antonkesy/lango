"""
AST rewriter that applies operator precedence declarations to fix parsing.
"""

from typing import Dict, List, Optional, Tuple, Union

from lango.systemo.ast.nodes import (
    Associativity,
    BoolLiteral,
    Constructor,
    ConstructorExpression,
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
    NegativeFloat,
    NegativeInt,
    PrecedenceDeclaration,
    Program,
    Statement,
    StringLiteral,
    SymbolicOperation,
    Variable,
)


class PrecedenceRewriter:
    def __init__(self) -> None:
        self.precedences: Dict[str, Tuple[int, Associativity]] = {}

    def add_precedence(
        self,
        operator: str,
        precedence: int,
        associativity: Associativity,
    ) -> None:
        """Add a precedence declaration."""
        self.precedences[operator] = (precedence, associativity)

    def get_precedence(self, operator: str) -> int:
        """Get precedence for an operator, default to 0 if not declared."""
        return self.precedences.get(operator, (0, Associativity.LEFT))[0]

    def get_associativity(self, operator: str) -> Associativity:
        """Get associativity for an operator, default to left if not declared."""
        return self.precedences.get(operator, (0, Associativity.LEFT))[1]

    def rewrite_program(self, program: Program) -> Program:
        """Rewrite an entire program, first collecting precedence declarations."""
        # First pass: collect precedence declarations
        for stmt in program.statements:
            if isinstance(stmt, PrecedenceDeclaration):
                self.add_precedence(stmt.operator, stmt.precedence, stmt.associativity)

        # Second pass: rewrite expressions
        new_statements = []
        for stmt in program.statements:
            new_stmt = self.rewrite_statement(stmt)
            new_statements.append(new_stmt)

        return Program(new_statements)

    def rewrite_statement(self, stmt: Statement) -> Statement:
        """Rewrite a statement."""
        match stmt:
            case FunctionDefinition(
                function_name=name,
                patterns=patterns,
                body=body,
                ty=ty,
            ):
                new_body = self.rewrite_expression(body)
                return FunctionDefinition(name, patterns, new_body, ty)
            case InstanceDeclaration(
                instance_name=name,
                type_signature=sig,
                function_definition=func_def,
                ty=ty,
            ):
                new_func_def = self.rewrite_statement(func_def)
                if isinstance(new_func_def, FunctionDefinition):
                    return InstanceDeclaration(name, sig, new_func_def, ty)
                else:
                    return stmt
            case LetStatement(variable=var, value=value, ty=ty):
                new_value = self.rewrite_expression(value)
                return LetStatement(var, new_value, ty)
            case DataDeclaration() | PrecedenceDeclaration():
                return stmt
            case _:
                # Handle expression statements
                if hasattr(stmt, "__class__") and any(
                    isinstance(stmt, cls)
                    for cls in [
                        IntLiteral,
                        FloatLiteral,
                        StringLiteral,
                        BoolLiteral,
                        Variable,
                        Constructor,
                        SymbolicOperation,
                        IfElse,
                        DoBlock,
                        FunctionApplication,
                        ConstructorExpression,
                        GroupedExpression,
                        NegativeInt,
                        NegativeFloat,
                        ListLiteral,
                    ]
                ):
                    return self.rewrite_expression(stmt)
                return stmt

    def rewrite_expression(self, expr: Expression) -> Expression:
        """Rewrite an expression to respect precedence."""
        match expr:
            case SymbolicOperation(operator=op, operands=operands, ty=ty):
                # First rewrite operands recursively
                new_operands = [
                    self.rewrite_expression(operand) for operand in operands
                ]

                # If this is a binary operation, check if we need to restructure
                if len(new_operands) == 2:
                    return self.rewrite_binary_operation(
                        op,
                        new_operands[0],
                        new_operands[1],
                        ty,
                    )
                else:
                    return SymbolicOperation(op, new_operands, ty)

            case IfElse(
                condition=cond,
                then_expr=then_expr,
                else_expr=else_expr,
                ty=ty,
            ):
                new_cond = self.rewrite_expression(cond)
                new_then = self.rewrite_expression(then_expr)
                new_else = self.rewrite_expression(else_expr)
                return IfElse(new_cond, new_then, new_else, ty)

            case FunctionApplication(function=func, argument=arg, ty=ty):
                new_func = self.rewrite_expression(func)
                new_arg = self.rewrite_expression(arg)
                return FunctionApplication(new_func, new_arg, ty)

            case DoBlock(statements=stmts, ty=ty):
                new_stmts = [self.rewrite_statement(stmt) for stmt in stmts]
                return DoBlock(new_stmts, ty)

            case GroupedExpression(expression=inner_expr, ty=ty):
                new_inner = self.rewrite_expression(inner_expr)
                return GroupedExpression(new_inner, ty)

            case ListLiteral(elements=elements, ty=ty):
                new_elements = [self.rewrite_expression(elem) for elem in elements]
                return ListLiteral(new_elements, ty)

            case ConstructorExpression(constructor_name=name, fields=fields, ty=ty):
                new_fields = []
                for field in fields:
                    new_value = self.rewrite_expression(field.value)
                    new_fields.append(
                        field.__class__(field.field_name, new_value, field.ty),
                    )
                return ConstructorExpression(name, new_fields, ty)

            case _:
                # Leaf nodes (literals, variables, constructors)
                return expr

    def rewrite_binary_operation(
        self,
        op: str,
        left: Expression,
        right: Expression,
        ty,
    ) -> Expression:
        """Rewrite a binary operation, potentially restructuring based on precedence."""

        # If the right operand is also a binary operation, check precedence
        if isinstance(right, SymbolicOperation) and len(right.operands) == 2:
            right_op = right.operator
            left_prec = self.get_precedence(op)
            right_prec = self.get_precedence(right_op)

            # If current operator has higher precedence, or same precedence with left associativity,
            # we need to restructure
            if left_prec > right_prec or (
                left_prec == right_prec
                and self.get_associativity(op) == Associativity.LEFT
            ):

                # Restructure: (a op1 (b op2 c)) -> ((a op1 b) op2 c)
                # where op1 has higher precedence than op2, or same precedence with left associativity
                right_left = right.operands[0]
                right_right = right.operands[1]

                # Create new left operation: (a op1 b)
                new_left = SymbolicOperation(op, [left, right_left], None)
                # Create new right operation: ((a op1 b) op2 c)
                return SymbolicOperation(right_op, [new_left, right_right], ty)

        # If the left operand is also a binary operation, check precedence
        if isinstance(left, SymbolicOperation) and len(left.operands) == 2:
            left_op = left.operator
            left_prec = self.get_precedence(left_op)
            right_prec = self.get_precedence(op)

            # If current operator has higher precedence than left operator
            if right_prec > left_prec:
                # Restructure: ((a op1 b) op2 c) -> (a op1 (b op2 c))
                # where op2 has higher precedence than op1
                left_left = left.operands[0]
                left_right = left.operands[1]

                # Create new right operation: (b op2 c)
                new_right = SymbolicOperation(op, [left_right, right], None)
                # Create new left operation: (a op1 (b op2 c))
                return SymbolicOperation(left_op, [left_left, new_right], ty)

        # No restructuring needed
        return SymbolicOperation(op, [left, right], ty)


def rewrite_precedence(program: Program) -> Program:
    rewriter = PrecedenceRewriter()
    return rewriter.rewrite_program(program)
