"""Monotypes shared by MiniO and System O.

    tau ::= alpha | tau -> tau' | D tau_1 ... tau_n | (tau_1, ..., tau_n)

The primitive types ``Int``, ``Float``, ``Bool``, ``String``, ``Char`` and
``()`` are nullary type constructors (``TypeCon``).  System O represents
lists as ``DataType("List", (tau,))``; MiniO as ``TypeApp(TypeCon("List"), tau)``.
The two representations never meet: each type checker only sees its own.

Types are immutable and hashable.  ``TypeSubstitution``, ``TypeScheme``,
``FreshVarGenerator`` and ``generalize`` implement plain Hindley/Milner for
MiniO; System O has its own constrained schemes in
:mod:`lango.systemo.typechecker.types`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from lango.shared.typechecker.names import var_name

type Subst = dict[str, "Type"]


class Type(ABC):
    __slots__ = ()

    @abstractmethod
    def children(self) -> tuple[Type, ...]: ...

    @abstractmethod
    def substitute(self, subst: Subst) -> Type: ...

    @abstractmethod
    def __str__(self) -> str: ...

    def free_vars(self) -> set[str]:
        return set().union(*(child.free_vars() for child in self.children()))

    def apply_substitution(self, subst: TypeSubstitution) -> Type:
        return self.substitute(subst.mapping)


@dataclass(frozen=True, slots=True)
class TypeVar(Type):
    name: str

    def children(self) -> tuple[Type, ...]:
        return ()

    def free_vars(self) -> set[str]:
        return {self.name}

    def substitute(self, subst: Subst) -> Type:
        return subst.get(self.name, self)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class TypeCon(Type):
    name: str

    def children(self) -> tuple[Type, ...]:
        return ()

    def substitute(self, subst: Subst) -> Type:
        return self

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class TypeApp(Type):
    """Application of a type constructor (MiniO lists: ``List a``)."""

    constructor: Type
    argument: Type

    def children(self) -> tuple[Type, ...]:
        return (self.constructor, self.argument)

    def substitute(self, subst: Subst) -> Type:
        return TypeApp(
            self.constructor.substitute(subst),
            self.argument.substitute(subst),
        )

    def __str__(self) -> str:
        return f"({self.constructor} {self.argument})"


@dataclass(frozen=True, slots=True)
class FunctionType(Type):
    param: Type
    result: Type

    def children(self) -> tuple[Type, ...]:
        return (self.param, self.result)

    def substitute(self, subst: Subst) -> Type:
        return FunctionType(self.param.substitute(subst), self.result.substitute(subst))

    def __str__(self) -> str:
        if isinstance(self.param, FunctionType):
            return f"({self.param}) -> {self.result}"
        return f"{self.param} -> {self.result}"


@dataclass(frozen=True, slots=True)
class DataType(Type):
    name: str
    type_args: tuple[Type, ...] = ()

    def children(self) -> tuple[Type, ...]:
        return self.type_args

    def substitute(self, subst: Subst) -> Type:
        return DataType(
            self.name,
            tuple(arg.substitute(subst) for arg in self.type_args),
        )

    def __str__(self) -> str:
        return " ".join([self.name, *map(str, self.type_args)])


@dataclass(frozen=True, slots=True)
class TupleType(Type):
    element_types: tuple[Type, ...]

    def children(self) -> tuple[Type, ...]:
        return self.element_types

    def substitute(self, subst: Subst) -> Type:
        return TupleType(tuple(elem.substitute(subst) for elem in self.element_types))

    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self.element_types)) + ")"


def function(*types: Type) -> Type:
    """``function(a, b, c)`` is ``a -> b -> c``."""
    result = types[-1]
    for param in reversed(types[:-1]):
        result = FunctionType(param, result)
    return result


def unfold_function(t: Type) -> tuple[list[Type], Type]:
    """``a -> b -> c`` is ``([a, b], c)``."""
    params: list[Type] = []
    while isinstance(t, FunctionType):
        params.append(t.param)
        t = t.result
    return params, t


def ordered_free_vars(t: Type) -> list[str]:
    """Free type variables in order of first occurrence (left to right)."""
    result: list[str] = []

    def go(t: Type) -> None:
        match t:
            case TypeVar(name=name):
                if name not in result:
                    result.append(name)
            case _:
                for child in t.children():
                    go(child)

    go(t)
    return result


# Built-in types
INT_TYPE = TypeCon("Int")
CHAR_TYPE = TypeCon("Char")
STRING_TYPE = TypeCon("String")
FLOAT_TYPE = TypeCon("Float")
BOOL_TYPE = TypeCon("Bool")
UNIT_TYPE = TypeCon("()")  # For do blocks and putStr

PRIMITIVE_TYPES: dict[str, Type] = {
    t.name: t
    for t in (INT_TYPE, FLOAT_TYPE, BOOL_TYPE, STRING_TYPE, CHAR_TYPE, UNIT_TYPE)
}


# --------------------------------------------------------------------------
# Hindley/Milner schemes and substitutions (MiniO)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeSubstitution:
    mapping: Subst = field(default_factory=dict)

    def apply(self, t: Type) -> Type:
        return t.substitute(self.mapping)

    def compose(self, other: TypeSubstitution) -> TypeSubstitution:
        """``self`` after ``other``."""
        mapping = {var: self.apply(typ) for var, typ in other.mapping.items()}
        for var, typ in self.mapping.items():
            mapping.setdefault(var, typ)
        return TypeSubstitution(mapping)

    def __str__(self) -> str:
        if not self.mapping:
            return "∅"
        return (
            "{" + ", ".join(f"{var} ↦ {typ}" for var, typ in self.mapping.items()) + "}"
        )


@dataclass(frozen=True, slots=True)
class TypeScheme:
    quantified_vars: frozenset[str]
    type: Type

    def __init__(self, quantified_vars: set[str] | frozenset[str], type_: Type) -> None:
        object.__setattr__(self, "quantified_vars", frozenset(quantified_vars))
        object.__setattr__(self, "type", type_)

    def free_vars(self) -> set[str]:
        return self.type.free_vars() - self.quantified_vars

    def substitute(self, subst: TypeSubstitution) -> TypeScheme:
        mapping = {
            var: typ
            for var, typ in subst.mapping.items()
            if var not in self.quantified_vars
        }
        return TypeScheme(self.quantified_vars, self.type.substitute(mapping))

    def instantiate(self, fresh_var_gen: FreshVarGenerator) -> Type:
        mapping: Subst = {
            var: fresh_var_gen.fresh_var() for var in self.quantified_vars
        }
        return self.type.substitute(mapping)

    def __str__(self) -> str:
        if not self.quantified_vars:
            return str(self.type)
        return f"∀ {' '.join(sorted(self.quantified_vars))} . {self.type}"


class FreshVarGenerator:
    def __init__(self) -> None:
        self.counter = 0

    def fresh(self) -> str:
        name = var_name(self.counter)
        self.counter += 1
        return name

    def fresh_var(self) -> TypeVar:
        return TypeVar(self.fresh())


def generalize(type_env_free_vars: set[str], typ: Type) -> TypeScheme:
    return normalize_type_scheme(TypeScheme(typ.free_vars() - type_env_free_vars, typ))


def normalize_type_scheme(scheme: TypeScheme) -> TypeScheme:
    """Rename the type variables to ``a``, ``b``, ... in order of occurrence."""
    mapping: Subst = {
        old: TypeVar(var_name(i))
        for i, old in enumerate(ordered_free_vars(scheme.type))
    }
    quantified = {
        new.name
        for old, new in mapping.items()
        if old in scheme.quantified_vars and isinstance(new, TypeVar)
    }
    return TypeScheme(quantified, scheme.type.substitute(mapping))
