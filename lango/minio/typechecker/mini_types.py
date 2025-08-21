"""
Type representations for the Hindley-Milner type system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


class Type(ABC):
    """Base class for all types"""

    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Return the set of free type variables in this type"""
        pass

    @abstractmethod
    def substitute(self, subst: Dict[str, "Type"]) -> "Type":
        """Apply a substitution to this type"""
        pass

    def apply_substitution(self, subst: "TypeSubstitution") -> "Type":
        """Apply a TypeSubstitution to this type"""
        return self.substitute(subst.mapping)

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass(frozen=True)
class TypeVar(Type):
    """Type variable (e.g., 'a', 'b')"""

    name: str

    def free_vars(self) -> Set[str]:
        return {self.name}

    def substitute(self, subst: Dict[str, Type]) -> Type:
        return subst.get(self.name, self)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeCon(Type):
    """Type constructor (e.g., Int, String, Bool, Float)"""

    name: str

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, subst: Dict[str, Type]) -> Type:
        return self

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeApp(Type):
    """Type application (e.g., List a, Maybe Int)"""

    constructor: Type
    argument: Type

    def free_vars(self) -> Set[str]:
        return self.constructor.free_vars() | self.argument.free_vars()

    def substitute(self, subst: Dict[str, Type]) -> Type:
        return TypeApp(
            self.constructor.substitute(subst),
            self.argument.substitute(subst),
        )

    def __str__(self) -> str:
        return f"({self.constructor} {self.argument})"


@dataclass(frozen=True)
class FunctionType(Type):
    """Function type (e.g., Int -> String, a -> b -> c)"""

    param: Type
    result: Type

    def free_vars(self) -> Set[str]:
        return self.param.free_vars() | self.result.free_vars()

    def substitute(self, subst: Dict[str, Type]) -> Type:
        return FunctionType(self.param.substitute(subst), self.result.substitute(subst))

    def __str__(self) -> str:
        # Handle right associativity of function types
        if isinstance(self.param, FunctionType):
            return f"({self.param}) -> {self.result}"
        else:
            return f"{self.param} -> {self.result}"


@dataclass(frozen=True)
class DataType(Type):
    """Custom data type (e.g., Person, Maybe a)"""

    name: str
    type_args: List[Type]

    def free_vars(self) -> Set[str]:
        result = set()
        for arg in self.type_args:
            result |= arg.free_vars()
        return result

    def substitute(self, subst: Dict[str, Type]) -> Type:
        return DataType(self.name, [arg.substitute(subst) for arg in self.type_args])

    def __str__(self) -> str:
        if not self.type_args:
            return self.name
        args_str = " ".join(str(arg) for arg in self.type_args)
        return f"{self.name} {args_str}"


# Built-in types
INT_TYPE = TypeCon("Int")
STRING_TYPE = TypeCon("String")
FLOAT_TYPE = TypeCon("Float")
BOOL_TYPE = TypeCon("Bool")
UNIT_TYPE = TypeCon("()")  # For do blocks and putStr


class TypeSubstitution:
    """Represents a type substitution (mapping from type variables to types)"""

    def __init__(self, mapping: Optional[Dict[str, Type]] = None):
        self.mapping = mapping or {}

    def apply(self, t: Type) -> Type:
        """Apply this substitution to a type"""
        return t.substitute(self.mapping)

    def compose(self, other: "TypeSubstitution") -> "TypeSubstitution":
        """Compose two substitutions: (self ∘ other)"""
        new_mapping = {}

        # Apply self to all mappings in other
        for var, typ in other.mapping.items():
            new_mapping[var] = self.apply(typ)

        # Add mappings from self that aren't in other
        for var, typ in self.mapping.items():
            if var not in new_mapping:
                new_mapping[var] = typ

        return TypeSubstitution(new_mapping)

    def __str__(self) -> str:
        if not self.mapping:
            return "∅"
        items = [f"{var} ↦ {typ}" for var, typ in self.mapping.items()]
        return "{" + ", ".join(items) + "}"


class TypeScheme:
    """Polymorphic type scheme (∀ a₁ a₂ ... aₙ . τ)"""

    def __init__(self, quantified_vars: Set[str], type_: Type):
        self.quantified_vars = quantified_vars
        self.type = type_

    def free_vars(self) -> Set[str]:
        """Free variables are those in the type minus the quantified ones"""
        return self.type.free_vars() - self.quantified_vars

    def substitute(self, subst: TypeSubstitution) -> "TypeScheme":
        """Apply substitution, being careful not to substitute quantified variables"""
        # Remove quantified variables from the substitution
        filtered_mapping = {
            var: typ
            for var, typ in subst.mapping.items()
            if var not in self.quantified_vars
        }
        filtered_subst = TypeSubstitution(filtered_mapping)

        return TypeScheme(self.quantified_vars, filtered_subst.apply(self.type))

    def instantiate(self, fresh_var_gen) -> Type:
        """Create a fresh instance of this type scheme by replacing quantified variables"""
        if not self.quantified_vars:
            return self.type

        subst_mapping: Dict[str, Type] = {}
        for var in self.quantified_vars:
            fresh_var = fresh_var_gen.fresh()
            subst_mapping[var] = TypeVar(fresh_var)

        subst = TypeSubstitution(subst_mapping)
        return subst.apply(self.type)

    def __str__(self) -> str:
        if not self.quantified_vars:
            return str(self.type)
        vars_str = " ".join(sorted(self.quantified_vars))
        return f"∀ {vars_str} . {self.type}"


class FreshVarGenerator:
    """Generates fresh type variables"""

    def __init__(self):
        self.counter = 0

    def fresh(self) -> str:
        """Generate a fresh type variable name"""
        name = f"t{self.counter}"
        self.counter += 1
        return name


def generalize(type_env_free_vars: Set[str], typ: Type) -> TypeScheme:
    """Generalize a type by quantifying over variables not free in the environment"""
    free_in_type = typ.free_vars()
    quantified = free_in_type - type_env_free_vars
    return TypeScheme(quantified, typ)
