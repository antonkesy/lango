"""
Unification algorithm for the Hindley-Milner type system
"""

from typing import List, Tuple

from .types import (
    DataType,
    FunctionType,
    Type,
    TypeApp,
    TypeCon,
    TypeSubstitution,
    TypeVar,
)


class UnificationError(Exception):
    pass


def occurs_check(var: str, typ: Type) -> bool:
    """Check if a type variable occurs within a type (prevents infinite types)"""
    if isinstance(typ, TypeVar):
        return var == typ.name
    elif isinstance(typ, TypeCon):
        return False
    elif isinstance(typ, FunctionType):
        return occurs_check(var, typ.param) or occurs_check(var, typ.result)
    elif isinstance(typ, TypeApp):
        return occurs_check(var, typ.constructor) or occurs_check(var, typ.argument)
    elif isinstance(typ, DataType):
        return any(occurs_check(var, arg) for arg in typ.type_args)
    else:
        raise UnificationError(f"Unknown type in occurs check: {type(typ)}")


def unify_one(t1: Type, t2: Type) -> TypeSubstitution:
    """Unify two types and return the most general unifier"""

    # Same type
    if t1 == t2:
        return TypeSubstitution()

    # Type variable cases
    if isinstance(t1, TypeVar):
        if occurs_check(t1.name, t2):
            raise UnificationError(f"Occurs check failed: {t1.name} occurs in {t2}")
        return TypeSubstitution({t1.name: t2})

    if isinstance(t2, TypeVar):
        if occurs_check(t2.name, t1):
            raise UnificationError(f"Occurs check failed: {t2.name} occurs in {t1}")
        return TypeSubstitution({t2.name: t1})

    # Type constructor cases
    if isinstance(t1, TypeCon) and isinstance(t2, TypeCon):
        if t1.name == t2.name:
            return TypeSubstitution()
        else:
            raise UnificationError(
                f"Cannot unify type constructors {t1.name} and {t2.name}",
            )

    # Function type cases
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        s1 = unify_one(t1.param, t2.param)
        s2 = unify_one(s1.apply(t1.result), s1.apply(t2.result))
        return s2.compose(s1)

    # Type application cases
    if isinstance(t1, TypeApp) and isinstance(t2, TypeApp):
        s1 = unify_one(t1.constructor, t2.constructor)
        s2 = unify_one(s1.apply(t1.argument), s1.apply(t2.argument))
        return s2.compose(s1)

    # Data type cases
    if isinstance(t1, DataType) and isinstance(t2, DataType):
        if t1.name != t2.name or len(t1.type_args) != len(t2.type_args):
            raise UnificationError(f"Cannot unify data types {t1} and {t2}")

        subst = TypeSubstitution()
        for arg1, arg2 in zip(t1.type_args, t2.type_args):
            s = unify_one(subst.apply(arg1), subst.apply(arg2))
            subst = s.compose(subst)

        return subst

    # No other cases match
    raise UnificationError(f"Cannot unify {t1} and {t2}")


def unify(constraints: List[Tuple[Type, Type]]) -> TypeSubstitution:
    """Unify a list of type constraints"""
    subst = TypeSubstitution()

    for t1, t2 in constraints:
        # Apply current substitution to both types
        t1_subst = subst.apply(t1)
        t2_subst = subst.apply(t2)

        # Unify the substituted types
        new_subst = unify_one(t1_subst, t2_subst)

        # Compose with existing substitution
        subst = new_subst.compose(subst)

    return subst
