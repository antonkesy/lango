from lango.shared.typechecker.lango_types import (
    DataType,
    FunctionType,
    TupleType,
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
    match typ:
        case TypeVar(name=name):
            return var == name
        case TypeCon():
            return False
        case FunctionType(param=param, result=result):
            return occurs_check(var, param) or occurs_check(var, result)
        case TypeApp(constructor=constructor, argument=argument):
            return occurs_check(var, constructor) or occurs_check(var, argument)
        case DataType(type_args=type_args):
            return any(occurs_check(var, arg) for arg in type_args)
        case TupleType(element_types=element_types):
            return any(occurs_check(var, elem) for elem in element_types)
        case _:
            raise UnificationError(f"Unknown type in occurs check: {type(typ)}")


def unify_one(t1: Type, t2: Type) -> TypeSubstitution:
    """Unify two types and return the most general unifier"""

    # Same type
    if t1 == t2:
        return TypeSubstitution()

    # Type variable cases
    match t1:
        case TypeVar(name=name):
            if occurs_check(name, t2):
                raise UnificationError(f"Occurs check failed: {name} occurs in {t2}")
            return TypeSubstitution({name: t2})
        case _:
            pass

    match t2:
        case TypeVar(name=name):
            if occurs_check(name, t1):
                raise UnificationError(f"Occurs check failed: {name} occurs in {t1}")
            return TypeSubstitution({name: t1})
        case _:
            pass

    # Type constructor and other cases
    match (t1, t2):
        case (TypeCon(name=name1), TypeCon(name=name2)):
            if name1 == name2:
                return TypeSubstitution()
            else:
                raise UnificationError(
                    f"Cannot unify type constructors {name1} and {name2}",
                )
        case (
            FunctionType(param=param1, result=result1),
            FunctionType(param=param2, result=result2),
        ):
            s1 = unify_one(param1, param2)
            s2 = unify_one(s1.apply(result1), s1.apply(result2))
            return s2.compose(s1)
        case (
            TypeApp(constructor=constructor1, argument=argument1),
            TypeApp(constructor=constructor2, argument=argument2),
        ):
            s1 = unify_one(constructor1, constructor2)
            s2 = unify_one(s1.apply(argument1), s1.apply(argument2))
            return s2.compose(s1)
        case (
            DataType(name=name1, type_args=type_args1),
            DataType(name=name2, type_args=type_args2),
        ):
            if name1 != name2 or len(type_args1) != len(type_args2):
                raise UnificationError(f"Cannot unify data types {t1} and {t2}")

            subst = TypeSubstitution()
            for arg1, arg2 in zip(type_args1, type_args2):
                s = unify_one(subst.apply(arg1), subst.apply(arg2))
                subst = s.compose(subst)
            return subst
        case (
            TupleType(element_types=elem_types1),
            TupleType(element_types=elem_types2),
        ):
            if len(elem_types1) != len(elem_types2):
                raise UnificationError(
                    f"Cannot unify tuples of different lengths: {t1} and {t2}",
                )

            subst = TypeSubstitution()
            for elem1, elem2 in zip(elem_types1, elem_types2):
                s = unify_one(subst.apply(elem1), subst.apply(elem2))
                subst = s.compose(subst)
            return subst
        case _:
            raise UnificationError(f"Cannot unify {t1} and {t2}")
