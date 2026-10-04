"""Robinson unification for MiniO (System O has constrained unification in
:mod:`lango.systemo.typechecker.infer`)."""

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


def _bind(name: str, t: Type) -> TypeSubstitution:
    if name in t.free_vars():
        raise UnificationError(f"Occurs check failed: {name} occurs in {t}")
    return TypeSubstitution({name: t})


def _unify_all(pairs: list[tuple[Type, Type]]) -> TypeSubstitution:
    subst = TypeSubstitution()
    for t1, t2 in pairs:
        subst = unify_one(subst.apply(t1), subst.apply(t2)).compose(subst)
    return subst


def unify_one(t1: Type, t2: Type) -> TypeSubstitution:
    match (t1, t2):
        case _ if t1 == t2:
            return TypeSubstitution()
        case (TypeVar(name=name), _):
            return _bind(name, t2)
        case (_, TypeVar(name=name)):
            return _bind(name, t1)
        case (TypeCon(name=name1), TypeCon(name=name2)):
            raise UnificationError(
                f"Cannot unify type constructors {name1} and {name2}"
            )
        case (FunctionType(), FunctionType()):
            return _unify_all([(t1.param, t2.param), (t1.result, t2.result)])
        case (TypeApp(), TypeApp()):
            return _unify_all(
                [(t1.constructor, t2.constructor), (t1.argument, t2.argument)],
            )
        case (
            DataType(name=name1, type_args=args1),
            DataType(name=name2, type_args=args2),
        ):
            if name1 != name2 or len(args1) != len(args2):
                raise UnificationError(f"Cannot unify data types {t1} and {t2}")
            return _unify_all(list(zip(args1, args2)))
        case (TupleType(element_types=elems1), TupleType(element_types=elems2)):
            if len(elems1) != len(elems2):
                raise UnificationError(
                    f"Cannot unify tuples of different lengths: {t1} and {t2}",
                )
            return _unify_all(list(zip(elems1, elems2)))
        case _:
            raise UnificationError(f"Cannot unify {t1} and {t2}")
