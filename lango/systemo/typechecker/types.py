"""Types, constrained type schemes and helpers for System O.

The representation of monotypes is shared with MiniO
(:mod:`lango.shared.typechecker.lango_types`).  A System O type is

    tau ::= alpha | tau -> tau' | D tau_1 ... tau_n

where ``D`` is a datatype constructor: the primitive types ``Int``,
``Float``, ``Bool``, ``String``, ``Char`` and ``()`` are nullary datatype
constructors (``TypeCon``), lists are ``List tau`` (``DataType``), tuples
are ``Tuple<n>`` (``TupleType``) and user datatypes are ``DataType``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lango.shared.typechecker.lango_types import (
    BOOL_TYPE,
    CHAR_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UNIT_TYPE,
    DataType,
    FunctionType,
    TupleType,
    Type,
    TypeCon,
    TypeVar,
)

LIST = "List"
FUNCTION = "->"


def list_of(element: Type) -> Type:
    return DataType(LIST, [element])


def function(*types: Type) -> Type:
    """``function(a, b, c)`` is ``a -> b -> c``."""
    result = types[-1]
    for param in reversed(types[:-1]):
        result = FunctionType(param, result)
    return result


def tuple_tycon(arity: int) -> str:
    return f"Tuple{arity}"


def tycon_name(t: Type) -> str:
    """The outermost type constructor ``T`` of a non-variable type."""
    match t:
        case TypeCon(name=name):
            return name
        case DataType(name=name):
            return name
        case TupleType(element_types=elems):
            return tuple_tycon(len(elems))
        case FunctionType():
            return FUNCTION
        case _:
            raise ValueError(f"Type {t} has no outermost type constructor")


def tycon_args(t: Type) -> List[Type]:
    match t:
        case TypeCon():
            return []
        case DataType(type_args=args):
            return list(args)
        case TupleType(element_types=elems):
            return list(elems)
        case FunctionType(param=param, result=result):
            return [param, result]
        case _:
            raise ValueError(f"Type {t} has no outermost type constructor")


def unfold_function(t: Type) -> Tuple[List[Type], Type]:
    params: List[Type] = []
    while isinstance(t, FunctionType):
        params.append(t.param)
        t = t.result
    return params, t


def free_vars(t: Type) -> List[str]:
    """Free type variables in order of first occurrence."""
    result: List[str] = []

    def go(t: Type) -> None:
        match t:
            case TypeVar(name=name):
                if name not in result:
                    result.append(name)
            case FunctionType(param=param, result=res):
                go(param)
                go(res)
            case DataType(type_args=args):
                for arg in args:
                    go(arg)
            case TupleType(element_types=elems):
                for elem in elems:
                    go(elem)
            case _:
                pass

    go(t)
    return result


def substitute(t: Type, mapping: Dict[str, Type]) -> Type:
    return t.substitute(mapping)


# A constraint set on a type variable: ``o_1 : alpha -> tau_1, ..., o_n : alpha -> tau_n``
# with pairwise distinct ``o_i``, kept sorted lexicographically by ``o_i``
# (the paper fixes this order to make the dictionary passing transform coherent).
ConstraintSet = List[Tuple[str, Type]]


@dataclass
class Scheme:
    """A type scheme ``forall alpha_1 . pi_1 => ... forall alpha_n . pi_n => tau``.

    ``quantified`` lists the bound variables in order together with their
    constraint sets; the Hindley/Milner scheme ``forall alpha . tau`` is the
    special case of empty constraint sets.
    """

    quantified: List[Tuple[str, ConstraintSet]] = field(default_factory=list)
    type: Type = UNIT_TYPE

    @property
    def dictionary_params(self) -> List[Tuple[str, str]]:
        """The ``(o, alpha)`` pairs for which a use of this scheme passes a dictionary."""
        return [
            (o, var) for var, constraints in self.quantified for o, _ in constraints
        ]

    def free_vars(self) -> List[str]:
        bound = {var for var, _ in self.quantified}
        names = free_vars(self.type)
        for _, constraints in self.quantified:
            for _, tau in constraints:
                names.extend(free_vars(tau))
        return [name for name in dict.fromkeys(names) if name not in bound]

    def __str__(self) -> str:
        return scheme_to_str(self)


PRIMITIVE_TYPES: Dict[str, Type] = {
    "Int": INT_TYPE,
    "Float": FLOAT_TYPE,
    "Bool": BOOL_TYPE,
    "String": STRING_TYPE,
    "Char": CHAR_TYPE,
    "()": UNIT_TYPE,
}


def _var_names() -> List[str]:
    names = [chr(ord("a") + i) for i in range(26)]
    return names + [f"{n}{i}" for i in range(1, 10) for n in names]


def type_to_str(t: Type, names: Optional[Dict[str, str]] = None) -> str:
    names = names if names is not None else {}

    def name_of(var: str) -> str:
        if var not in names:
            names[var] = _var_names()[len(names)]
        return names[var]

    def go(t: Type, atom: bool) -> str:
        match t:
            case TypeVar(name=name):
                return name_of(name)
            case TypeCon(name=name):
                return name
            case DataType(name="List", type_args=[elem]):
                return f"[{go(elem, False)}]"
            case DataType(name=name, type_args=[]):
                return name
            case DataType(name=name, type_args=args):
                text = " ".join([name] + [go(arg, True) for arg in args])
                return f"({text})" if atom else text
            case TupleType(element_types=elems):
                return "(" + ", ".join(go(e, False) for e in elems) + ")"
            case FunctionType(param=param, result=result):
                left = go(param, False)
                if isinstance(param, FunctionType):
                    left = f"({left})"
                text = f"{left} -> {go(result, False)}"
                return f"({text})" if atom else text
            case _:
                return str(t)

    return go(t, False)


def scheme_to_str(scheme: Scheme, names: Optional[Dict[str, str]] = None) -> str:
    names = names if names is not None else {}
    body = type_to_str(scheme.type, names)
    constraints = [
        f"{display_name(o)} :: {type_to_str(FunctionType(TypeVar(var), tau), names)}"
        for var, constraint_set in scheme.quantified
        for o, tau in constraint_set
    ]
    if constraints:
        return "(" + ", ".join(constraints) + ") => " + body
    return body


def display_name(name: str) -> str:
    """Operators are written in parentheses: ``(==)``."""
    return name if name[0].isalpha() else f"({name})"
