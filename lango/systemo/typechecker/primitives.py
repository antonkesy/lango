"""The initial typothesis Gamma_0: primitive functions and constants.

The primitives are implemented in :mod:`lango.systemo.runtime`; the prelude
(``lango/systemo/prelude/*.syso``) wraps them in overloaded instances.
"""

from lango.shared.typechecker.lango_types import (
    BOOL_TYPE,
    CHAR_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UNIT_TYPE,
    Type,
    TypeVar,
    function,
)
from lango.systemo.typechecker.types import Scheme, list_of

_A = TypeVar("a")


def mono(*types: Type) -> Scheme:
    """The monomorphic scheme of ``t_1 -> ... -> t_n``."""
    return Scheme([], function(*types))


PRIMITIVES: dict[str, Scheme] = {
    "primError": Scheme([("a", [])], function(STRING_TYPE, _A)),
    "primPutStr": mono(STRING_TYPE, UNIT_TYPE),
    # Int
    "primIntAdd": mono(INT_TYPE, INT_TYPE, INT_TYPE),
    "primIntSub": mono(INT_TYPE, INT_TYPE, INT_TYPE),
    "primIntMul": mono(INT_TYPE, INT_TYPE, INT_TYPE),
    "primIntDiv": mono(INT_TYPE, INT_TYPE, FLOAT_TYPE),
    "primIntPow": mono(INT_TYPE, INT_TYPE, INT_TYPE),
    "primIntNeg": mono(INT_TYPE, INT_TYPE),
    "primIntLt": mono(INT_TYPE, INT_TYPE, BOOL_TYPE),
    "primIntLe": mono(INT_TYPE, INT_TYPE, BOOL_TYPE),
    "primIntGt": mono(INT_TYPE, INT_TYPE, BOOL_TYPE),
    "primIntGe": mono(INT_TYPE, INT_TYPE, BOOL_TYPE),
    "primIntEq": mono(INT_TYPE, INT_TYPE, BOOL_TYPE),
    "primIntShow": mono(INT_TYPE, STRING_TYPE),
    # Float
    "primFloatAdd": mono(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE),
    "primFloatSub": mono(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE),
    "primFloatMul": mono(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE),
    "primFloatDiv": mono(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE),
    "primFloatPow": mono(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE),
    "primFloatNeg": mono(FLOAT_TYPE, FLOAT_TYPE),
    "primFloatLt": mono(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE),
    "primFloatLe": mono(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE),
    "primFloatGt": mono(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE),
    "primFloatGe": mono(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE),
    "primFloatEq": mono(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE),
    "primFloatShow": mono(FLOAT_TYPE, STRING_TYPE),
    # Bool
    "primBoolAnd": mono(BOOL_TYPE, BOOL_TYPE, BOOL_TYPE),
    "primBoolOr": mono(BOOL_TYPE, BOOL_TYPE, BOOL_TYPE),
    "primBoolEq": mono(BOOL_TYPE, BOOL_TYPE, BOOL_TYPE),
    # String / Char
    "primStringConcat": mono(STRING_TYPE, STRING_TYPE, STRING_TYPE),
    "primStringShow": mono(STRING_TYPE, STRING_TYPE),
    "primCharShow": mono(CHAR_TYPE, STRING_TYPE),
    # List
    "primListConcat": Scheme(
        [("a", [])],
        function(list_of(_A), list_of(_A), list_of(_A)),
    ),
}

# Special floating point values are constants of the initial typothesis
# (they are written like constructors of Float).
CONSTANTS: dict[str, Scheme] = {
    "NaN": Scheme([], FLOAT_TYPE),
    "Infinity": Scheme([], FLOAT_TYPE),
}
