"""The initial typothesis Gamma_0: primitive functions and constants.

The primitives are implemented in :mod:`lango.systemo.runtime`; the prelude
(``lango/systemo/prelude/*.syso``) wraps them in overloaded instances.
"""

from typing import Dict

from lango.shared.typechecker.lango_types import (
    BOOL_TYPE,
    CHAR_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UNIT_TYPE,
    TypeVar,
    function,
)
from lango.systemo.typechecker.types import Scheme, list_of

_A = TypeVar("a")


PRIMITIVES: Dict[str, Scheme] = {
    "primError": Scheme([("a", [])], function(STRING_TYPE, _A)),
    "primPutStr": Scheme([], function(STRING_TYPE, UNIT_TYPE)),
    # Int
    "primIntAdd": Scheme([], function(INT_TYPE, INT_TYPE, INT_TYPE)),
    "primIntSub": Scheme([], function(INT_TYPE, INT_TYPE, INT_TYPE)),
    "primIntMul": Scheme([], function(INT_TYPE, INT_TYPE, INT_TYPE)),
    "primIntDiv": Scheme([], function(INT_TYPE, INT_TYPE, FLOAT_TYPE)),
    "primIntPow": Scheme([], function(INT_TYPE, INT_TYPE, INT_TYPE)),
    "primIntNeg": Scheme([], function(INT_TYPE, INT_TYPE)),
    "primIntLt": Scheme([], function(INT_TYPE, INT_TYPE, BOOL_TYPE)),
    "primIntLe": Scheme([], function(INT_TYPE, INT_TYPE, BOOL_TYPE)),
    "primIntGt": Scheme([], function(INT_TYPE, INT_TYPE, BOOL_TYPE)),
    "primIntGe": Scheme([], function(INT_TYPE, INT_TYPE, BOOL_TYPE)),
    "primIntEq": Scheme([], function(INT_TYPE, INT_TYPE, BOOL_TYPE)),
    "primIntShow": Scheme([], function(INT_TYPE, STRING_TYPE)),
    # Float
    "primFloatAdd": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE)),
    "primFloatSub": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE)),
    "primFloatMul": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE)),
    "primFloatDiv": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE)),
    "primFloatPow": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE)),
    "primFloatNeg": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE)),
    "primFloatLt": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE)),
    "primFloatLe": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE)),
    "primFloatGt": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE)),
    "primFloatGe": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE)),
    "primFloatEq": Scheme([], function(FLOAT_TYPE, FLOAT_TYPE, BOOL_TYPE)),
    "primFloatShow": Scheme([], function(FLOAT_TYPE, STRING_TYPE)),
    # Bool
    "primBoolAnd": Scheme([], function(BOOL_TYPE, BOOL_TYPE, BOOL_TYPE)),
    "primBoolOr": Scheme([], function(BOOL_TYPE, BOOL_TYPE, BOOL_TYPE)),
    "primBoolEq": Scheme([], function(BOOL_TYPE, BOOL_TYPE, BOOL_TYPE)),
    # String / Char
    "primStringConcat": Scheme([], function(STRING_TYPE, STRING_TYPE, STRING_TYPE)),
    "primStringShow": Scheme([], function(STRING_TYPE, STRING_TYPE)),
    "primCharShow": Scheme([], function(CHAR_TYPE, STRING_TYPE)),
    # List
    "primListConcat": Scheme(
        [("a", [])],
        function(list_of(_A), list_of(_A), list_of(_A)),
    ),
}

# Special floating point values are constants of the initial typothesis
# (they are written like constructors of Float).
CONSTANTS: Dict[str, Scheme] = {
    "NaN": Scheme([], FLOAT_TYPE),
    "Infinity": Scheme([], FLOAT_TYPE),
}
