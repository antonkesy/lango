"""Type reconstruction for System O.

This is the algorithm of Figures 6 and 7 of "A Second Look at Overloading"
(Odersky, Wadler, Wehr 1995):

* ``unify`` is constrained unification: binding a type variable ``alpha``
  to a type requires the constraints on ``alpha`` to be satisfied, which
  ``mkinst`` establishes either by moving the constraint to another type
  variable or by finding the (unique) instance ``o : sigma_T`` for the
  outermost type constructor ``T``.
* ``newinst`` instantiates a type scheme with fresh type variables and
  records the instantiated constraints; ``gen`` generalises a type,
  moving the constraints of the quantified variables into the scheme.
* an instance declaration ``inst o :: sigma_T { e }`` is checked by
  generalising the inferred type of ``e``, skolemising the declared
  scheme and unifying the two (``tp`` for ``inst``).
* an overloaded identifier ``o`` has the type ``forall a b . (o : a -> b) => a -> b``.

Besides the types, inference records the *evidence* for every overloading
constraint it creates (which instance, or which dictionary parameter of the
enclosing generalised binding, satisfies it).  The dictionary passing
transform of Section 4 -- and the monomorphising compiler -- are functions
of this evidence.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from lango.shared.ast.nodes import (
    ArrowType,
    ASTNode,
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    ConstrainedType,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DataDeclaration,
    DoBlock,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    GroupedType,
    IfElse,
    InstanceDeclaration,
    IntLiteral,
    LetStatement,
    ListLiteral,
    ListPattern,
    ListType,
    LiteralPattern,
    NegativeFloat,
    NegativeFloatPattern,
    NegativeInt,
    NegativeIntPattern,
    Pattern,
    Program,
    Statement,
    StringLiteral,
    TupleLiteral,
    TuplePattern,
)
from lango.shared.ast.nodes import TupleType as ASTTupleType
from lango.shared.ast.nodes import (
    TypeApplication,
    TypeConstructor,
    TypeExpression,
    TypeVariable,
    Variable,
    VariablePattern,
)
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
from lango.systemo.typechecker.primitives import CONSTANTS, PRIMITIVES
from lango.systemo.typechecker.types import (
    LIST,
    PRIMITIVE_TYPES,
    ConstraintSet,
    Scheme,
    free_vars,
    function,
    list_of,
    scheme_to_str,
    substitute,
    tycon_args,
    tycon_name,
    type_to_str,
    unfold_function,
)


class TypeInferenceError(Exception):
    pass


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


@dataclass
class InstanceEvidence:
    """The constraint is satisfied by the instance ``o : sigma_T``,
    whose own constraints are satisfied by ``arguments`` (in scheme order)."""

    name: str
    tycon: str
    arguments: List["Constraint"]


@dataclass
class ParamEvidence:
    """The constraint ``o : alpha -> tau`` was generalised: it is satisfied
    by the dictionary parameter for ``(o, alpha)`` of the enclosing binding."""

    name: str
    var: str


@dataclass
class AmbiguousEvidence:
    """The constrained type variable is neither generalised nor
    instantiated (e.g. ``[] == []``): by coherence any dictionary will do."""

    name: str


Evidence = Union[InstanceEvidence, ParamEvidence, AmbiguousEvidence]


@dataclass(eq=False)
class Constraint:
    """A constraint ``o : var -> result`` that demands evidence."""

    name: str
    var: str
    result: Type
    evidence: Optional[Evidence] = None
    alias: Optional["Constraint"] = None

    def canonical(self) -> "Constraint":
        c = self
        while c.alias is not None:
            c = c.alias
        return c


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


@dataclass
class LetBinding:
    scheme: Scheme


@dataclass
class MonoBinding:
    """A lambda/pattern bound variable."""

    type: Type


@dataclass
class RecBinding:
    """The function currently being defined (monomorphic recursion)."""

    type: Type


@dataclass
class PrimBinding:
    scheme: Scheme


@dataclass
class ConBinding:
    scheme: Scheme
    field_names: List[str]
    tycon: str


Binding = Union[LetBinding, MonoBinding, RecBinding, PrimBinding, ConBinding]


class Env:
    def __init__(self, bindings: Optional[Dict[str, Binding]] = None) -> None:
        self.bindings: Dict[str, Binding] = bindings or {}

    def lookup(self, name: str) -> Optional[Binding]:
        return self.bindings.get(name)

    def extend(self, name: str, binding: Binding) -> "Env":
        new = dict(self.bindings)
        new[name] = binding
        return Env(new)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class VarUse:
    """How an occurrence of a variable was typed."""

    kind: str  # "let" | "mono" | "self" | "prim" | "constructor" | "overloaded"
    name: str
    evidence: List[Constraint] = field(default_factory=list)  # for "let"
    constraint: Optional[Constraint] = None  # for "overloaded"


@dataclass
class ConstructorInfo:
    name: str
    tycon: str
    field_types: List[Type]
    field_names: List[str]
    scheme: Scheme


@dataclass
class DataInfo:
    name: str
    params: List[str]
    constructors: List[ConstructorInfo]


@dataclass
class FunctionDecl:
    """A (possibly recursive) top-level binding ``let u = e``."""

    name: str
    clauses: List[FunctionDefinition]
    scheme: Scheme
    arity: int


@dataclass
class InstanceInfo:
    """An instance ``inst o :: sigma_T = e``."""

    name: str
    tycon: str
    scheme: Scheme  # the declared sigma_T
    clauses: List[FunctionDefinition]
    arity: int
    inferred: Scheme = field(default_factory=Scheme)
    # evidence for the constraints of the *inferred* scheme of the body,
    # expressed in terms of the declared scheme (see ``check_instance``)
    body_evidence: Dict[Tuple[str, str], Constraint] = field(default_factory=dict)
    # skolem type constructor -> quantified variable of the declared scheme
    skolem_vars: Dict[str, str] = field(default_factory=dict)
    skolem: bool = False


Decl = Union[FunctionDecl, InstanceInfo]


@dataclass
class TypedProgram:
    program: Program
    data_types: Dict[str, DataInfo]
    constructors: Dict[str, ConstructorInfo]
    decls: List[Decl]
    overloaded: Set[str]
    var_uses: Dict[int, VarUse]
    let_schemes: Dict[int, Scheme]

    def var_use(self, node: Union[Variable, Constructor]) -> VarUse:
        return self.var_uses[id(node)]

    def let_scheme(self, node: LetStatement) -> Scheme:
        return self.let_schemes[id(node)]


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


class TypeInferrer:
    def __init__(self) -> None:
        self.counter = 0
        self.subst: Dict[str, Type] = {}
        # constraints on unbound type variables: var -> o -> constraint
        self.constraints: Dict[str, Dict[str, Constraint]] = defaultdict(dict)
        # instances: o -> T -> instance
        self.instances: Dict[str, Dict[str, InstanceInfo]] = defaultdict(dict)
        self.overloaded: Set[str] = set()
        # the built-in list type constructor has no user-visible constructors
        self.data_types: Dict[str, DataInfo] = {LIST: DataInfo(LIST, ["a"], [])}
        self.constructors: Dict[str, ConstructorInfo] = {}
        self.var_uses: Dict[int, VarUse] = {}
        self.let_schemes: Dict[int, Scheme] = {}
        self.typed_nodes: List[ASTNode] = []

    # --- fresh variables and substitution ------------------------------------

    def fresh(self) -> TypeVar:
        self.counter += 1
        return TypeVar(f"t{self.counter}")

    def resolve(self, t: Type) -> Type:
        match t:
            case TypeVar(name=name):
                bound = self.subst.get(name)
                if bound is None:
                    return t
                resolved = self.resolve(bound)
                self.subst[name] = resolved
                return resolved
            case FunctionType(param=param, result=result):
                return FunctionType(self.resolve(param), self.resolve(result))
            case DataType(name=name, type_args=args):
                return DataType(name, [self.resolve(arg) for arg in args])
            case TupleType(element_types=elems):
                return TupleType([self.resolve(elem) for elem in elems])
            case _:
                return t

    def resolve_scheme(self, scheme: Scheme) -> Scheme:
        return Scheme(
            [
                (var, [(o, self.resolve(tau)) for o, tau in constraints])
                for var, constraints in scheme.quantified
            ],
            self.resolve(scheme.type),
        )

    # --- constrained unification (Figure 6) ----------------------------------

    def unify(self, t1: Type, t2: Type) -> None:
        t1 = self.resolve(t1)
        t2 = self.resolve(t2)
        match (t1, t2):
            case (TypeVar(name=a), TypeVar(name=b)):
                if a != b:
                    self.bind(a, t2)
            case (TypeVar(name=a), _):
                self.bind(a, t2)
            case (_, TypeVar(name=b)):
                self.bind(b, t1)
            case _:
                if tycon_name(t1) != tycon_name(t2):
                    raise TypeInferenceError(
                        f"Cannot unify {type_to_str(t1)} with {type_to_str(t2)}",
                    )
                args1, args2 = tycon_args(t1), tycon_args(t2)
                if len(args1) != len(args2):
                    raise TypeInferenceError(
                        f"Cannot unify {type_to_str(t1)} with {type_to_str(t2)}",
                    )
                for arg1, arg2 in zip(args1, args2):
                    self.unify(arg1, arg2)

    def bind(self, var: str, t: Type) -> None:
        if var in free_vars(t):
            raise TypeInferenceError(
                f"Cannot construct the infinite type {var} = {type_to_str(t)}",
            )
        self.subst[var] = t
        # [tau/alpha] S, then re-establish the constraints of alpha (foldr mkinst)
        pending = self.constraints.pop(var, {})
        for o in sorted(pending):
            self.mkinst(pending[o], t)

    def mkinst(self, constraint: Constraint, t: Type) -> None:
        t = self.resolve(t)
        o = constraint.name
        match t:
            case TypeVar(name=beta):
                existing = self.constraints[beta].get(o)
                if existing is not None:
                    # the argument type determines the instance type: o : beta -> tau
                    # and o : beta -> tau' must agree
                    constraint.alias = existing
                    self.unify(constraint.result, existing.result)
                else:
                    constraint.var = beta
                    self.constraints[beta][o] = constraint
            case _:
                tycon = tycon_name(t)
                instance = self.instances[o].get(tycon)
                if instance is None:
                    raise TypeInferenceError(
                        f"No instance of '{o}' for type constructor '{tycon}' "
                        f"(required at type {type_to_str(FunctionType(t, self.resolve(constraint.result)))})",
                    )
                instance_type, arguments = self.newinst(instance.scheme)
                constraint.evidence = InstanceEvidence(o, tycon, arguments)
                self.unify(FunctionType(t, constraint.result), instance_type)

    # --- instantiation and generalisation (Figure 7) -------------------------

    def newinst(self, scheme: Scheme) -> Tuple[Type, List[Constraint]]:
        mapping: Dict[str, Type] = {var: self.fresh() for var, _ in scheme.quantified}
        created: List[Constraint] = []
        for var, constraints in scheme.quantified:
            fresh_var = mapping[var]
            assert isinstance(fresh_var, TypeVar)
            for o, tau in constraints:
                constraint = Constraint(o, fresh_var.name, substitute(tau, mapping))
                self.constraints[fresh_var.name][o] = constraint
                created.append(constraint)
        return substitute(scheme.type, mapping), created

    def overloaded_use(self, o: str) -> Tuple[Type, Constraint]:
        """``tp(o) = newinst(forall a b . (o : a -> b) => a -> b)``."""
        a, b = self.fresh(), self.fresh()
        constraint = Constraint(o, a.name, b)
        self.constraints[a.name][o] = constraint
        return FunctionType(a, b), constraint

    def env_free_vars(self, env: Env) -> Set[str]:
        """Type variables of ``S Gamma``, including those reachable through
        the constraints on them (``Gamma`` contains the constraint bindings)."""
        result: Set[str] = set()
        for binding in env.bindings.values():
            match binding:
                case MonoBinding(type=t) | RecBinding(type=t):
                    result.update(free_vars(self.resolve(t)))
                case LetBinding(scheme=scheme):
                    result.update(self.resolve_scheme(scheme).free_vars())
                case _:
                    pass
        worklist = list(result)
        while worklist:
            var = worklist.pop()
            for constraint in self.constraints.get(var, {}).values():
                for other in free_vars(self.resolve(constraint.result)):
                    if other not in result:
                        result.add(other)
                        worklist.append(other)
        return result

    def gen(self, t: Type, env: Env) -> Scheme:
        t = self.resolve(t)
        env_vars = self.env_free_vars(env)
        quantified: List[Tuple[str, ConstraintSet]] = []
        seen: Set[str] = set()
        worklist = free_vars(t)
        while worklist:
            var = worklist.pop(0)
            if var in seen or var in env_vars:
                continue
            seen.add(var)
            pending = self.constraints.pop(var, {})
            constraint_set: ConstraintSet = []
            for o in sorted(pending):
                constraint = pending[o]
                constraint.evidence = ParamEvidence(o, var)
                tau = self.resolve(constraint.result)
                constraint_set.append((o, tau))
                worklist.extend(free_vars(tau))
            quantified.append((var, constraint_set))
        return Scheme(quantified, t)

    def skolemize(self, scheme: Scheme) -> Tuple[Type, Dict[str, str]]:
        """Replace the quantified variables by fresh nullary type constructors
        and turn their constraints into instances for those constructors."""
        skolems: Dict[str, str] = {}
        mapping: Dict[str, Type] = {}
        for var, _ in scheme.quantified:
            self.counter += 1
            name = f"$Sk{self.counter}"
            mapping[var] = TypeCon(name)
            skolems[name] = var
        for var, constraints in scheme.quantified:
            skolem = mapping[var]
            for o, tau in constraints:
                self.instances[o][tycon_name(skolem)] = InstanceInfo(
                    name=o,
                    tycon=tycon_name(skolem),
                    scheme=Scheme([], FunctionType(skolem, substitute(tau, mapping))),
                    clauses=[],
                    arity=0,
                    skolem=True,
                )
        return substitute(scheme.type, mapping), skolems

    def remove_skolems(self, skolems: Dict[str, str]) -> None:
        for instances in self.instances.values():
            for name in list(instances):
                if name in skolems:
                    del instances[name]

    def discard_ambiguous_constraints(self) -> None:
        """Constraints left on type variables that are neither generalised nor
        instantiated are ambiguous; by coherence any evidence is valid."""
        for pending in self.constraints.values():
            for constraint in pending.values():
                if constraint.evidence is None:
                    constraint.evidence = AmbiguousEvidence(constraint.name)
        self.constraints.clear()

    # --- type expressions ------------------------------------------------------

    def parse_type(
        self,
        node: Union[TypeExpression, ASTNode],
        scope: Optional[Dict[str, Type]] = None,
    ) -> Type:
        """Translate a type expression; ``scope`` maps the type variables in
        scope (``None`` allows any variable, e.g. in an instance scheme)."""
        match node:
            case TypeConstructor(name=name):
                if name in PRIMITIVE_TYPES:
                    return PRIMITIVE_TYPES[name]
                return self._data_type(name, [])
            case TypeVariable(name=name):
                if scope is None:
                    return TypeVar(name)
                if name not in scope:
                    raise TypeInferenceError(f"Type variable '{name}' is not in scope")
                return scope[name]
            case ArrowType(from_type=from_type, to_type=to_type):
                return FunctionType(
                    self.parse_type(from_type, scope),
                    self.parse_type(to_type, scope),
                )
            case TypeApplication(constructor=constructor, argument=argument):
                head = self.parse_type(constructor, scope)
                arg = self.parse_type(argument, scope)
                match head:
                    case DataType(name=name, type_args=args):
                        return self._data_type(name, args + [arg])
                    case _:
                        raise TypeInferenceError(
                            f"Type {type_to_str(head)} cannot be applied to an argument",
                        )
            case ListType(element_type=element_type):
                return list_of(self.parse_type(element_type, scope))
            case ASTTupleType(element_types=element_types):
                return TupleType([self.parse_type(e, scope) for e in element_types])
            case GroupedType(type_expr=type_expr):
                return self.parse_type(type_expr, scope)
            case _:
                raise TypeInferenceError(f"Cannot parse type expression: {node}")

    def _data_type(self, name: str, args: List[Type]) -> Type:
        info = self.data_types.get(name)
        if info is None:
            raise TypeInferenceError(f"Unknown type '{name}'")
        if len(args) > len(info.params):
            raise TypeInferenceError(
                f"Type '{name}' expects {len(info.params)} arguments, got {len(args)}",
            )
        return DataType(name, args)

    def _check_saturated(self, t: Type) -> None:
        match t:
            case DataType(name=name, type_args=args):
                if len(args) != len(self.data_types[name].params):
                    raise TypeInferenceError(
                        f"Type '{name}' expects {len(self.data_types[name].params)} "
                        f"arguments, got {len(args)}",
                    )
                for arg in args:
                    self._check_saturated(arg)
            case FunctionType(param=param, result=result):
                self._check_saturated(param)
                self._check_saturated(result)
            case TupleType(element_types=elems):
                for elem in elems:
                    self._check_saturated(elem)
            case _:
                pass

    def parse_declared_scheme(
        self,
        node: ConstrainedType,
        o: str,
    ) -> Tuple[Scheme, str]:
        """Translate the type scheme of ``inst o :: sigma_T`` and check that it
        has the form required by the paper:

            sigma_T = T a_1 ... a_n -> tau        (tv(tau) subset of {a_i})
                    | forall a . pi_a => sigma_T   (tv(pi_a) subset of tv(sigma_T))
        """
        body = self.parse_type(node.type_expr)
        self._check_saturated(body)
        declared = scheme_to_str(Scheme([], body))
        match body:
            case FunctionType(param=param, result=result):
                pass
            case _:
                raise TypeInferenceError(
                    f"Instance type '{declared}' of '{o}' must be a function type",
                )
        if isinstance(param, TypeVar):
            raise TypeInferenceError(
                f"Instance type '{declared}' of '{o}' must have a type constructor "
                f"as its argument type",
            )
        tycon = tycon_name(param)
        arg_vars = [
            arg.name if isinstance(arg, TypeVar) else None for arg in tycon_args(param)
        ]
        if None in arg_vars or len(set(arg_vars)) != len(arg_vars):
            raise TypeInferenceError(
                f"Instance type '{declared}' of '{o}' must be parametric in the "
                f"arguments of '{tycon}' (distinct type variables)",
            )
        scheme_vars: List[str] = [var for var in arg_vars if var is not None]
        if not set(free_vars(result)) <= set(scheme_vars):
            raise TypeInferenceError(
                f"Instance type '{declared}' of '{o}': the argument type must "
                f"determine the result type uniquely",
            )
        constraint_sets: Dict[str, ConstraintSet] = {var: [] for var in scheme_vars}
        for constraint in node.constraints:
            ctype = self.parse_type(constraint.type_expr)
            self._check_saturated(ctype)
            match ctype:
                case FunctionType(param=TypeVar(name=var), result=tau) if (
                    var in constraint_sets
                ):
                    pass
                case _:
                    raise TypeInferenceError(
                        f"Constraint '{constraint.name} :: {type_to_str(ctype)}' of "
                        f"'{o}' must constrain a type variable of '{declared}'",
                    )
            if not set(free_vars(tau)) <= set(scheme_vars):
                raise TypeInferenceError(
                    f"Constraint '{constraint.name} :: {type_to_str(ctype)}' of '{o}' "
                    f"mentions type variables not bound by the instance type",
                )
            if any(name == constraint.name for name, _ in constraint_sets[var]):
                raise TypeInferenceError(
                    f"Duplicate constraint on '{constraint.name}' for type variable "
                    f"'{var}' in the instance type of '{o}'",
                )
            constraint_sets[var].append((constraint.name, tau))
        # rename the user's type variables to fresh internal ones
        mapping: Dict[str, Type] = {var: self.fresh() for var in scheme_vars}
        quantified: List[Tuple[str, ConstraintSet]] = []
        for var in scheme_vars:
            fresh_var = mapping[var]
            assert isinstance(fresh_var, TypeVar)
            quantified.append(
                (
                    fresh_var.name,
                    sorted(
                        (name, substitute(tau, mapping))
                        for name, tau in constraint_sets[var]
                    ),
                ),
            )
        return Scheme(quantified, substitute(body, mapping)), tycon

    # --- data declarations -----------------------------------------------------

    def declare_data(self, decls: Sequence[DataDeclaration]) -> None:
        for decl in decls:
            if decl.type_name in self.data_types or decl.type_name in PRIMITIVE_TYPES:
                raise TypeInferenceError(f"Type '{decl.type_name}' is defined twice")
            params = [param.name for param in decl.type_params]
            if len(set(params)) != len(params):
                raise TypeInferenceError(
                    f"Duplicate type parameter in data declaration '{decl.type_name}'",
                )
            self.data_types[decl.type_name] = DataInfo(decl.type_name, params, [])
        for decl in decls:
            info = self.data_types[decl.type_name]
            scope: Dict[str, Type] = {param: TypeVar(param) for param in info.params}
            result = DataType(decl.type_name, [TypeVar(param) for param in info.params])
            for constructor in decl.constructors:
                if constructor.name in self.constructors:
                    raise TypeInferenceError(
                        f"Constructor '{constructor.name}' is defined twice",
                    )
                if constructor.record_constructor is not None:
                    fields = constructor.record_constructor.fields
                    field_names = [f.name for f in fields]
                    field_types = [self.parse_type(f.field_type, scope) for f in fields]
                else:
                    field_names = []
                    field_types = [
                        self.parse_type(atom, scope)
                        for atom in constructor.type_atoms or []
                    ]
                for field_type in field_types:
                    self._check_saturated(field_type)
                scheme = Scheme(
                    [(param, []) for param in info.params],
                    function(*field_types, result),
                )
                con = ConstructorInfo(
                    constructor.name,
                    decl.type_name,
                    field_types,
                    field_names,
                    scheme,
                )
                info.constructors.append(con)
                self.constructors[constructor.name] = con

    def initial_env(self) -> Env:
        env = Env()
        for name, scheme in PRIMITIVES.items():
            env = env.extend(name, PrimBinding(scheme))
        for name, scheme in CONSTANTS.items():
            env = env.extend(name, PrimBinding(scheme))
        for name, con in self.constructors.items():
            env = env.extend(name, ConBinding(con.scheme, con.field_names, con.tycon))
        return env

    # --- expressions -------------------------------------------------------------

    def annotate(self, node: ASTNode, t: Type) -> Type:
        node.ty = t  # type: ignore[attr-defined]
        self.typed_nodes.append(node)
        return t

    def infer(self, expr: Expression, env: Env) -> Type:
        return self.annotate(expr, self._infer(expr, env))

    def _infer(self, expr: Expression, env: Env) -> Type:
        match expr:
            case IntLiteral() | NegativeInt():
                return INT_TYPE
            case FloatLiteral() | NegativeFloat():
                return FLOAT_TYPE
            case StringLiteral():
                return STRING_TYPE
            case CharLiteral():
                return CHAR_TYPE
            case BoolLiteral():
                return BOOL_TYPE
            case ListLiteral(elements=elements):
                element = self.fresh()
                for e in elements:
                    self.unify(self.infer(e, env), element)
                return list_of(element)
            case TupleLiteral(elements=elements):
                return TupleType([self.infer(e, env) for e in elements])
            case Variable(name=name):
                return self.infer_variable(expr, name, env)
            case Constructor(name=name):
                return self.infer_variable(expr, name, env)
            case FunctionApplication(function=f, argument=a):
                function_type = self.infer(f, env)
                argument_type = self.infer(a, env)
                result = self.fresh()
                self.unify(function_type, FunctionType(argument_type, result))
                return result
            case IfElse(condition=c, then_expr=t, else_expr=e):
                self.unify(self.infer(c, env), BOOL_TYPE)
                then_type = self.infer(t, env)
                self.unify(then_type, self.infer(e, env))
                return then_type
            case GroupedExpression(expression=inner):
                return self.infer(inner, env)
            case DoBlock(statements=statements):
                return self.infer_block(statements, env)
            case ConstructorExpression(constructor_name=name, fields=fields):
                return self.infer_record(expr, name, fields, env)
            case _:
                raise TypeInferenceError(f"Unhandled expression: {type(expr).__name__}")

    def infer_variable(
        self,
        node: Union[Variable, Constructor],
        name: str,
        env: Env,
    ) -> Type:
        binding = env.lookup(name)
        match binding:
            case None:
                if name in self.overloaded:
                    t, constraint = self.overloaded_use(name)
                    self.var_uses[id(node)] = VarUse(
                        "overloaded",
                        name,
                        constraint=constraint,
                    )
                    return t
                raise TypeInferenceError(f"Unknown variable '{name}'")
            case MonoBinding(type=t):
                self.var_uses[id(node)] = VarUse("mono", name)
                return t
            case RecBinding(type=t):
                self.var_uses[id(node)] = VarUse("self", name)
                return t
            case LetBinding(scheme=scheme):
                t, evidence = self.newinst(scheme)
                self.var_uses[id(node)] = VarUse("let", name, evidence=evidence)
                return t
            case PrimBinding(scheme=scheme):
                t, _ = self.newinst(scheme)
                self.var_uses[id(node)] = VarUse("prim", name)
                return t
            case ConBinding(scheme=scheme):
                t, _ = self.newinst(scheme)
                self.var_uses[id(node)] = VarUse("constructor", name)
                return t
        raise TypeInferenceError(f"Unknown variable '{name}'")

    def infer_record(
        self,
        node: ConstructorExpression,
        name: str,
        fields: Sequence[object],
        env: Env,
    ) -> Type:
        binding = env.lookup(name)
        if not isinstance(binding, ConBinding):
            raise TypeInferenceError(f"Unknown constructor '{name}'")
        if not binding.field_names:
            raise TypeInferenceError(f"Constructor '{name}' has no named fields")
        t, _ = self.newinst(binding.scheme)
        params, result = unfold_function(t)
        given = {f.field_name: f.value for f in fields}  # type: ignore[attr-defined]
        if set(given) != set(binding.field_names):
            raise TypeInferenceError(
                f"Constructor '{name}' expects fields {binding.field_names}, "
                f"got {sorted(given)}",
            )
        for field_name, param in zip(binding.field_names, params):
            self.unify(self.infer(given[field_name], env), param)
        self.var_uses[id(node)] = VarUse("constructor", name)
        return result

    def infer_block(self, statements: Sequence[Statement], env: Env) -> Type:
        result: Type = UNIT_TYPE
        for stmt in statements:
            match stmt:
                case LetStatement(variable=name, value=value):
                    # let u = e in e' (non-recursive)
                    scheme = self.gen(self.infer(value, env), env)
                    self.let_schemes[id(stmt)] = scheme
                    env = env.extend(name, LetBinding(scheme))
                    result = UNIT_TYPE
                case _:
                    result = self.infer(stmt, env)  # type: ignore[arg-type]
        return result

    # --- patterns --------------------------------------------------------------

    def infer_pattern(
        self,
        pattern: Pattern,
        env: Env,
        bound: Set[str],
    ) -> Tuple[Type, Env]:
        t, env = self._infer_pattern(pattern, env, bound)
        self.annotate(pattern, t)
        return t, env

    def _infer_pattern(
        self,
        pattern: Pattern,
        env: Env,
        bound: Set[str],
    ) -> Tuple[Type, Env]:
        match pattern:
            case VariablePattern(name=name):
                if name in bound:
                    raise TypeInferenceError(
                        f"Variable '{name}' is bound twice in a pattern",
                    )
                bound.add(name)
                t = self.fresh()
                return t, env.extend(name, MonoBinding(t))
            case LiteralPattern(value=literal):
                return self.infer(literal, env), env
            case NegativeIntPattern():
                return INT_TYPE, env
            case NegativeFloatPattern():
                return FLOAT_TYPE, env
            case ConstructorPattern(constructor=name, patterns=subpatterns):
                binding = env.lookup(name)
                if not isinstance(binding, ConBinding):
                    raise TypeInferenceError(f"Unknown constructor '{name}' in pattern")
                constructor_type, _ = self.newinst(binding.scheme)
                params, result = unfold_function(constructor_type)
                if len(params) != len(subpatterns):
                    raise TypeInferenceError(
                        f"Constructor '{name}' takes {len(params)} arguments, "
                        f"but the pattern has {len(subpatterns)}",
                    )
                for sub, param in zip(subpatterns, params):
                    sub_type, env = self.infer_pattern(sub, env, bound)
                    self.unify(sub_type, param)
                return result, env
            case ConsPattern(head=head, tail=tail):
                head_type, env = self.infer_pattern(head, env, bound)
                tail_type, env = self.infer_pattern(tail, env, bound)
                self.unify(tail_type, list_of(head_type))
                return tail_type, env
            case ListPattern(patterns=subpatterns):
                element = self.fresh()
                for sub in subpatterns:
                    sub_type, env = self.infer_pattern(sub, env, bound)
                    self.unify(sub_type, element)
                return list_of(element), env
            case TuplePattern(patterns=subpatterns):
                types: List[Type] = []
                for sub in subpatterns:
                    sub_type, env = self.infer_pattern(sub, env, bound)
                    types.append(sub_type)
                return TupleType(types), env
            case _:
                raise TypeInferenceError(f"Unhandled pattern: {type(pattern).__name__}")

    # --- bindings ----------------------------------------------------------------

    def infer_clauses(
        self,
        clauses: Sequence[FunctionDefinition],
        env: Env,
        function_type: Type,
    ) -> None:
        """Every clause ``f p_1 ... p_n = e`` has the type ``function_type``."""
        arity = len(clauses[0].patterns)
        for clause in clauses:
            if len(clause.patterns) != arity:
                raise TypeInferenceError(
                    f"Clauses of '{clause.function_name}' have different numbers of parameters",
                )
            clause_env = env
            bound: Set[str] = set()
            param_types: List[Type] = []
            for pattern in clause.patterns:
                param_type, clause_env = self.infer_pattern(pattern, clause_env, bound)
                param_types.append(param_type)
            body_type = self.infer(clause.body, clause_env)
            clause_type = function(*param_types, body_type)
            self.annotate(clause, clause_type)
            try:
                self.unify(function_type, clause_type)
            except TypeInferenceError as e:
                raise TypeInferenceError(
                    f"In the definition of '{clause.function_name}': {e}",
                ) from e

    def infer_function(
        self,
        name: str,
        clauses: Sequence[FunctionDefinition],
        env: Env,
    ) -> Scheme:
        """``let u = e`` where ``u`` may occur (monomorphically) in ``e``."""
        function_type = self.fresh()
        self.infer_clauses(
            clauses,
            env.extend(name, RecBinding(function_type)),
            function_type,
        )
        return self.gen(function_type, env)

    def check_instance(self, decl: InstanceDeclaration, env: Env) -> InstanceInfo:
        o = decl.instance_name
        assert isinstance(decl.type_signature, ConstrainedType)
        declared, tycon = self.parse_declared_scheme(decl.type_signature, o)
        if tycon in self.instances[o]:
            raise TypeInferenceError(
                f"'{o}' already has an instance for type constructor '{tycon}'",
            )
        # tp(e) in Gamma: instance declarations are not recursive
        function_type = self.fresh()
        self.infer_clauses(decl.clauses, env, function_type)
        inferred = self.gen(function_type, env)
        # the inferred scheme must be at least as general as the declared one
        skolem_type, skolems = self.skolemize(declared)
        instance_type, copies = self.newinst(inferred)
        try:
            self.unify(skolem_type, instance_type)
        except TypeInferenceError as e:
            raise TypeInferenceError(
                f"Instance '{o} :: {scheme_to_str(declared)}' is implemented by a "
                f"function of type '{scheme_to_str(self.resolve_scheme(inferred))}': {e}",
            ) from e
        finally:
            self.remove_skolems(skolems)
        body_evidence: Dict[Tuple[str, str], Constraint] = {}
        index = 0
        for var, constraints in inferred.quantified:
            for name, _ in constraints:
                body_evidence[(name, var)] = copies[index]
                index += 1
        info = InstanceInfo(
            name=o,
            tycon=tycon,
            scheme=declared,
            clauses=list(decl.clauses),
            arity=len(decl.clauses[0].patterns),
            inferred=inferred,
            body_evidence=body_evidence,
            skolem_vars=skolems,
        )
        self.instances[o][tycon] = info
        return info

    # --- programs ----------------------------------------------------------------

    def infer_program(self, program: Program) -> TypedProgram:
        data_decls = [s for s in program.statements if isinstance(s, DataDeclaration)]
        self.declare_data(data_decls)
        self.overloaded = {
            s.instance_name
            for s in program.statements
            if isinstance(s, InstanceDeclaration)
        }
        env = self.initial_env()
        decls: List[Decl] = []
        statements = list(program.statements)
        index = 0
        while index < len(statements):
            stmt = statements[index]
            match stmt:
                case DataDeclaration():
                    index += 1
                case InstanceDeclaration():
                    decls.append(self.check_instance(stmt, env))
                    self.discard_ambiguous_constraints()
                    index += 1
                case FunctionDefinition(function_name=name):
                    clauses = [stmt]
                    while (
                        index + len(clauses) < len(statements)
                        and isinstance(
                            statements[index + len(clauses)],
                            FunctionDefinition,
                        )
                        and statements[index + len(clauses)].function_name == name  # type: ignore[union-attr]
                    ):
                        clauses.append(statements[index + len(clauses)])  # type: ignore[arg-type]
                    index += len(clauses)
                    if name in self.overloaded:
                        raise TypeInferenceError(
                            f"'{name}' is overloaded and cannot also be defined as a function",
                        )
                    if env.lookup(name) is not None:
                        raise TypeInferenceError(f"'{name}' is bound more than once")
                    scheme = self.infer_function(name, clauses, env)
                    self.discard_ambiguous_constraints()
                    env = env.extend(name, LetBinding(scheme))
                    decls.append(
                        FunctionDecl(name, clauses, scheme, len(clauses[0].patterns)),
                    )
                case _:
                    raise TypeInferenceError(
                        f"Unexpected statement: {type(stmt).__name__}",
                    )
        for node in self.typed_nodes:
            node.ty = self.resolve(node.ty)  # type: ignore[attr-defined]
        for decl in decls:
            match decl:
                case FunctionDecl():
                    decl.scheme = self.resolve_scheme(decl.scheme)
                case InstanceInfo():
                    decl.inferred = self.resolve_scheme(decl.inferred)
        return TypedProgram(
            program=program,
            data_types=self.data_types,
            constructors=self.constructors,
            decls=decls,
            overloaded=self.overloaded,
            var_uses=self.var_uses,
            let_schemes=self.let_schemes,
        )


def infer_program(program: Program) -> TypedProgram:
    return TypeInferrer().infer_program(program)


def resolve_evidence(constraint: Constraint) -> Evidence:
    evidence = constraint.canonical().evidence
    if evidence is None:
        return AmbiguousEvidence(constraint.name)
    return evidence
