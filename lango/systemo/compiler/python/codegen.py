"""Compilation of System O to Python.

The generated code is the dictionary passing transform of Section 4 of the
paper: a binding of type ``forall a . (o_1 : a -> t_1, ..., o_n : a -> t_n) => t``
becomes a function of ``n`` dictionaries (the implementations of the
overloaded identifiers, in the fixed lexicographic order of the ``o_i``), an
overloaded identifier at an instance type becomes the instance function
``u_{o, sigma_T}`` and an overloaded identifier at a constrained type variable
becomes the corresponding dictionary parameter.

Which dictionary is passed where is completely determined by the *evidence*
recorded during type inference, so the same generator supports two
strategies:

* ``DICTIONARY_PASSING`` passes dictionaries at run time, exactly as in the
  paper;
* ``MONOMORPHIZATION`` resolves the evidence at compile time: every
  constrained binding is copied once per distinct tuple of dictionaries it is
  used with, with the dictionary parameters replaced by the instance
  functions themselves.  This is possible because System O has no
  polymorphic recursion, so the set of instantiations is finite.
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from lango.shared.ast.nodes import (
    BoolLiteral,
    CharLiteral,
    ConsPattern,
    Constructor,
    ConstructorExpression,
    ConstructorPattern,
    DoBlock,
    Expression,
    FloatLiteral,
    FunctionApplication,
    FunctionDefinition,
    GroupedExpression,
    IfElse,
    IntLiteral,
    LetStatement,
    ListLiteral,
    ListPattern,
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
    Variable,
    VariablePattern,
    is_expression,
)
from lango.systemo import runtime
from lango.systemo.typechecker.infer import (
    AmbiguousEvidence,
    Constraint,
    Decl,
    FunctionDecl,
    InstanceEvidence,
    InstanceInfo,
    ParamEvidence,
    TypedProgram,
    UseKind,
    infer_program,
    resolve_evidence,
)
from lango.systemo.typechecker.types import Scheme


class Strategy(StrEnum):
    DICTIONARY_PASSING = "dictionary_passing"
    MONOMORPHIZATION = "monomorphization"


class CompileError(Exception):
    pass


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

SYMBOL_NAMES = {
    "?": "question",
    "+": "plus",
    "-": "minus",
    "*": "star",
    "/": "slash",
    "^": "caret",
    "=": "eq",
    "<": "lt",
    ">": "gt",
    "&": "amp",
    "|": "bar",
    "!": "bang",
    "@": "at",
}


def mangle(name: str) -> str:
    """A Python identifier for a System O variable or operator."""
    if name[0].isalpha() or name[0] == "_":
        return f"v_{name}"
    return "op_" + "_".join(SYMBOL_NAMES[c] for c in name)


def mangle_tycon(tycon: str) -> str:
    return {"->": "Fun", "()": "Unit"}.get(tycon, tycon)


def instance_name(o: str, tycon: str) -> str:
    mangled = mangle(o)
    return f"i_{mangled.removeprefix('v_')}_{mangle_tycon(tycon)}"


def constructor_name(name: str) -> str:
    return f"c_{name}"


def dictionary_param(o: str, var: str) -> str:
    return f"d_{mangle(o)}_{var}"


def apply(function: str, arguments: Sequence[str]) -> str:
    """``f(a_1)(a_2)...``: a curried application."""
    return function + "".join(f"({a})" for a in arguments)


# --------------------------------------------------------------------------
# Evidence trees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceTree:
    name: str
    tycon: str
    arguments: tuple["Tree", ...]


@dataclass(frozen=True)
class ParamTree:
    """A dictionary parameter of the enclosing function (dictionary passing only)."""

    python_name: str


@dataclass(frozen=True)
class UndefTree:
    pass


type Tree = InstanceTree | ParamTree | UndefTree


def tree_key(tree: Tree) -> str:
    match tree:
        case InstanceTree(name=name, tycon=tycon, arguments=args):
            key = f"{mangle(name)}_{mangle_tycon(tycon)}"
            if args:
                key += f"_of_{'_and_'.join(tree_key(a) for a in args)}_end"
            return key
        case UndefTree():
            return "undef"
        case ParamTree(python_name=python_name):
            return python_name


def specialization_name(base: str, trees: Sequence[Tree]) -> str:
    """The name of the copy of ``base`` for the dictionaries ``trees``."""
    return f"{base}__{'__'.join(tree_key(t) for t in trees)}"


@dataclass
class Context:
    """What the evidence recorded by the type checker means at this point of
    the generated code."""

    # (o, var) -> evidence for the dictionary parameters of enclosing bindings
    params: dict[tuple[str, str], Tree] = field(default_factory=dict)
    # for instance bodies: constraints of the inferred scheme -> their evidence
    body_evidence: dict[tuple[str, str], Constraint] = field(default_factory=dict)
    # skolem type constructor -> quantified variable of the declared scheme
    skolem_vars: dict[str, str] = field(default_factory=dict)
    # the Python expression for a recursive occurrence of the binding being defined
    self_expr: str = ""
    # constrained local lets in scope (monomorphization)
    local_lets: dict[str, "LocalLet"] = field(default_factory=dict)

    def with_params(self, params: dict[tuple[str, str], Tree]) -> "Context":
        return replace(self, params={**self.params, **params})


# --------------------------------------------------------------------------
# Code generation
# --------------------------------------------------------------------------


@dataclass
class Specialization:
    decl: Decl
    trees: tuple[Tree, ...]
    python_name: str


class Emitter:
    """Statement lines of the function body currently being generated."""

    def __init__(self, indent: int) -> None:
        self.lines: list[str] = []
        self.indent = indent

    def add(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def add_lines(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.add(line)

    def child(self) -> "Emitter":
        """An emitter for a nested block; ``extend`` it back in when done."""
        return Emitter(self.indent + 1)

    def extend(self, block: "Emitter") -> None:
        self.lines.extend(block.lines)

    def placeholder(self, comment: str) -> int:
        """Reserve a line to be filled in by ``fill`` later."""
        self.add(f"# {comment}")
        return len(self.lines) - 1

    def fill(self, position: int, block: "Emitter") -> None:
        """Replace the placeholder at ``position`` by the lines of ``block``.
        Positions after it shift, so fill placeholders last-to-first."""
        self.lines[position : position + 1] = block.lines


class CodeGenerator:
    def __init__(self, typed: TypedProgram, strategy: Strategy) -> None:
        self.typed = typed
        self.strategy = strategy
        self.monomorphize = strategy == Strategy.MONOMORPHIZATION
        self.counter = 0
        self.decl_of_name: dict[str, FunctionDecl] = {
            d.name: d for d in typed.decls if isinstance(d, FunctionDecl)
        }
        self.instance_of: dict[tuple[str, str], InstanceInfo] = {
            (d.name, d.tycon): d for d in typed.decls if isinstance(d, InstanceInfo)
        }
        # monomorphization: specializations already generated / to generate
        self.specializations: dict[tuple[int, str], Specialization] = {}
        self.pending: deque[Specialization] = deque()
        self.emitted: set[str] = set()
        self.module = Emitter(0)

    def fresh(self, base: str) -> str:
        self.counter += 1
        return f"_{base}{self.counter}"

    # --- program -----------------------------------------------------------------

    def generate(self) -> str:
        self.module.add_lines(Path(runtime.__file__).read_text().splitlines())
        self.module.add("")
        self.module.add("")
        self.module.add("# --- Data constructors ---")
        for con in self.typed.constructors.values():
            self.module.add(
                f"{constructor_name(con.name)} = constructor("
                f"{con.name!r}, {con.tycon!r}, {len(con.field_types)})",
            )
        for decl in self.typed.decls:
            self.module.add("")
            self.generate_decl(decl)
        return "\n".join(self.module.lines) + "\n"

    def generate_decl(self, decl: Decl) -> None:
        if self.monomorphize and decl.scheme.dictionary_params:
            # generated on demand, once per tuple of dictionaries (see flush_pending)
            return
        params = [dictionary_param(o, var) for o, var in decl.scheme.dictionary_params]
        trees = tuple(ParamTree(param) for param in params)
        lines = self.generate_binding(
            self.python_name(decl),
            decl,
            params,
            self.context_for(decl, trees),
        )
        self.flush_pending()
        self.module.add_lines(lines)

    def python_name(self, decl: Decl) -> str:
        match decl:
            case FunctionDecl(name=name):
                return mangle(name)
            case InstanceInfo(name=name, tycon=tycon):
                return instance_name(name, tycon)

    def context_for(
        self,
        decl: Decl,
        trees: tuple[Tree, ...],
    ) -> Context:
        """The evidence for the dictionary parameters of ``decl``."""
        params = dict(zip(decl.scheme.dictionary_params, trees))
        match decl:
            case FunctionDecl():
                return Context(params=params)
            case InstanceInfo():
                return Context(
                    params=params,
                    body_evidence=decl.body_evidence,
                    skolem_vars=decl.skolem_vars,
                )

    def flush_pending(self) -> None:
        """Emit the specializations requested while generating a declaration.

        They are emitted before it; everything they refer to was declared
        earlier (declarations are scoped sequentially)."""
        while self.pending:
            spec = self.pending.popleft()
            if spec.python_name in self.emitted:
                continue
            self.emitted.add(spec.python_name)
            lines = self.generate_binding(
                spec.python_name,
                spec.decl,
                [],
                self.context_for(spec.decl, spec.trees),
            )
            # generating it may have requested further specializations
            self.flush_pending()
            self.module.add_lines(lines)
            self.module.add("")

    # --- bindings --------------------------------------------------------------

    def generate_binding(
        self,
        python_name: str,
        decl: Decl,
        dictionary_params: list[str],
        context: Context,
    ) -> list[str]:
        """``python_name = \\d_1 ... d_k . \\a_1 ... a_n . match clauses``"""
        context = replace(context, self_expr=apply(python_name, dictionary_params))
        emitter = Emitter(0)
        if not dictionary_params and decl.arity == 0:
            value = self.compile_expression(decl.clauses[0].body, context, emitter)
            emitter.add(f"{python_name} = {value}")
            return emitter.lines
        args = [f"a{i}" for i in range(decl.arity)]

        def fill_body(body: Emitter) -> None:
            if decl.arity == 0:
                value = self.compile_expression(decl.clauses[0].body, context, body)
                body.add(f"return {value}")
            else:
                for clause in decl.clauses:
                    self.compile_clause(clause, args, context, body)
                body.add(f"return pattern_match_failure({decl.name!r})")

        self.emit_curried(
            emitter,
            python_name,
            "impl",
            dictionary_params + args,
            fill_body,
        )
        return emitter.lines

    def emit_curried(
        self,
        emitter: Emitter,
        python_name: str,
        base: str,
        params: list[str],
        fill_body: Callable[[Emitter], None],
    ) -> None:
        """``python_name = curry(n, impl)`` for a fresh ``def impl(params)``."""
        impl = self.fresh(base)
        emitter.add(f"def {impl}({', '.join(params)}):")
        body = emitter.child()
        fill_body(body)
        emitter.extend(body)
        emitter.add(f"{python_name} = curry({len(params)}, {impl})")

    def compile_clause(
        self,
        clause: FunctionDefinition,
        args: list[str],
        context: Context,
        emitter: Emitter,
    ) -> None:
        conditions: list[str] = []
        bindings: list[tuple[str, str]] = []
        for pattern, arg in zip(clause.patterns, args):
            self.compile_pattern(pattern, arg, conditions, bindings)
        condition = " and ".join(conditions) if conditions else "True"
        emitter.add(f"if {condition}:")
        body = emitter.child()
        for name, value in bindings:
            body.add(f"{mangle(name)} = {value}")
        result = self.compile_expression(clause.body, context, body)
        body.add(f"return {result}")
        emitter.extend(body)

    def compile_pattern(
        self,
        pattern: Pattern,
        subject: str,
        conditions: list[str],
        bindings: list[tuple[str, str]],
    ) -> None:
        match pattern:
            case VariablePattern(name=name):
                bindings.append((name, subject))
            case LiteralPattern(value=literal):
                conditions.append(f"{subject} == {self.compile_literal(literal)}")
            case NegativeIntPattern(value=value) | NegativeFloatPattern(value=value):
                conditions.append(f"{subject} == {value!r}")
            case ConstructorPattern(constructor=name, patterns=subpatterns):
                conditions.append(f"{subject}.name == {name!r}")
                for index, sub in enumerate(subpatterns):
                    self.compile_pattern(
                        sub,
                        f"{subject}.args[{index}]",
                        conditions,
                        bindings,
                    )
            case ConsPattern(head=head, tail=tail):
                conditions.append(f"len({subject}) > 0")
                self.compile_pattern(head, f"{subject}[0]", conditions, bindings)
                self.compile_pattern(tail, f"{subject}[1:]", conditions, bindings)
            case ListPattern(patterns=subpatterns):
                conditions.append(f"len({subject}) == {len(subpatterns)}")
                for index, sub in enumerate(subpatterns):
                    self.compile_pattern(
                        sub,
                        f"{subject}[{index}]",
                        conditions,
                        bindings,
                    )
            case TuplePattern(patterns=subpatterns):
                for index, sub in enumerate(subpatterns):
                    self.compile_pattern(
                        sub,
                        f"{subject}[{index}]",
                        conditions,
                        bindings,
                    )
            case _:
                raise CompileError(f"Unhandled pattern {type(pattern).__name__}")

    # --- expressions --------------------------------------------------------------

    def compile_literal(self, literal: Expression) -> str:
        match literal:
            case IntLiteral(value=value) | NegativeInt(value=value):
                return f"({value!r})" if value < 0 else repr(value)
            case FloatLiteral(value=value) | NegativeFloat(value=value):
                return f"({value!r})" if value < 0 else repr(value)
            case StringLiteral(value=value):
                return repr(value)
            case CharLiteral(value=value):
                return f"Char({value!r})"
            case BoolLiteral(value=value):
                return repr(value)
            case _:
                raise CompileError(f"Not a literal: {type(literal).__name__}")

    def compile_expression(
        self,
        expr: Expression,
        context: Context,
        emitter: Emitter,
    ) -> str:
        """Returns a Python expression; statements it needs (nested function
        definitions for ``do`` blocks) are added to ``emitter`` first."""
        match expr:
            case (
                IntLiteral()
                | NegativeInt()
                | FloatLiteral()
                | NegativeFloat()
                | StringLiteral()
                | CharLiteral()
                | BoolLiteral()
            ):
                return self.compile_literal(expr)
            case ListLiteral(elements=elements):
                items = [self.compile_expression(e, context, emitter) for e in elements]
                return f"[{', '.join(items)}]"
            case TupleLiteral(elements=elements):
                items = [self.compile_expression(e, context, emitter) for e in elements]
                return f"({''.join(f'{item}, ' for item in items)})"
            case Variable() | Constructor():
                return self.compile_variable(expr, context)
            case FunctionApplication(function=function, argument=argument):
                f = self.compile_expression(function, context, emitter)
                a = self.compile_expression(argument, context, emitter)
                return f"{f}({a})"
            case IfElse(condition=condition, then_expr=then_expr, else_expr=else_expr):
                c = self.compile_expression(condition, context, emitter)
                t = self.compile_expression(then_expr, context, emitter)
                e = self.compile_expression(else_expr, context, emitter)
                return f"({t} if {c} else {e})"
            case GroupedExpression(expression=inner):
                return self.compile_expression(inner, context, emitter)
            case DoBlock(statements=statements):
                return self.compile_block(statements, context, emitter)
            case ConstructorExpression(constructor_name=name, fields=fields):
                info = self.typed.constructors[name]
                values = {
                    f.field_name: self.compile_expression(f.value, context, emitter)
                    for f in fields
                }
                return apply(
                    constructor_name(name),
                    [values[field_name] for field_name in info.field_names],
                )
            case _:
                raise CompileError(f"Unhandled expression {type(expr).__name__}")

    def compile_variable(
        self,
        node: Variable | Constructor,
        context: Context,
    ) -> str:
        use = self.typed.var_use(node)
        match use.kind:
            case UseKind.MONO:
                return mangle(use.name)
            case UseKind.PRIM:
                return use.name
            case UseKind.CONSTRUCTOR:
                return constructor_name(use.name)
            case UseKind.SELF:
                return context.self_expr
            case UseKind.OVERLOADED:
                assert use.constraint is not None
                return self.compile_tree(self.closed_tree(use.constraint, context))
            case UseKind.LET:
                trees = tuple(self.closed_tree(c, context) for c in use.evidence)
                return self.compile_let_use(use.name, trees, context)

    def compile_let_use(
        self,
        name: str,
        trees: tuple[Tree, ...],
        context: Context,
    ) -> str:
        if not trees:
            return mangle(name)
        if self.monomorphize:
            local = context.local_lets.get(name)
            if local is not None:
                return local.request(trees)
            return self.request_specialization(self.decl_of_name[name], trees)
        return apply(mangle(name), [self.compile_tree(tree) for tree in trees])

    # --- evidence -------------------------------------------------------------------

    def closed_tree(self, constraint: Constraint, context: Context) -> Tree:
        """The evidence for ``constraint`` as seen from ``context``."""
        evidence = resolve_evidence(constraint)
        match evidence:
            case InstanceEvidence(name=name, tycon=tycon, arguments=arguments):
                if tycon in context.skolem_vars:
                    return self.param_tree(name, context.skolem_vars[tycon], context)
                return InstanceTree(
                    name,
                    tycon,
                    tuple(self.closed_tree(a, context) for a in arguments),
                )
            case ParamEvidence(name=name, var=var):
                if (name, var) in context.body_evidence:
                    return self.closed_tree(context.body_evidence[(name, var)], context)
                return self.param_tree(name, var, context)
            case AmbiguousEvidence():
                return UndefTree()

    def param_tree(self, name: str, var: str, context: Context) -> Tree:
        tree = context.params.get((name, var))
        if tree is None:
            raise CompileError(f"No dictionary for constraint {name} on {var}")
        return tree

    def compile_tree(self, tree: Tree) -> str:
        match tree:
            case UndefTree():
                return "undef"
            case ParamTree(python_name=python_name):
                return python_name
            case InstanceTree(name=name, tycon=tycon, arguments=arguments):
                if self.monomorphize and arguments:
                    return self.request_specialization(
                        self.instance_of[(name, tycon)],
                        arguments,
                    )
                return apply(
                    instance_name(name, tycon),
                    [self.compile_tree(argument) for argument in arguments],
                )

    def request_specialization(self, decl: Decl, trees: tuple[Tree, ...]) -> str:
        """The name of the copy of ``decl`` for the dictionaries ``trees``;
        it is generated when the current declaration is done (``flush_pending``)."""
        python_name = specialization_name(self.python_name(decl), trees)
        key = (id(decl), python_name)
        if key not in self.specializations:
            spec = Specialization(decl, trees, python_name)
            self.specializations[key] = spec
            self.pending.append(spec)
        return python_name

    # --- blocks -------------------------------------------------------------------

    def compile_block(
        self,
        statements: Sequence[Statement],
        context: Context,
        emitter: Emitter,
    ) -> str:
        """``do { s_1; ...; s_n }`` becomes a nested function returning the
        value of the last statement (``()`` when it is a ``let``)."""
        name = self.fresh("block")
        emitter.add(f"def {name}():")
        body = emitter.child()
        local_lets: list[LocalLet] = []
        result = "None"  # the unit value, when the block ends with a let
        for stmt in statements:
            match stmt:
                case LetStatement(variable=var, value=value):
                    local = self.compile_let(stmt, var, value, context, body)
                    if local is not None:
                        local_lets.append(local)
                        context = replace(
                            context,
                            local_lets={**context.local_lets, var: local},
                        )
                    result = "None"
                case _:
                    assert is_expression(stmt)
                    body.add(f"_ = {self.compile_expression(stmt, context, body)}")
                    result = "_"
        body.add(f"return {result}")
        # Constrained local lets (monomorphization): the uses in the rest of
        # the block are now known.  Later lets may use earlier ones, so they
        # are generated last-to-first, which also keeps the positions valid.
        for local in reversed(local_lets):
            body.fill(local.position, self.generate_local_specializations(local, body))
        emitter.extend(body)
        return f"{name}()"

    def compile_let(
        self,
        stmt: LetStatement,
        name: str,
        value: Expression,
        context: Context,
        emitter: Emitter,
    ) -> "LocalLet | None":
        scheme = self.typed.let_scheme(stmt)
        if not scheme.dictionary_params:
            emitter.add(
                f"{mangle(name)} = {self.compile_expression(value, context, emitter)}",
            )
            return None
        if self.monomorphize:
            position = emitter.placeholder(f"let {name}")
            return LocalLet(name, value, scheme, context, position)
        params = [dictionary_param(o, var) for o, var in scheme.dictionary_params]
        inner = context.with_params(
            {
                (o, var): ParamTree(dictionary_param(o, var))
                for o, var in scheme.dictionary_params
            },
        )

        def fill_body(body: Emitter) -> None:
            body.add(f"return {self.compile_expression(value, inner, body)}")

        self.emit_curried(emitter, mangle(name), "let", params, fill_body)
        return None

    def generate_local_specializations(
        self,
        local: "LocalLet",
        block: Emitter,
    ) -> Emitter:
        """One copy of the let per distinct tuple of dictionaries it was used
        with; generating a copy may request further copies."""
        emitter = Emitter(block.indent)
        done: set[str] = set()
        while len(done) < len(local.requests):
            for python_name, trees in list(local.requests.items()):
                if python_name in done:
                    continue
                done.add(python_name)
                inner = local.context.with_params(
                    dict(zip(local.scheme.dictionary_params, trees)),
                )
                value = self.compile_expression(local.value, inner, emitter)
                emitter.add(f"{python_name} = {value}")
        return emitter


@dataclass
class LocalLet:
    """A constrained ``let`` inside a block, specialized per use (monomorphization)."""

    name: str
    value: Expression
    scheme: Scheme
    context: Context
    position: int  # of the placeholder line in the block
    requests: dict[str, tuple[Tree, ...]] = field(default_factory=dict)

    def request(self, trees: tuple[Tree, ...]) -> str:
        python_name = specialization_name(mangle(self.name), trees)
        self.requests[python_name] = trees
        return python_name


def generate(program: Program, strategy: Strategy) -> str:
    return CodeGenerator(infer_program(program), strategy).generate()
