# lango

[![Build Status](https://github.com/antonkesy/lango/workflows/Docker%20Build/badge.svg)](https://github.com/antonkesy/lango/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

Implementation of the programming language SystemO as described in ["A Second Look at Overloading"](https://dl.acm.org/doi/pdf/10.1145/224164.224195) published in 1995 by Martin Odersky, Philip Wadler, and Martin Wehr.
Practical part of my Master's thesis, ["Implementing a Compiler for System O: A Sound Type System with Principal Types for Overloading"](TODO).

This Python project provides a command-line interface for parsing, type checking, interpreting, and compiling programs written for two languages: MiniO and SystemO.

- MiniO: A minimal subset of SystemO without overloading support.
  - Compiler targets: Python and Go.
- SystemO: The language described in the paper, featuring overloading.
  - Compiler target: Python.
  - Two strategies for handling overloading: Monomorphization and Dictionary Passing.

## MiniO Example

See a more complex example at [`examples/minio/example.minio`](examples/minio/example.minio) or checkout the test files in [`./test/files/minio/`](./test/files/minio/)

```haskell
data Point = MkPoint Float Float;

printPoint (MkPoint x y) = "Point(" ++ show x ++ ", " ++ show y ++ ")";

isTrue True = "True";
isTrue False = "False";

main = do {
  let { p = MkPoint 3.0 4.0 };
  putStr ("Point p: " ++ printPoint p);
  putStr ("isTrue True: " ++ isTrue True);
  putStr ("isTrue False: " ++ isTrue False);
};
```

`lango run minio examples/minio/short.minio`:

> Point p: Point(3.0, 4.0)isTrue True: TrueisTrue False: False

## SystemO Example

See a more complex example at [`examples/systemo/example.syso`](examples/systemo/example.syso) or checkout the test files in [`./test/files/systemo/`](./test/files/systemo/)

```haskell
-- Overloads for f
inst f :: Int -> String {
  f x = "Int: " ++ show x;
};

inst f :: Bool -> String {
  f True = "True";
  f False = "False";
};

-- Overloaded functions
main = do {
    putStr (f 10);
    putStr (f True);
};
```

`lango run systemo examples/systemo/short.syso`:

> Int: 10True

### System O in a nutshell

The implementation follows the paper closely; the type checker is the
constrained unification / type reconstruction algorithm of Section 6, the
`dictionary_passing` compiler is the translation of Section 4, the
`monomorphization` compiler resolves the same dictionaries at compile time,
and the interpreter implements the untyped dynamic semantics of Section 3
(overloaded functions dispatch on the type constructor of their first
argument).

- **Overloaded identifiers** have no class declaration. An identifier is
  overloaded by giving it instances: `inst o :: sigma { clauses }`.
  Every use of `o` has the type `(o :: a -> b) => a -> b`; the constraint is
  discharged when `a` becomes known.
- **Instance types** must have the form `T a1 ... an -> t` where the `ai` are
  distinct type variables and `t` only mentions the `ai`: an instance works
  uniformly for all values built from the type constructor `T`, and the
  argument type determines the result type. `o` has at most one instance per
  type constructor.
- **Constraints** on the type variables of an instance are written in front
  of the type, e.g. `inst show :: (show :: a -> String) => [a] -> String`.
  A constraint `o :: a -> t` says that `o` must be defined at `a` with
  result type `t`.
- **Inferred types** are constrained type schemes, for example
  `elem :: ((==) :: a -> b -> Bool) => a -> [b] -> Bool`
  (`lango types systemo file.syso` prints them). No type annotations are
  ever required: every typable program has a principal type and no program is
  ambiguous (`[] == []` is `True`).
- **Instance declarations are not recursive**: the body of an instance of `o`
  for `T` cannot use `o` at `T` (use a helper function instead).
- **Declarations are scoped sequentially**, like `let u = e in p` and
  `inst o :: s = e in p` in the paper: a function or instance is visible in
  the declarations that follow it. Functions may be recursive (monomorphic
  recursion) and consist of several clauses.
- **Numeric literals are not overloaded**: `1` is an `Int` and `1.0` a
  `Float`. Prefix minus `-x` is sugar for the overloaded function `negate x`.
- The [prelude](./lango/systemo/prelude/) is written in SystemO itself on top
  of a few primitives (`primIntAdd`, ...); it is just a normal program
  prefix.

## Installation

### Containerized Setup

A [Dockerfile](./Dockerfile) is provided for easy setup and experimentation without local installation.

```bash
# Build the Docker image
docker build -t lango .

# Run the container with the lango CLI
docker run -it --rm lango <OPTIONS>
docker run -it --rm lango run minio examples/minio/example.minio
```

### Local Setup

#### Prerequisites

- Python 3.13

#### Example Installation from Source

```bash
git clone https://github.com/antonkesy/lango.git
cd lango
python3.13 -m venv venv # Create a virtual environment
source venv/bin/activate # Activate the virtual environment
pip install .
```

> **Tip**: A [Makefile](./Makefile) is available with convenient commands for tasks. Run `make` to see available commands.

## Usage

The lango CLI provides several commands for working with both languages:

### Basic Commands and Examples

```bash
lango --help

# Parse a file and print the AST
lango parse <language> <file>
lango parse minio examples/minio/example.minio
lango parse systemo examples/systemo/example.syso

# Prints all types of source file
lango types <language> <file>
lango types minio examples/minio/example.minio
lango types systemo examples/systemo/example.syso

# Interpret a source file
lango run <language> <file>
lango run minio examples/minio/example.minio
lango run systemo examples/systemo/example.syso

# Compile to target language
lango compile <language> <file> -o <output> [--target <target>] [--strategy <strategy>]
lango compile minio examples/minio/example.minio -o output.py --target python
lango compile minio examples/minio/example.minio -o output.go --target go
lango compile systemo examples/systemo/example.syso -o output.py --strategy monomorphization
lango compile systemo examples/systemo/example.syso -o output.py --strategy dictionary_passing
```

## Tests

Unit tests covering all aspects of the implementation are located in the [`./test/`](./test/) directory.

> **Tip**: The tests can be run inside the Docker container as well using `docker build --target test -t lango-test .` and then `docker run -it --rm lango-test`

### Requirements

- Python 3.13
- `runghc` to run Haskell test files
- `go` to run Go test files

### Installation

```bash
pip install . [dev]
```

## Benchmarks

A elementary benchmark for a [naive Fibonacci implementation](./benchmark/fib.minio) is provided for all languages under [`./benchmark/`](./benchmark/).

Run automated benchmarks with:

```bash
python3 ./benchmark/benchmark.py
```
