from lark import Lark, ParseTree


def parse(path: str) -> ParseTree:
    parser = Lark.open(
        "./lango/minio/minio.lark",
        parser="lalr",
    )

    with open(path) as f:
        return parser.parse(f.read())
