import os

from lark import Lark, ParseTree


def parse(path: str) -> ParseTree:
    parser = Lark.open(
        "./lango/minio/minio.lark",
        parser="lalr",
    )

    # Load prelude files
    prelude_dir = "./lango/minio/prelude"
    prelude_content = ""

    if os.path.exists(prelude_dir):
        for filename in sorted(os.listdir(prelude_dir)):
            if filename.endswith(".minio"):
                prelude_file_path = os.path.join(prelude_dir, filename)
                try:
                    with open(prelude_file_path, "r") as prelude_file:
                        prelude_content += prelude_file.read() + "\n"
                except FileNotFoundError:
                    pass  # Skip missing files

    # Read main file
    with open(path) as f:
        main_content = f.read()

    # Combine prelude and main content
    combined_content = main_content + prelude_content

    return parser.parse(combined_content)
