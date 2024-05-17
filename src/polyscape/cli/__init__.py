"""Command Line Interface"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace

# Internal Imports
from polyscape.cli import train


# -------------------------------------------------------------------------
#   Entry-Point
# -------------------------------------------------------------------------


def main(argv: tuple[str, ...] | None = None):
    args = parse_cli_arguments(argv)
    args.execute(args)


# -------------------------------------------------------------------------
#   Non-Exported Functions
# -------------------------------------------------------------------------


def parse_cli_arguments(argv: tuple[str, ...] | None = None) -> Namespace:
    """
    NOTE: polyscape train <algorithm> <agent version> [args]
    """
    parser = ArgumentParser()
    cmd_subparsers = parser.add_subparsers(title="Command")
    train.add_arguments(cmd_subparsers.add_parser("train"))

    return parser.parse_args(argv)
