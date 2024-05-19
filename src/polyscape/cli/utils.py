"""CLI Utilities"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser

# External Imports
import gymnasium as gym


# -------------------------------------------------------------------------
#   Constants
# -------------------------------------------------------------------------

DIFFICULTIES: tuple[str, ...] = ("veryeasy", "easy", "medium")
REWARD_FUNCTIONS: tuple[str, ...] = (
    "nonzero",
    "sparse",
    "optimal-aware",
    "scaled-optimal-aware",
    "inverse-scaled-optimal-aware",
)


# -------------------------------------------------------------------------
#   Environment Functions
# -------------------------------------------------------------------------


def add_environment_arguments(parser: ArgumentParser):
    parser.set_defaults(
        environment_factory=lambda args: gym.make(
            make_env_name(args.difficulty),
            **{key: val for key, val in vars(args).items() if val is not None},
        ),
    )

    env = parser.add_argument_group("Basic Environment Arguments")
    env.add_argument("--difficulty", choices=DIFFICULTIES, default="veryeasy")
    env.add_argument("--seed", type=int)

    exact = parser.add_argument_group("Exact Environment Arguments")
    exact.add_argument("--n-axes", type=int)
    exact.add_argument("--n-polytope-distributions", type=int)
    exact.add_argument("--t-deadline", type=int)
    exact.add_argument("--reward-method", choices=REWARD_FUNCTIONS)
    exact.add_argument("--failure-reward", type=float)
    exact.add_argument("--success-reward", type=float)
    exact.add_argument("--step-reward", type=float)
    exact.add_argument("--min-distribution-concentration", type=int)
    exact.add_argument("--max-distribution-concentration", type=int)
    exact.add_argument("--min-escape-distance", type=float)
    exact.add_argument("--max-escape-distance", type=float)
    exact.add_argument("--step-size", type=float)
    exact.add_argument("--use-stochastic-step-size", type=bool)
    exact.add_argument("--stochastic-step-size-scale", type=float)
    exact.add_argument("--distribution-concentrations", type=int, nargs="*")
    exact.add_argument("--distribution-distribution", type=float, nargs="*")
    exact.add_argument("--render-mode", type=str)
    exact.add_argument("--dtype", type=str)


def make_env_name(d: str) -> str:
    base = "PolytopeEscape{}-v0"
    match d:
        case "veryeasy":
            return base.format("VeryEasy")

        case "easy":
            return base.format("Easy")

        case "medium":
            return base.format("Medium")

        case _:
            raise ValueError
