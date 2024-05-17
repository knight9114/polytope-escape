"""Train CLI Arguments"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
import secrets

# External Imports
import gymnasium as gym
import jax
from tensorboardX import SummaryWriter  # type: ignore

# Internal Imports
from polyscape.api.train import reinforce, reinforce_with_baseline
from polyscape.agents.v0.mlp import actor_critic_agent_init
from polyscape.optim import adam, schedules


# -------------------------------------------------------------------------
#   Constants
# -------------------------------------------------------------------------

DIFFICULTIES: tuple[str, ...] = ("veryeasy", "easy", "medium")


# -------------------------------------------------------------------------
#   CLI Functions
# -------------------------------------------------------------------------


def add_arguments(parser: ArgumentParser):
    parser.set_defaults(execute=execute)
    algos = parser.add_subparsers(title="Algorithm")
    add_reinforce_arguments(algos.add_parser("reinforce"))
    add_reinforce_with_baseline_arguments(algos.add_parser("reinforce-with-baseline"))


# -------------------------------------------------------------------------
#   Non-Exported Functions
# -------------------------------------------------------------------------


def execute(args: Namespace):
    seed = secrets.randbits(30) if args.seed is None else args.seed
    rng = jax.random.key(seed)
    env = args.environment_factory(args)
    params, agent_fn = args.agent_factory(args, env, rng)
    optstate, opt_fn = args.optimizer_factory(args, params)
    lr_schedule = args.lr_schedule_factory(args)
    logger = args.logger_factory(args)
    train_fn = args.train_fn_factory(args)

    train_fn(
        agent_fn=agent_fn,
        params=params,
        optimizer_fn=opt_fn,
        optstate=optstate,
        lr_schedule_fn=lr_schedule,
        env=env,
        logger=logger,
    )


def add_reinforce_arguments(parser: ArgumentParser):
    parser.set_defaults(
        train_fn_factory=lambda args: partial(
            reinforce.train,
            total_timesteps=args.total_timesteps,
            discount_factor=args.discount_factor,
        )
    )

    def add_args(p: ArgumentParser):
        algo = p.add_argument_group("REINFORCE Hyperparameters")
        algo.add_argument("--total-timesteps", type=int, default=1000)
        algo.add_argument("--discount-factor", type=float, default=0.99)

    agent_subparsers = parser.add_subparsers(title="Agent Version")
    v0: ArgumentParser = agent_subparsers.add_parser("v0")
    add_experiment_arguments(v0)
    add_actor_critic_v0_arguments(v0)
    add_args(v0)


def add_reinforce_with_baseline_arguments(parser: ArgumentParser):
    parser.set_defaults(
        train_fn_factory=lambda args: partial(
            reinforce_with_baseline.train,
            total_timesteps=args.total_timesteps,
            discount_factor=args.discount_factor,
        )
    )

    def add_args(p: ArgumentParser):
        algo = p.add_argument_group("REINFORCE Hyperparameters")
        algo.add_argument("--total-timesteps", type=int, default=1000)
        algo.add_argument("--discount-factor", type=float, default=0.99)

    agent_subparsers = parser.add_subparsers(title="Agent Version")
    v0: ArgumentParser = agent_subparsers.add_parser("v0")
    add_experiment_arguments(v0)
    add_actor_critic_v0_arguments(v0)
    add_args(v0)


def add_experiment_arguments(parser: ArgumentParser):
    parser.set_defaults(
        logger_factory=logger_from_namespace,
        environment_factory=lambda args: gym.make(make_env_name(args.difficulty)),
    )

    exp = parser.add_argument_group("Experiment Arguments")
    exp.add_argument("--difficulty", choices=DIFFICULTIES, default="veryeasy")
    exp.add_argument("--root-directory", type=Path, default=Path("./experiments"))
    exp.add_argument("--agent-name", default="agent")
    exp.add_argument("--agent-version")
    exp.add_argument("--seed", type=int)


def add_actor_critic_v0_arguments(parser: ArgumentParser):
    parser.set_defaults(
        agent_factory=lambda args, env, rng: actor_critic_agent_init(
            rng,
            env.observation_space.shape,
            env.action_space.n,
            args.d_hidden_layers,
            args.n_hidden_layers,
            dropout=args.dropout,
        ),
        optimizer_factory=lambda args, params: adam.init(
            params,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            weight_decay=args.weight_decay,
        ),
        lr_schedule_factory=lambda args: partial(schedules.constant, lr=args.lr),
    )

    hparams = parser.add_argument_group("Agent Hyperparameters")
    hparams.add_argument("--d-hidden-layers", type=int, default=64)
    hparams.add_argument("--n-hidden-layers", type=int, default=2)
    hparams.add_argument("--dropout", type=float, default=0.0)
    hparams.add_argument("--no-use-bias", dest="use_bias", action="store_false")
    hparams.add_argument("--lr", type=float, default=1e-3)
    hparams.add_argument("--beta-1", type=float, default=0.9)
    hparams.add_argument("--beta-2", type=float, default=0.95)
    hparams.add_argument("--weight-decay", type=float, default=0.0)


def make_experiment_path(root: Path, name: str | None, version: str | None) -> Path:
    prefix = root / (name or "agent")
    found_dirs = list(prefix.glob("*"))

    if version is not None:
        assert (prefix / version) not in found_dirs

    idx = 0
    base = "version_{}"
    while (prefix / base.format(idx)) in found_dirs:
        idx += 1

    return prefix / base.format(idx)


def logger_from_namespace(args: Namespace) -> SummaryWriter:
    return SummaryWriter(
        logdir=make_experiment_path(
            args.root_directory,
            args.agent_name,
            args.agent_version,
        ).as_posix(),
        flush_secs=30,
    )


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
