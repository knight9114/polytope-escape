"""REINFORCE Trainer"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = ["train"]


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from functools import partial
import secrets

# External Imports
import gymnasium as gym
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree
import tqdm
from tensorboardX import SummaryWriter  # type: ignore

# Internal Imports
from polyscape.typing import (
    ActorCriticForwardFn,
    OptimizerStepFn,
    LearningRateScheduleFn,
)


# -------------------------------------------------------------------------
#   Exported Functions
# -------------------------------------------------------------------------


@partial(jax.value_and_grad, argnums=1, has_aux=True)
def episode_objective_fn(
    agent_fn: ActorCriticForwardFn,
    params: PyTree[Float[Array, "..."], "A"],
    env: gym.Env,
    discount_factor: float,
    seed: int,
) -> tuple[PyTree[Float[Array, "..."], "A"], dict[str, Array]]:
    rng = jax.random.key(seed)
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    loss = jnp.zeros(())
    discounted_returns = jnp.zeros(())
    total_rewards = jnp.zeros(())

    pbar = tqdm.trange(
        getattr(env, "spec.max_episode_steps", 128),
        desc="step",
        leave=False,
    )
    while not done:
        kfn, kpi, rng = jax.random.split(rng, 3)
        pi, _ = agent_fn(params, obs, rng=kfn)
        action = jax.random.categorical(kpi, pi)
        logprob = jax.nn.log_softmax(pi, -1)[action]

        obs, reward, finished, truncated, _ = env.step(action)
        done = finished or truncated
        step += 1

        discounted_returns = reward + (discounted_returns * discount_factor)
        total_rewards += reward
        loss = loss - logprob
        pbar.update()

    pbar.close()
    loss = (loss / step) * discounted_returns
    info = {
        "episode-length": jnp.array(step),
        "total-rewards": jnp.array(total_rewards),
        "discounted-returns": jnp.array(discounted_returns),
        "loss": loss,
    }

    return loss, info


def train(
    agent_fn: ActorCriticForwardFn,
    params: PyTree[Float[Array, "..."], "A"],
    optimizer_fn: OptimizerStepFn,
    optstate: PyTree[Float[Array, "..."], "O"],
    lr_schedule_fn: LearningRateScheduleFn,
    env: gym.Env,
    logger: SummaryWriter | None = None,
    total_timesteps: int = 10_000,
    discount_factor: float = 0.99,
    seed: int | None = None,
) -> PyTree[Float[Array, "..."], "A"]:
    seed = secrets.randbits(30) if seed is None else seed

    for episode in tqdm.trange(total_timesteps, desc="episodes"):
        (_, logdict), grads = episode_objective_fn(
            agent_fn,
            params,
            env=env,
            discount_factor=discount_factor,
            seed=seed + episode,
        )

        params, optstate = optimizer_fn(
            params=params,
            grads=grads,
            optstate=optstate,
            step=episode,
            lr=lr_schedule_fn(episode),
        )

        if logger is not None:
            for key, val in logdict.items():
                if jnp.isinf(val) or jnp.isnan(val):
                    print(f"FOUND INF/NAN: {episode=}  {key=}  {val=}")
                logger.add_scalar(key, jnp.array(val).item(), episode)
