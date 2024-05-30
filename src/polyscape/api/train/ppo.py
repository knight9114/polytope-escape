"""PPO Trainer"""

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
from typing import Any

# External Imports
import gymnasium as gym
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
import tqdm
from tensorboardX import SummaryWriter  # type: ignore

# Internal Imports
from polyscape.typing import (
    ActorCriticForwardFn,
    OptimizerStepFn,
    LearningRateScheduleFn,
    ObsType,
)


# -------------------------------------------------------------------------
#   Exported Functions
# -------------------------------------------------------------------------


@partial(jax.value_and_grad, argnums=1, has_aux=True)
def episode_objective_fn(
    agent_fn: ActorCriticForwardFn,
    params: PyTree[Float[Array, "..."], "A"],
    trajectory: PyTree[Array, "T"],
    rng: PRNGKeyArray,
    clipping_bound: float = 0.2,
    discount_factor: float = 0.99,
    gae_factor: float = 0.95,
    value_fn_factor: float = 1.0,
    entropy_fn_factor: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[PyTree[Float[Array, "..."], "A"], dict[str, Array]]:
    k1, k2 = jax.random.split(rng, 2)
    obs_, actions_, rewards_, next_obs_, old_probs_a_, masks_ = zip(*trajectory)

    obs = tuple([jnp.stack(s) for s in zip(*obs_)])
    actions: Int[Array, "horizon"] = jnp.stack(actions_)
    rewards: Float[Array, "horizon"] = jnp.stack(rewards_).reshape(-1, 1)
    old_probs_a = jnp.stack(old_probs_a_)
    masks = jnp.stack(masks_, dtype=jnp.float32).reshape(-1, 1)
    next_obs = tuple([jnp.stack(s) for s in zip(*next_obs_)])

    pi, v = agent_fn(params, obs, rng=k1)
    _, next_v = agent_fn(params, next_obs, rng=k2)

    td_targets = rewards + (discount_factor * next_v * (jnp.ones(()) - masks))
    deltas = td_targets - v
    _, advantages = jax.lax.scan(
        f=partial(_advantage_loop_body, scale=discount_factor * gae_factor),
        init=jnp.zeros((1,), dtype=deltas.dtype),
        xs=jnp.flip(deltas, 0),
    )
    advantages = jnp.flip(advantages, 0)

    new_probs = jax.nn.softmax(pi + epsilon, -1)
    new_probs_a = new_probs[jnp.arange(pi.shape[0]), actions]
    ratios = jnp.exp(jnp.log(new_probs_a) - jnp.log(old_probs_a)).reshape(-1, 1)
    surrogate = -jnp.minimum(
        ratios * advantages,
        jnp.clip(ratios, 1 - clipping_bound, 1 + clipping_bound) * advantages,
    ).mean()
    value_fn_loss = value_fn_factor * jnp.power(v - td_targets, 2).mean()
    entropy = jax.scipy.special.entr(new_probs + epsilon).mean()
    entropy_loss = entropy_fn_factor * entropy

    loss = surrogate + value_fn_loss + entropy_loss
    info = {
        "loss": loss,
        "value-fn-loss": value_fn_loss,
        "entropy": entropy,
        "total-horizon-rewards": jnp.sum(rewards),
        "horizon-timesteps": jnp.array(actions.shape[0]),
    }

    return loss, info


def _advantage_loop_body(advantage, delta, scale):
    advantage = scale * advantage + delta
    return advantage, advantage


def generate_trajectory(
    agent_fn: ActorCriticForwardFn,
    params: PyTree[Float[Array, "..."], "A"],
    env: gym.Env,
    state: tuple[ObsType, float, bool, bool, dict[str, Any]],
    horizon_timesteps: int,
    seed: int,
    epsilon: float = 1e-8,
) -> tuple[int, PyTree[Array, "T"], ObsType]:
    rng = jax.random.key(seed)
    t = 0
    obs, *_ = state

    pbar = tqdm.trange(horizon_timesteps, leave=False, desc="trajectory")
    trajectory = []
    while t < horizon_timesteps:
        kfn, kpi, rng = jax.random.split(rng, 3)
        pi, v = agent_fn(params, obs, rng=kfn)
        action = jax.random.categorical(kpi, pi)
        prob = jax.nn.softmax(pi + epsilon, -1)[action]

        next_obs, reward, finished, truncated, _ = env.step(action)
        done = finished or truncated
        t += 1

        trajectory.append((obs, action, reward, next_obs, prob, done))
        obs = next_obs

        pbar.update()

        if done:
            break

    pbar.close()

    return t, trajectory, obs


def train(
    agent_fn: ActorCriticForwardFn,
    params: PyTree[Float[Array, "..."], "A"],
    optimizer_fn: OptimizerStepFn,
    optstate: PyTree[Float[Array, "..."], "O"],
    lr_schedule_fn: LearningRateScheduleFn,
    env: gym.Env,
    logger: SummaryWriter | None = None,
    total_timesteps: int = 10_000,
    horizon_timesteps: int = 64,
    epochs_per_trajectory: int = 3,
    clipping_epsilon: float = 0.2,
    discount_factor: float = 0.99,
    gae_factor: float = 0.95,
    value_fn_factor: float = 1.0,
    entropy_fn_factor: float = 0.0,
    epsilon: float = 1e-8,
    seed: int | None = None,
) -> PyTree[Float[Array, "..."], "A"]:
    seed = secrets.randbits(30) if seed is None else seed
    obs, info = env.reset(seed=seed)
    rng = jax.random.key(seed)

    pbar = tqdm.trange(total_timesteps, desc="timesteps", leave=True)
    global_step = 0
    optstep = 0
    while global_step < total_timesteps:
        steps_taken, trajectory, obs = generate_trajectory(
            agent_fn=agent_fn,
            params=params,
            env=env,
            state=(obs, 0.0, False, False, info),
            horizon_timesteps=horizon_timesteps,
            seed=seed + global_step,
            epsilon=epsilon,
        )

        metrics: dict[str, Float[Array, ""]] = NotImplemented
        for epoch in range(epochs_per_trajectory):
            key, rng = jax.random.split(rng, 2)
            (_, logdict), grads = episode_objective_fn(
                agent_fn,
                params,
                trajectory=trajectory,
                rng=key,
                clipping_bound=clipping_epsilon,
                discount_factor=discount_factor,
                gae_factor=gae_factor,
                value_fn_factor=value_fn_factor,
                entropy_fn_factor=entropy_fn_factor,
            )

            params, optstate = optimizer_fn(
                params=params,
                grads=grads,
                optstate=optstate,
                step=optstep,
                lr=lr_schedule_fn(optstep),
            )

            if epoch == 0:
                metrics = logdict

            else:
                for key, val in logdict.items():
                    if jnp.any(jnp.isnan(val)):
                        print(f"DEBUG: {metrics=}  {logdict=}")
                        raise ValueError
                    metrics[key] += (val - metrics[key]) / (epoch + 1)

            optstep += 1

        if logger is not None:
            for key, val in metrics.items():
                logger.add_scalar(key, jnp.array(val).item(), global_step)

        global_step += steps_taken
        pbar.update(steps_taken)

        if steps_taken < horizon_timesteps:
            obs, info = env.reset(seed=seed + global_step)

    pbar.close()
