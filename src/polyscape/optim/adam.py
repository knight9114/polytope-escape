"""Adaptive Moments Optimizer"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = ["init", "step"]


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from functools import partial

# External Imports
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree

# Internal Imports
from polyscape.typing import OptimizerStepFn


# -------------------------------------------------------------------------
#   Exported Functions
# -------------------------------------------------------------------------


def init(
    params: PyTree[Float[Array, "..."], "A"],
    *,
    beta_1: float = 0.9,
    beta_2: float = 0.95,
    weight_decay: float = 0.0,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[
    PyTree[Float[Array, "..."], "O"],
    OptimizerStepFn,
]:
    m = tree_map(lambda p: jnp.zeros_like(p), params)
    v = tree_map(lambda p: jnp.zeros_like(p), params)

    step_fn = partial(
        step,
        beta_1=beta_1,
        beta_2=beta_2,
        weight_decay=weight_decay,
        epsilon=epsilon,
    )

    return (m, v), jax.jit(step_fn)


def step(
    params: PyTree[Float[Array, "..."], "A"],
    grads: PyTree[Float[Array, "..."], "A"],
    optstate: PyTree[Float[Array, "..."], "O"],
    step: float,
    lr: float,
    *,
    beta_1: float = 0.9,
    beta_2: float = 0.95,
    weight_decay: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[
    PyTree[Float[Array, "..."], "A"],
    PyTree[Float[Array, "..."], "O"],
]:
    m, v = optstate
    m = tree_map(lambda g, t: beta_1 * t + (1 - beta_1) * g, grads, m)
    v = tree_map(lambda g, t: beta_1 * t + (1 - beta_1) * g**2, grads, v)

    m_hat = tree_map(lambda t: t / (1 - beta_1**step), m)
    v_hat = tree_map(lambda t: t / (1 - beta_2**step), v)

    decay = tree_map(lambda p: p * weight_decay, params)
    scale = tree_map(lambda m, v: m / jnp.sqrt(v + epsilon), m_hat, v_hat)
    grad_step = tree_map(lambda s, d: s + d, scale, decay)
    update = tree_map(lambda p, g: p - lr * g, params, grad_step)

    return update, (m, v)
