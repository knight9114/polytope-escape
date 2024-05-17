"""Stochastic Gradient Descent Optimizer"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = ["init", "step"]


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# External Imports
import jax
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree

# Internal Imports
from polyscape.typing import OptimizerStepFn


# -------------------------------------------------------------------------
#   Exported Functions
# -------------------------------------------------------------------------


def init(
    params: PyTree[Float[Array, "..."], "A"],
    **kwargs,
) -> tuple[
    PyTree[Float[Array, "..."], "O"],
    OptimizerStepFn,
]:
    return (), jax.jit(step)


def step(
    params: PyTree[Float[Array, "..."], "A"],
    grads: PyTree[Float[Array, "..."], "A"],
    optstate: PyTree[Float[Array, "..."], "O"],
    step: float,
    lr: float,
) -> tuple[
    PyTree[Float[Array, "..."], "A"],
    PyTree[Float[Array, "..."], "O"],
]:
    update = tree_map(lambda p, g: p - lr * g, params, grads)
    return update, optstate
