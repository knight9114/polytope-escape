"""Polytope Escape Types"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = [
    "ActType",
    "ObsType",
    "ActorCriticForwardFn",
    "ValueNetworkForwardFn",
    "OptimizerStepFn",
    "LearningRateScheduleFn",
]


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from typing import Protocol, TypeAlias

# External Imports
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


# -------------------------------------------------------------------------
#   Environment Types
# -------------------------------------------------------------------------

ObsType: TypeAlias = PyTree[Float[Array, "n_axes"], "E"]
ActType: TypeAlias = int


# -------------------------------------------------------------------------
#   Agent Types
# -------------------------------------------------------------------------


class ActorCriticForwardFn(Protocol):
    def __call__(
        self,
        params: PyTree[Float[Array, "..."], "A"],
        obs: PyTree[Float[Array, "..."], "E"],
        *,
        rng: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "n_axes"], Float[Array, ""]]: ...


class ValueNetworkForwardFn(Protocol):
    def __call__(
        self,
        params: PyTree[Float[Array, "..."], "A"],
        obs: PyTree[Float[Array, "..."], "E"],
        *,
        rng: PRNGKeyArray | None = None,
    ) -> Float[Array, "n_axes"]: ...


# -------------------------------------------------------------------------
#   Optimizer Types
# -------------------------------------------------------------------------


class OptimizerStepFn(Protocol):
    def __call__(
        self,
        params: PyTree[Float[Array, "..."], "A"],
        grads: PyTree[Float[Array, "..."], "A"],
        optstate: PyTree[Float[Array, "..."], "O"],
        step: float,
        lr: float,
    ) -> tuple[
        PyTree[Float[Array, "..."], "A"],
        PyTree[Float[Array, "..."], "O"],
    ]: ...


class LearningRateScheduleFn(Protocol):
    def __call__(self, step: int) -> float: ...
