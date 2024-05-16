"""Polytope Escape Types"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = []


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
    ) -> tuple[Float[Array, "n_axes"], Float[Array, ""]]:
        pass


class ValueNetworkForwardFn(Protocol):
    def __call__(
        self,
        params: PyTree[Float[Array, "..."], "A"],
        obs: PyTree[Float[Array, "..."], "E"],
        *,
        rng: PRNGKeyArray | None = None,
    ) -> Float[Array, "n_axes"]:
        pass
