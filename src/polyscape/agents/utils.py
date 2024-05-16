"""Agent Utilities"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = []


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# External Imports
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# -------------------------------------------------------------------------
#   Functions
# -------------------------------------------------------------------------


def dropout_forward(
    x: Float[Array, "..."],
    *,
    rng: PRNGKeyArray | None = None,
    dropout: float = 0.0,
) -> Float[Array, "..."]:
    if rng is None or dropout == 0:
        return x

    keep = 1 - dropout
    mask = jax.random.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)
