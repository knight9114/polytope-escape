"""MLP Agents Version 0"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = ["actor_critic_agent_init", "actor_critic_agent_forward"]


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from functools import partial
import math

# External Imports
import jax
from jax.nn.initializers import Initializer, xavier_normal
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray, PyTree

# Internal Imports
from polyscape.typing import ActorCriticForwardFn
from polyscape.agents.utils import dropout_forward


# -------------------------------------------------------------------------
#   Actor-Critic Functions
# -------------------------------------------------------------------------


def actor_critic_agent_init(
    rng: PRNGKeyArray,
    observation_shape: tuple[int, ...],
    n_actions: int,
    d_hidden_layers: int = 2,
    n_hidden_layers: int = 64,
    dropout: float = 0.0,
    *,
    use_bias: bool = True,
    weight_initializer: Initializer | None = None,
    bias_initializer: Initializer | None = None,
) -> tuple[PyTree[Float[Array, "..."], "A"], ActorCriticForwardFn]:
    *kstack, kpi, kq = jax.random.split(rng, 1 + n_hidden_layers + 2)
    d_obs = math.prod(observation_shape)
    dims = (d_obs, d_hidden_layers) + (d_hidden_layers,) * n_hidden_layers
    init_layer = partial(
        linear_layer_init,
        use_bias=use_bias,
        weight_initializer=weight_initializer,
        bias_initializer=bias_initializer,
    )

    pstack = tuple(
        [
            init_layer(k, d_in, d_out)
            for k, d_in, d_out in zip(kstack, dims[:-1], dims[1:])
        ]
    )

    ppi = init_layer(kpi, d_hidden_layers, n_actions)
    pq = init_layer(kq, d_hidden_layers, 1)

    forward_fn = partial(
        actor_critic_agent_forward,
        use_bias=use_bias,
        dropout=dropout,
    )

    return (pstack, ppi, pq), forward_fn


def actor_critic_agent_forward(
    params: PyTree[Float[Array, "..."], "A"],
    obs: PyTree[Float[Array, "..."], "E"],
    *,
    use_bias: bool = True,
    rng: PRNGKeyArray | None = None,
    dropout: float = 0.0,
) -> tuple[Float[Array, "n_axes"], Float[Array, ""]]:
    pstack, ppi, pq = params
    n = len(pstack)
    keys: tuple[None, ...] | PRNGKeyArray = (
        (None,) * n if rng is None else jax.random.split(rng, n)
    )

    x = obs[-1]
    for k, ps in zip(keys, pstack):
        x = linear_layer_forward(ps, x, use_bias=use_bias, rng=k, dropout=dropout)
        x = jax.nn.silu(x)

    pi = linear_layer_forward(ppi, x, use_bias=use_bias)
    q = linear_layer_forward(pq, x, use_bias=use_bias)

    return pi, q


# -------------------------------------------------------------------------
#   Utility Functions
# -------------------------------------------------------------------------


def linear_layer_init(
    rng: PRNGKeyArray,
    d_in: int,
    d_out: int,
    *,
    use_bias: bool = True,
    weight_initializer: Initializer | None = None,
    bias_initializer: Initializer | None = None,
    dtype: DTypeLike = jnp.float32,
) -> PyTree[Float[Array, "*d_in d_out"], "L"]:
    weight_initializer = weight_initializer or xavier_normal(dtype=dtype)
    bias_initializer = bias_initializer or xavier_normal(dtype=dtype)

    kw, kb = jax.random.split(rng, 2)
    w = weight_initializer(kw, [d_in, d_out])
    if use_bias:
        b = bias_initializer(kb, [d_out, 1]).ravel()
        return (w, b)

    return (w,)


def linear_layer_forward(
    params: PyTree[Float[Array, "*d_in d_out"], "L"],
    x: Float[Array, "..."],
    *,
    use_bias: bool = True,
    rng: PRNGKeyArray | None = None,
    dropout: float = 0.0,
) -> Float[Array, "..."]:
    x = x @ params[0]
    if use_bias:
        x = x + params[1]

    return dropout_forward(x, rng=rng, dropout=dropout)
