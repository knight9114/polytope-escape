"""Polytope Escape Environment"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = ["PolytopeEscape"]


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from typing import Any, Literal, Self, TypeAlias

# External Imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

# Internal Imports
from polyscape.typing import ObsType, ActType


# -------------------------------------------------------------------------
#   Types
# -------------------------------------------------------------------------

RewardChoices: TypeAlias = Literal[
    "nonzero",
    "sparse",
    "optimal-aware",
    "scaled-optimal-aware",
    "inverse-scaled-optimal-aware",
]

# -------------------------------------------------------------------------
#   Environment
# -------------------------------------------------------------------------


class PolytopeEscape(gym.Env[ObsType, ActType]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        *,
        n_axes: int = 10,
        n_polytope_distributions: int = 1,
        t_deadline: int = 128,
        reward_method: RewardChoices = "nonzero",
        failure_reward: float = -100.0,
        success_reward: float = +100.0,
        step_reward: float = 1.0,
        min_distribution_concentration: int = 0,
        max_distribution_concentration: int = 10,
        min_escape_distance: float = 10.0,
        max_escape_distance: float = 64.0,
        step_size: float = 1.0,
        use_stochastic_step_size: bool = False,
        stochastic_step_size_scale: float = 1.0,
        distribution_concentrations: Int[Array, "n_dists n_axes"] | None = None,
        distribution_distribution: Float[Array, "n_dists"] | None = None,
        render_mode: str | None = None,
        dtype: DTypeLike = jnp.float32,
        **kwargs,
    ):
        assert render_mode is None
        super().__init__()
        self.observation_space = spaces.Box(
            low=0,
            high=max_escape_distance,
            shape=(n_axes,),
            dtype=jnp.float32,
        )
        self.action_space = spaces.Discrete(n=n_axes)  # type: ignore

        self._n = n_axes
        self._deadline: int = t_deadline
        self._dtype = dtype
        self._step_size: float = step_size
        self._stochastic: bool = use_stochastic_step_size
        self._step_size_scale: float = stochastic_step_size_scale
        self._min_dist: float = min_escape_distance
        self._max_dist: float = max_escape_distance
        self._reward_method: str = reward_method
        self._fail_reward: float = failure_reward
        self._step_reward: float = step_reward
        self._win_reward: float = success_reward
        self._p_alpha = distribution_distribution or make_p_alphas(
            rng=self.get_jax_rng_key(),
            n_alphas=n_polytope_distributions,
        )
        self._alphas = distribution_concentrations or make_alphas(
            rng=self.get_jax_rng_key(),
            n_alphas=n_polytope_distributions,
            n_axes=n_axes,
            min_val=min_distribution_concentration,
            max_val=max_distribution_concentration,
        )

        self._t: int = NotImplemented
        self._goal: Float[Array, "n_axes"] = NotImplemented
        self._state: ObsType = NotImplemented

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Array]]:
        super().reset(seed=seed, options=options)
        rng = self.get_jax_rng_key()
        ki, kd = jax.random.split(rng, 2)

        idx = jax.random.choice(ki, self._p_alpha)
        alphas = self._alphas[idx, :]
        dist = jax.random.dirichlet(kd, alphas, dtype=self._dtype)
        goal = (self._max_dist - self._min_dist) * (1 - dist) + self._min_dist
        state = (jnp.zeros_like(goal),)

        self._t = 0
        self._goal = goal
        self._state = state

        return state, {
            "alphas": alphas,
            "distribution": dist,
            "goal": goal,
        }

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        self._t += 1
        step = self.get_step_size()
        delta = jax.nn.one_hot(action, self._n, dtype=self._dtype) * step
        self._state = self._state + delta

        done, truncate = self.get_is_finished()
        reward = self.get_reward(action, done, truncate)
        info = {
            "step": self._t,
            "distances": self._goal - self._state,
            "optimal-move": jnp.argmin(self._goal - self._state),
        }

        return (self._state,), reward, done, truncate, info

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def get_jax_rng_key(self) -> PRNGKeyArray:
        return jax.random.key(self.np_random.integers(np.iinfo("int32").max))

    def get_step_size(self) -> float:
        if not self._stochastic:
            return self._step_size

        rng = self.get_jax_rng_key()
        base = jax.random.gamma(rng, self._step_size, dtype=self._dtype).item()
        return base * self._step_size_scale

    def get_reward(self, action: ActType, done: bool, trunc: bool) -> float:
        if done:
            return self._win_reward

        if trunc:
            return self._fail_reward

        was_optimal = action == jnp.argmin(self._goal - self._state)
        match self._reward_method:
            case "nonzero":
                return -1 * self._step_reward

            case "sparse":
                return 0.0

            case "optimal-aware":
                return self._step_reward * (1 if was_optimal else -1)

            case "scaled-optimal-aware":
                r = self._step_reward * (1 if was_optimal else -1)
                return r * (self._t / self._deadline)

            case "inverse-scaled-optimal-aware":
                r = self._step_reward * (1 if was_optimal else -1)
                return r * ((self._deadline - self._t) / self._deadline)

            case _:
                raise ValueError(f"invalid `reward_method={self._reward_method}`")

    def get_is_finished(self) -> tuple[bool, bool]:
        done, trunc = False, False
        if jnp.any(self._state >= self._goal):
            done = True

        if not done and self._t > self._deadline:
            trunc = True

        return done, trunc

    @classmethod
    def from_distribution_concentrations(
        cls,
        alphas: Int[Array, "n_dists n_axes"],
        p: Float[Array, "n_dists"] | None = None,
    ) -> Self:
        return cls(distribution_concentrations=alphas, distribution_distribution=p)


# -------------------------------------------------------------------------
#   Utilities
# -------------------------------------------------------------------------


def make_alphas(
    rng: PRNGKeyArray,
    n_alphas: int,
    n_axes: int,
    min_val: int,
    max_val: int,
) -> Int[Array, "n_dist n_axes"]:
    assert max_val > min_val and max_val > 0
    output: list[Int[Array, "n_dist"]] = []
    while len(output) < n_alphas:
        k, rng = jax.random.split(rng, 2)
        alphas = jax.random.randint(k, [n_axes], min_val, max_val)
        if alphas.sum() > 0:
            output.append(alphas)

    return jnp.array(output)


def make_p_alphas(rng: PRNGKeyArray, n_alphas: int) -> Float[Array, "n_dist"]:
    return jax.random.dirichlet(rng, jnp.ones((n_alphas,)))
