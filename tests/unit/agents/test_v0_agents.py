"""Tests for Agent V0"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# External Imports
import gymnasium as gym
import pytest
import jax
from jaxtyping import PRNGKeyArray

# Internal Imports
from polyscape.agents.v0.mlp import actor_critic_agent_init


# -------------------------------------------------------------------------
#   Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def rng() -> PRNGKeyArray:
    return jax.random.key(42)


@pytest.fixture
def env() -> gym.Env:
    return gym.make("PolytopeEscapeVeryEasy-v0")


# -------------------------------------------------------------------------
#   MLP Tests
# -------------------------------------------------------------------------


class TestMlpAgents:
    d_hidden_layers: int = 64
    n_hidden_layers: int = 2

    def test_actor_critic_init(self, rng, env):
        (pstack, ppi, pq), _ = actor_critic_agent_init(
            rng,
            env.observation_space.shape,
            env.action_space.n,
            d_hidden_layers=self.d_hidden_layers,
            n_hidden_layers=self.n_hidden_layers,
        )

        assert len(pstack) == 1 + self.n_hidden_layers
        assert len(pstack[0]) == 2
        assert ppi[0].shape == (self.d_hidden_layers, env.action_space.n)
        assert ppi[1].shape == (env.action_space.n,)
        assert pq[0].shape == (self.d_hidden_layers, 1)
        assert pq[1].shape == (1,)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_actor_critic_forward(self, rng, env):
        params, fn = actor_critic_agent_init(
            rng,
            env.observation_space.shape,
            env.action_space.n,
            d_hidden_layers=self.d_hidden_layers,
            n_hidden_layers=self.n_hidden_layers,
        )

        obs, _ = env.reset()
        pi, q = fn(params, obs)
        assert pi.shape == (env.action_space.n,)
        assert q.shape == (1,)

        obs, _ = env.reset()
        pi, q = fn(params, obs, rng=rng)
        assert pi.shape == (env.action_space.n,)
        assert q.shape == (1,)
