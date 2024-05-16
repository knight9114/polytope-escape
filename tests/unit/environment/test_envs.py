"""Tests for Polytope Escape Environments"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
import math

# External Imports
import gymnasium as gym
import pytest


# -------------------------------------------------------------------------
#   Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def very_easy_env() -> gym.Env:
    return gym.make("PolytopeEscapeVeryEasy-v0")


@pytest.fixture
def very_easy_impossible_env() -> gym.Env:
    return gym.make("PolytopeEscapeVeryEasy-v0", t_deadline=10)


# -------------------------------------------------------------------------
#   Environment Tests
# -------------------------------------------------------------------------


class TestVeryEasyEnvironment:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_optimal_trajectory(self, very_easy_env):
        _, info = very_easy_env.reset(seed=42)
        optimal_action = info["goal"].argmin()
        optimal_steps = math.ceil(info["goal"].min())

        steps = 0
        done = False
        while not done:
            _, reward, fin, trunc, _ = very_easy_env.step(optimal_action)
            done = fin or trunc
            steps += 1

        assert steps == optimal_steps
        assert reward == 100

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_worst_trajectory(self, very_easy_env):
        _, info = very_easy_env.reset(seed=42)
        worst_action = info["goal"].argmax()
        worst_steps = math.ceil(info["goal"].max())

        steps = 0
        done = False
        while not done:
            _, reward, fin, trunc, _ = very_easy_env.step(worst_action)
            done = fin or trunc
            steps += 1

        assert steps == worst_steps
        assert reward == 100

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_impossible_trajectory(self, very_easy_impossible_env):
        _, info = very_easy_impossible_env.reset(seed=42)
        worst_action = info["goal"].argmax()
        worst_steps = math.ceil(info["goal"].max())

        steps = 0
        done = False
        while not done:
            _, reward, fin, trunc, _ = very_easy_impossible_env.step(worst_action)
            done = fin or trunc
            steps += 1

        assert steps < worst_steps
        assert reward == -100
