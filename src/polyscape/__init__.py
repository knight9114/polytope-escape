from functools import partial
from gymnasium.envs.registration import register
from polyscape.environment.polyscape import PolytopeEscape

register(
    id="PolytopeEscapeVeryEasy-v0",
    entry_point=partial(
        PolytopeEscape,
        n_axes=3,
        seed=42,
    ),
    nondeterministic=False,
    max_episode_steps=128,
    disable_env_checker=True,
)

register(
    id="PolytopeEscapeEasy-v0",
    entry_point=partial(
        PolytopeEscape,
        n_axes=10,
        n_polytope_distributions=3,
        seed=42,
    ),
    nondeterministic=False,
    max_episode_steps=128,
    disable_env_checker=True,
)

register(
    id="PolytopeEscapeMedium-v0",
    entry_point=partial(
        PolytopeEscape,
        n_axes=10,
        n_polytope_distributions=3,
        use_stochastic_step_size=True,
        stochastic_step_size_scale=0.5,
        seed=42,
    ),
    nondeterministic=True,
    max_episode_steps=128,
    disable_env_checker=True,
)
