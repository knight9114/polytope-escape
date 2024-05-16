from gymnasium.envs.registration import register

register(
    id="PolytopeEscape-v0",
    entry_point="polyscape.environment.polyscape:PolytopeEscape",
)
