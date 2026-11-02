"""
Environments package for SO-101 pick-and-place tasks.
Uses lazy imports for environments to allow importing constants before SimulationApp initialization.
"""

from envs.constants import TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH

__all__ = [
    "SO101EnvProtocol",
    "LegoBrickPickPlaceEnv",
    "RandomizedSO101Env",
    "SimpleSO101Env",
    "TABLE_HEIGHT",
    "TABLE_LENGTH",
    "TABLE_WIDTH",
]


def __getattr__(name: str):
    if name in ("LegoBrickPickPlaceEnv", "RandomizedSO101Env"):
        from envs.randomized_env import LegoBrickPickPlaceEnv
        return LegoBrickPickPlaceEnv
    if name == "SimpleSO101Env":
        from envs.simple_pick_place_env import SimpleSO101Env
        return SimpleSO101Env
    if name == "SO101EnvProtocol":
        from envs.env_protocol import SO101EnvProtocol
        return SO101EnvProtocol
    raise AttributeError(f"module 'envs' has no attribute '{name}'")
