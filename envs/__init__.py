"""
Environments package for SO-101 pick-and-place tasks.
Uses lazy imports for environments to allow importing constants before SimulationApp initialization.
"""

from envs.constants import (
    WORKSPACE_HEIGHT,
    WORKSPACE_LENGTH,
    WORKSPACE_WIDTH,
    MAT_HEIGHT,
    WORKSPACE_BASE_HEIGHT,
)

__all__ = [
    "LegoBrickPickPlaceEnv",
    "WORKSPACE_HEIGHT",
    "WORKSPACE_LENGTH",
    "WORKSPACE_WIDTH",
    "MAT_HEIGHT",
    "WORKSPACE_BASE_HEIGHT",
]


def __getattr__(name: str):
    if name == "LegoBrickPickPlaceEnv":
        from envs.randomized_env import LegoBrickPickPlaceEnv
        return LegoBrickPickPlaceEnv
    raise AttributeError(f"module 'envs' has no attribute '{name}'")
