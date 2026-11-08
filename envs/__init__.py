"""
Environments package for SO-101 pick-and-place tasks.
Uses lazy imports for environments to allow importing constants before SimulationApp initialization.
"""

import sys
from types import ModuleType
from envs.constants import (
    WORKSPACE_HEIGHT,
    WORKSPACE_LENGTH,
    WORKSPACE_WIDTH,
    MAT_HEIGHT,
    WORKSPACE_BASE_HEIGHT,
    LEGO_BRICK_HALF_EXTENTS,
    LEGO_BRICK_HALF_HEIGHT,
    BIN_HALF_EXTENTS,
    BIN_HALF_LENGTH,
    BIN_HALF_WIDTH,
    BIN_HALF_HEIGHT,
)

__all__ = [
    "LegoBrickPickPlaceEnv",
    "WORKSPACE_HEIGHT",
    "WORKSPACE_LENGTH",
    "WORKSPACE_WIDTH",
    "MAT_HEIGHT",
    "WORKSPACE_BASE_HEIGHT",
    "LEGO_BRICK_HALF_EXTENTS",
    "LEGO_BRICK_HALF_HEIGHT",
    "BIN_HALF_EXTENTS",
    "BIN_HALF_LENGTH",
    "BIN_HALF_WIDTH",
    "BIN_HALF_HEIGHT",
]


class _EnvsModule(ModuleType):
    @property
    def LegoBrickPickPlaceEnv(self):
        from .LegoBrickPickPlaceEnv import LegoBrickPickPlaceEnv as _cls
        return _cls


sys.modules[__name__].__class__ = _EnvsModule
