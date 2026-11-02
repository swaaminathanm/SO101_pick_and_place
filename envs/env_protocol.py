"""
Protocol defining the interface for SO-101 simulation environments.
Both RandomizedSO101Env and SimpleSO101Env implement this protocol.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
else:
    Articulation = Any
    RigidPrim = Any


@runtime_checkable
class SO101EnvProtocol(Protocol):
    """Protocol defining the standard interface for SO-101 pick-and-place environments."""

    @property
    def episode_id(self) -> int:
        """Current episode index."""
        ...

    @property
    def articulation(self) -> Articulation | None:
        """Handle to the robot Articulation."""
        ...

    @property
    def bin_prim(self) -> RigidPrim | None:
        """Handle to the bin RigidPrim to query world poses."""
        ...

    @property
    def lego_prims(self) -> list[RigidPrim]:
        """List of target objects (e.g. Lego bricks or cubes) in the environment."""
        ...

    @property
    def brick_center_offset(self) -> np.ndarray:
        """Centroid offset [x, y, z] from the object origin."""
        ...

    @property
    def half_width(self) -> float:
        """Half-width of the target object along the grasp axis."""
        ...

    @property
    def closed_pos(self) -> float:
        """Target gripper joint angle for grasping the object."""
        ...

    @property
    def camera_rig(self) -> Any | None:
        """Handle to the camera rig if enabled."""
        ...

    def get_observations(self) -> dict[str, Any]:
        """Returns multimodal observations (proprioception + camera frames)."""
        ...

    def setup_scene(self) -> None:
        """Populates the stage with robot, table, bin, and manipulable objects."""
        ...

    def initialize_after_play(self) -> None:
        """Initializes articulation and world interface after simulation playback begins."""
        ...

    def reset_domain(self) -> None:
        """Resets the domain (e.g. poses of target objects and containers) for a new episode."""
        ...

    def update(self) -> None:
        """Steps application/viewport updates."""
        ...
