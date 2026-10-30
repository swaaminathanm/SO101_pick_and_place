"""
World Interface module for SO-101 cuMotion motion planning.
Manages obstacle discovery via SceneQuery, collision representations with ObstacleStrategy,
and GPU-accelerated synchronization using CumotionWorldInterface and WorldBinding.
"""

from typing import List, Optional, Tuple, Sequence
import numpy as np

from isaacsim.core.experimental.objects import Mesh, Cone, Cylinder
from isaacsim.robot_motion.cumotion import CumotionWorldInterface
from isaacsim.robot_motion.cumotion.impl.kit_cumotion_debug_visualizer import KitCumotionDebugVisualizer
import isaacsim.robot_motion.experimental.motion_generation as mg

class SO101WorldInterface:
    """
    Manages the cuMotion world state, obstacle tracking, and coordinate transformations
    between Isaac Sim's USD stage and cuMotion GPU collision models.
    """

    def __init__(
        self,
        robot_prim_path: str = "/World/so101",
        exclude_prim_paths: Optional[List[str]] = None,
        search_box_min: Optional[Sequence[float]] = None,
        search_box_max: Optional[Sequence[float]] = None,
        safety_margin: float = 0.01,
    ):
        """
        Args:
            robot_prim_path: USD path of the robot articulation.
            exclude_prim_paths: List of USD prim paths to exclude from obstacles (e.g. target Lego bricks).
            search_box_min: [X, Y, Z] relative minimum bounds around robot base.
            search_box_max: [X, Y, Z] relative maximum bounds around robot base.
            safety_margin: Collision buffer padding (in meters) around obstacles.
        """
        self._robot_prim_path = robot_prim_path
        self._exclude_prim_paths = exclude_prim_paths or [robot_prim_path]

        self._search_box_min = search_box_min if search_box_min is not None else [-10.0, -10.0, -10.0]
        self._search_box_max = search_box_max if search_box_max is not None else [10.0, 10.0, 10.0]
        self._safety_margin = safety_margin

        self._world_interface: Optional[CumotionWorldInterface] = None
        self._world_binding: Optional[mg.WorldBinding] = None
        self._tracked_objects: List[str] = []

    def initialize(self, robot_pos, robot_ori) -> None:
        """
        Discovers scene obstacles with SceneQuery, configures obstacle strategies,
        and initializes WorldBinding and CumotionWorldInterface.

        Args:
            robot_pos: Robot root position (tensor or array-like [X, Y, Z]).
            robot_ori: Robot root orientation (quaternion [w, x, y, z]).
        """

        # Query scene obstacles within AABB
        self._tracked_objects = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_pos.numpy()[0],
            search_box_minimum=list(self._search_box_min),
            search_box_maximum=list(self._search_box_max),
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=self._exclude_prim_paths,
        )

        print(f"[SO101WorldInterface] Discovered {len(self._tracked_objects)} obstacle prims:")
        for obj in self._tracked_objects:
            print(f"  - {obj}")

        # Configure ObstacleStrategy (convert complex meshes to Oriented Bounding Boxes)
        obstacle_strategy = mg.ObstacleStrategy()
        for prim_type in (Mesh, Cone, Cylinder):
            obstacle_strategy.set_default_configuration(
                prim_type, mg.ObstacleConfiguration("obb", self._safety_margin)
            )

        # Instantiate CumotionWorldInterface and bind to tracked prims
        self._world_interface = CumotionWorldInterface(debug_visualizer=KitCumotionDebugVisualizer())
        self._world_binding = mg.WorldBinding(
            world_interface=self._world_interface,
            obstacle_strategy=obstacle_strategy,
            tracked_prims=self._tracked_objects,
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        self._world_binding.initialize()

        # Synchronize base transforms
        self.update_robot_transform_and_sync(robot_pos, robot_ori)
        print("[SO101WorldInterface] cuMotion WorldBinding initialized and synchronized.")

    def sync(self) -> None:
        """Synchronizes transforms of tracked obstacles with cuMotion GPU memory."""
        if self._world_binding is not None:
            self._world_binding.synchronize_transforms()

    def update_robot_transform_and_sync(self, robot_pos, robot_ori) -> None:
        """Updates robot base position if the robot moves in the world."""
        if self._world_interface is not None:
            self._world_interface.update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
        self.sync()

    @property
    def world_interface(self) -> Optional[CumotionWorldInterface]:
        """Returns the underlying CumotionWorldInterface for planners and controllers."""
        return self._world_interface

    @property
    def tracked_objects(self) -> List[str]:
        """Returns list of currently tracked obstacle prim paths."""
        return self._tracked_objects
