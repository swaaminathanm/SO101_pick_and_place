"""
Simple SO-101 Pick and Place Environment extending RandomizedSO101Env.
Reuses robot, table, and bin spawning from RandomizedSO101Env.
Spawns only a single cube at a fixed pose with no randomizations or inter-object collision sensors.
"""

import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane, DomeLight, Cube
from isaacsim.core.experimental.prims import RigidPrim, GeomPrim
from isaacsim.core.utils.viewports import set_camera_view

from envs.randomized_env import (
    RandomizedSO101Env,
    ROBOT_PRIM_PATH,
    ROBOT_USD_PATH,
    TABLE_PRIM_PATH,
    BIN_PRIM_PATH,
    RED_BRICK_PRIM_PATH,
    BLUE_BRICK_PRIM_PATH,
    GROUND_PLANE_PRIM_PATH,
    DOME_LIGHT_PRIM_PATH,
    TABLE_HEIGHT,
)

# Cube configuration: 4cm x 4cm x 4cm
CUBE_SIZE = 0.04
CUBE_MASS = 0.02  # 20 grams
CUBE_INIT_POS = [0.18, -0.10, TABLE_HEIGHT + CUBE_SIZE / 2.0]  # [0.18, -0.10, 0.10]
BIN_INIT_POS = [0.25, 0.12, TABLE_HEIGHT]


class SimpleSO101Env(RandomizedSO101Env):
    """
    Deterministic pick-and-place environment extending RandomizedSO101Env.
    Spawns only 1 cube at a fixed position with no randomizations and no collision sensors.
    """

    @property
    def brick_center_offset(self) -> np.ndarray:
        """Cube origin is at its geometric center."""
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def half_width(self) -> float:
        """Half-width of the cube for stationary jaw placement."""
        return CUBE_SIZE / 2.0

    def spawn_cube(
        self,
        prim_path: str = RED_BRICK_PRIM_PATH,
        initial_pos: list[float] | None = None,
        size: float = CUBE_SIZE,
        color: str = "#CC2222",
        mass: float = CUBE_MASS,
    ) -> RigidPrim:
        """Spawns and registers a physics-enabled simple Cube."""
        if initial_pos is None:
            initial_pos = CUBE_INIT_POS

        cube = Cube(
            paths=prim_path,
            positions=[initial_pos],
            sizes=size,
            colors=color,
        )
        GeomPrim(paths=cube.paths, apply_collision_apis=True)
        rigid = RigidPrim(
            paths=prim_path,
            masses=mass,
            reset_xform_op_properties=False,
        )
        self.lego_prims.append(rigid)
        return rigid

    def setup_scene(self):
        """Creates table, bin, robot, and a single cube (no second brick, no contact sensors)."""
        stage_utils.create_new_stage()

        GroundPlane(GROUND_PLANE_PRIM_PATH)

        self.light = DomeLight(DOME_LIGHT_PRIM_PATH)
        self.light.set_intensities(1000)

        # Reuse robot, table, and bin spawning from parent RandomizedSO101Env
        self.table_obj = self.spawn_table(prim_path=self._TABLE_PRIM_PATH)
        self.bin_obj = self.spawn_bin(prim_path=self._BIN_PRIM_PATH)
        self._articulation, self.robot_xform = self.spawn_robot(
            prim_path=self._ROBOT_PRIM_PATH,
            usd_path=ROBOT_USD_PATH,
            position=[0.0, 0.0, TABLE_HEIGHT],
            orientation=[1.0, 0.0, 0.0, 0.0],
        )

        # Single red cube only
        self.spawn_cube(
            prim_path=self._RED_BRICK_PATH,
            initial_pos=CUBE_INIT_POS,
            color="#CC2222",
        )

        # Setup Camera Rig if enabled
        if self._camera_rig is not None:
            self._camera_rig.setup_cameras(
                robot_prim_path=self._ROBOT_PRIM_PATH,
                robot_articulation_path=self._ARTICULATION_PATH,
            )

        # Position camera view
        set_camera_view(
            eye=[0.8, 0.8, 0.6],
            target=[0.20, 0.0, 0.0],
            camera_prim_path="/OmniverseKit_Persp",
        )

    def initialize_after_play(self):
        """Initializes joint positions and world interface without contact sensors."""
        dof_count = self._articulation.num_dofs
        init_pos = [0.0] * dof_count
        self._articulation.set_dof_positions(init_pos)
        self._articulation.set_dof_position_targets(init_pos)
        self._articulation.set_dof_velocities([0.0] * dof_count)

        if self.world_interface is not None:
            robot_pos, robot_ori = self._articulation.get_world_poses()
            self.world_interface.initialize(robot_pos, robot_ori)

        # Initialize Camera Rig annotators
        if self._camera_rig is not None:
            self._camera_rig.initialize()

    def reset_domain(self, episode_id: int | None = None):
        """Deterministic reset for 1 cube and bin (fixed poses, no randomizations)."""
        if episode_id is not None:
            self._episode_id = episode_id
        else:
            self._episode_id += 1

        print(f"\n>>> [Episode {self._episode_id}] [RESET] Resetting deterministic environment (1 cube, fixed pose)...")

        # Reset Bin to fixed position
        bin_pos = np.array(BIN_INIT_POS)
        if self.bin_xform is not None:
            self.bin_xform.set_world_poses(
                positions=[bin_pos],
                orientations=[[1.0, 0.0, 0.0, 0.0]],
            )
        if self.bin_obj is not None:
            self.bin_obj.set_world_poses(
                positions=[bin_pos],
                orientations=[[1.0, 0.0, 0.0, 0.0]],
            )
            self.bin_obj.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )

        # Reset single Cube to fixed position
        cube_pos = np.array(CUBE_INIT_POS)
        self.lego_prims[0].set_world_poses(
            positions=[cube_pos],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )
        self.lego_prims[0].set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )

        self.update()

        # Reset Robot Arm joints to exact home (0.0 rad)
        dof_count = self._articulation.num_dofs
        home_pose = [0.0] * dof_count
        self._articulation.set_dof_positions(home_pose)
        self._articulation.set_dof_position_targets(home_pose)
        self._articulation.set_dof_velocities([0.0] * dof_count)

        # Keep lighting constant
        self.light.set_intensities(1000)

        # Synchronize cuMotion world interface
        if self.world_interface is not None:
            robot_pos, robot_ori = self._articulation.get_world_poses()
            self.world_interface.update_robot_transform_and_sync(robot_pos, robot_ori)

        # Print target cube pose
        pos, _ = self.lego_prims[0].get_world_poses()
        p = pos.numpy()[0] if hasattr(pos, "numpy") else np.asarray(pos)[0]
        print(f"  [Episode {self._episode_id}] [Reset] Fixed Red Cube Pos: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
