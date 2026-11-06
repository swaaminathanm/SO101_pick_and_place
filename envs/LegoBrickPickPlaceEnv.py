import os
import random
from typing import Any, List, Optional
import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane, DomeLight, Cube
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim, GeomPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep
from pxr import Usd, UsdPhysics, UsdGeom, UsdShade, Sdf, Gf
from cameras import SO101CameraRig
from randomisation import LegoBrickRandomisation, BinRandomisation
from envs.constants import (
    ROBOT_PRIM_PATH,
    ROBOT_ARTICULATION_PATH,
    WORKSPACE_BASE_PRIM_PATH,
    WORKSPACE_MAT_PRIM_PATH,
    WALL_BACK_PRIM_PATH,
    WALL_RIGHT_PRIM_PATH,
    BIN_PRIM_PATH,
    RED_BRICK_PRIM_PATH,
    GROUND_PLANE_PRIM_PATH,
    DOME_LIGHT_PRIM_PATH,
    WORKSPACE_HEIGHT,
    WORKSPACE_LENGTH,
    WORKSPACE_WIDTH,
    MAT_HEIGHT,
    WORKSPACE_BASE_HEIGHT,
    WALL_FRONT_PRIM_PATH
)

# ==================== ASSET & WORKSPACE CONFIGURATION ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Path to assembled SO-101 USD
ROBOT_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "so101_new_calib", "so101_new_calib.usda")

# SimReady D00620 black silicone work mat asset
WORKSPACE_MAT_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "D00620", "simready_usd", "D00620.usd")

# Lego brick assets
LEGO_RED_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "lego_brick_red.usd")
LEGO_HALF_HEIGHT: float = 0.0142      # Half-height at scale 0.25 (keeps bottom resting on mat)
LEGO_RED_INIT_POS = [0.18, -0.10, WORKSPACE_HEIGHT + LEGO_HALF_HEIGHT]

BIN_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "D00935", "simready_usd", "D00935.usd")
BIN_SCALE = 0.4

LEGO_SCALE = [0.25, 0.25, 0.25]
# =========================================================================


class LegoBrickPickPlaceEnv:
    """
    Randomized pick-and-place environment for the SO-101 manipulator
    and a single red Lego brick with domain randomization (brick pose, bin pose, lighting).
    """
    WORKSPACE_HEIGHT: float = WORKSPACE_HEIGHT
    WORKSPACE_LENGTH: float = WORKSPACE_LENGTH
    WORKSPACE_WIDTH: float = WORKSPACE_WIDTH

    def __init__(
        self,
        simulation_app=None,
        world_interface=None,
        enable_cameras: bool = True,
        camera_resolution: tuple[int, int] = (1280, 720),
        randomizers: list[Any] | None = None,
    ):
        self.simulation_app = simulation_app
        self.world_interface = world_interface
        self.enable_cameras = enable_cameras
        self.camera_resolution = camera_resolution
        self.randomizers: list[Any] = list(randomizers) if randomizers is not None else []

        self._camera_rig: SO101CameraRig | None = None
        if self.enable_cameras:
            self._camera_rig = SO101CameraRig(resolution=self.camera_resolution)

        # Prim paths bound from constants
        self._ROBOT_PRIM_PATH = ROBOT_PRIM_PATH
        self._ARTICULATION_PATH = ROBOT_ARTICULATION_PATH
        self._WORKSPACE_BASE_PRIM_PATH = WORKSPACE_BASE_PRIM_PATH
        self._WORKSPACE_MAT_PRIM_PATH = WORKSPACE_MAT_PRIM_PATH
        self._WALL_BACK_PRIM_PATH = WALL_BACK_PRIM_PATH
        self._WALL_RIGHT_PRIM_PATH = WALL_RIGHT_PRIM_PATH
        self._BIN_PRIM_PATH = BIN_PRIM_PATH
        self._RED_BRICK_PATH = RED_BRICK_PRIM_PATH

        self._articulation: Articulation | None = None
        self.robot_xform: XformPrim | None = None
        
        self._lego_prims: list[RigidPrim] = []
        self.workspace_base_obj: Cube | None = None
        self.workspace_mat_obj: XformPrim | None = None
        self.wall_back_obj: Cube | None = None
        self.wall_right_obj: Cube | None = None
        self.wall_front_obj: Cube | None = None
        self.bin_obj = None
        self.bin_xform: XformPrim | None = None
        self._episode_id: int = 0

    def add_randomizer(self, randomizer: Any) -> None:
        """Appends a replicator domain randomizer to the environment's randomizer list."""
        self.randomizers.append(randomizer)

    @property
    def randomizer(self) -> Any | None:
        """Returns the first randomizer if present for backwards compatibility."""
        return self.randomizers[0] if self.randomizers else None

    @randomizer.setter
    def randomizer(self, value: Any) -> None:
        if value is None:
            self.randomizers = []
        elif isinstance(value, list):
            self.randomizers = value
        else:
            self.randomizers = [value]

    @property
    def camera_rig(self) -> SO101CameraRig | None:
        """Handle to the camera rig."""
        return self._camera_rig

    @property
    def lego_prims(self) -> list[RigidPrim]:
        """List of target objects in the scene."""
        return self._lego_prims

    @lego_prims.setter
    def lego_prims(self, value: list[RigidPrim]) -> None:
        self._lego_prims = value

    @property
    def episode_id(self) -> int:
        """Returns current episode ID."""
        return self._episode_id

    @property
    def articulation(self) -> Articulation | None:
        """Handle to the robot Articulation."""
        return self._articulation

    @property
    def bin_prim(self) -> RigidPrim | None:
        """Handle to the bin RigidPrim to query world poses."""
        return self.bin_obj

    @bin_prim.setter
    def bin_prim(self, value: RigidPrim | None) -> None:
        self.bin_obj = value

    @property
    def half_width(self) -> float:
        """Half-width of the Lego brick along grasp axis."""
        return 0.0194

    def update(self):
        """Helper to step Isaac Sim application updates."""
        self.simulation_app.update()

    def spawn_lego_brick(self, prim_path: str, usd_path: str, initial_pos: list[float]):
        """Spawns and registers a physics-enabled Lego brick directly at prim_path."""
        stage_utils.add_reference_to_stage(
            usd_path=usd_path,
            path=prim_path
        )
        self.update()
        scale = LEGO_SCALE
        rigid = RigidPrim(
            paths=prim_path,
            scales=[scale],
            positions=[initial_pos],
            reset_xform_op_properties=True,
            masses=0.02
        )
        self.lego_prims.append(rigid)
        return rigid

    def spawn_robot(
        self,
        prim_path: str = ROBOT_PRIM_PATH,
        usd_path: str = ROBOT_USD_PATH,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> tuple[Articulation, XformPrim]:
        """Spawns the SO-101 robot, configures PhysX, and returns (articulation, robot_xform)."""
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"Robot USD not found at: {usd_path}")
        
        if position is None:
            position = [0.0, 0.0, 0.0]
        if orientation is None:
            orientation = [1.0, 0.0, 0.0, 0.0]

        stage_utils.add_reference_to_stage(
            usd_path=usd_path,
            path=prim_path
        )
        
        robot_xform = XformPrim(prim_path, reset_xform_op_properties=True)
        robot_xform.set_world_poses(positions=[position], orientations=[orientation])
        self.update()
        
        # Set Physics variant to PhysX
        stage = stage_utils.get_current_stage()
        robot_prim = stage.GetPrimAtPath(prim_path)
        robot_prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        
        # Fix root_joint relationship so base is locked to the robot root
        root_joint = stage.GetPrimAtPath(f"{prim_path}/Physics/root_joint")
        if root_joint.IsValid():
            root_joint.GetRelationship("physics:body0").SetTargets([prim_path])
            
        # Initialize Articulation handle
        articulation = Articulation(self._ARTICULATION_PATH)
        self.update()
        
        return articulation, robot_xform

    def spawn_workspace(
        self,
        base_prim_path: str = WORKSPACE_BASE_PRIM_PATH,
        mat_prim_path: str = WORKSPACE_MAT_PRIM_PATH,
        mat_usd_path: str = WORKSPACE_MAT_USD_PATH,
        base_color: str = "#F0F0F0",
    ) -> tuple[Cube, XformPrim]:
        """
        Spawns the composite workspace:
          1. Floor white base cube: Sits directly on the floor from Z=0 to Z=WORKSPACE_BASE_HEIGHT.
          2. Black silicone work mat (D00620.usd): Sits on top of the base cube from
             Z=WORKSPACE_BASE_HEIGHT to Z=WORKSPACE_HEIGHT.
        
        Ensures: MAT_HEIGHT + WORKSPACE_BASE_HEIGHT == WORKSPACE_HEIGHT
        """
        # Floor White Base Cube
        base_pos = [WORKSPACE_LENGTH / 2.0, 0.0, WORKSPACE_BASE_HEIGHT / 2.0]
        base_scale = [
            0.3,
            0.3,
            0.02,
        ]

        workspace_base = Cube(
            paths=base_prim_path,
            positions=[base_pos],
            scales=[base_scale],
            colors=base_color,
        )
        GeomPrim(paths=workspace_base.paths, apply_collision_apis=True)

        # Black Silicone Work Mat (D00620.usd)
        mat_offset_x = 0.05
        mat_pos = [WORKSPACE_LENGTH / 2.0 + mat_offset_x, 0.0, WORKSPACE_BASE_HEIGHT]
        mat_ori = [0.7071068, 0.0, 0.0, 0.7071068]

        mat_scale = [1.2, 0.8, 1.2]

        stage_utils.add_reference_to_stage(
            usd_path=mat_usd_path,
            path=mat_prim_path,
        )
        self.update()

        # Remove dynamic RigidBodyAPI from child prims so the mat remains static
        stage = stage_utils.get_current_stage()
        mat_prim = stage.GetPrimAtPath(mat_prim_path)
        for child in mat_prim.GetChildren():
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)

        workspace_mat = XformPrim(mat_prim_path, reset_xform_op_properties=True)
        workspace_mat.set_world_poses(positions=[mat_pos], orientations=[mat_ori])
        workspace_mat.set_local_scales(scales=[mat_scale])
        self.update()

        # Wall back and right
        back_pos = [WORKSPACE_LENGTH / 2.0, 0.3, 0.75]
        back_scale = [0.5, 0.02, 0.75]
        wall_back = Cube(
            paths=WALL_BACK_PRIM_PATH,
            positions=[back_pos],
            scales=[back_scale],
            colors=base_color,
        )

        right_pos = [0.5, 0.0, 0.75]
        right_scale = [0.02, 0.5, 0.75]
        wall_right = Cube(
            paths=WALL_RIGHT_PRIM_PATH,
            positions=[right_pos],
            scales=[right_scale],
            colors=base_color,
        )

        front_pos = [WORKSPACE_LENGTH / 2.0, -0.3, 0.75]
        front_scale = [0.5, 0.02, 0.75]
        wall_front = Cube(
            paths=WALL_FRONT_PRIM_PATH,
            positions=[front_pos],
            scales=[front_scale],
            colors=base_color,
        )

        self.update()

        return workspace_base, workspace_mat, wall_back, wall_right, wall_front

    def spawn_bin(
        self,
        prim_path: str = BIN_PRIM_PATH,
        usd_path: str = BIN_USD_PATH,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        scale: list[float] | None = None,
    ) -> RigidPrim:
        """Spawns the bin asset onto the stage as a collision-aware kinematic rigid body."""
        if position is None:
            position = [0.25, 0.12, WORKSPACE_HEIGHT]
        if orientation is None:
            orientation = [1.0, 0.0, 0.0, 0.0]
        if scale is None:
            scale = [BIN_SCALE, BIN_SCALE, BIN_SCALE]

        stage_utils.add_reference_to_stage(
            usd_path=usd_path,
            path=prim_path
        )
        self.update()
        
        # Remove duplicate RigidBodyAPI from child prims to prevent conflicts
        stage = stage_utils.get_current_stage()
        bin_prim = stage.GetPrimAtPath(prim_path)
        
        for child in bin_prim.GetChildren():
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                child_rb = UsdPhysics.RigidBodyAPI(child)
                if child_rb:
                    child_rb.CreateRigidBodyEnabledAttr(False)
        
        # Ensure root Xform does NOT have CollisionAPI (cuMotion requires supported shape types like Mesh)
        if bin_prim.HasAPI(UsdPhysics.CollisionAPI):
            bin_prim.RemoveAPI(UsdPhysics.CollisionAPI)
        if bin_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            bin_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        
        # Apply collision API to child mesh prims so PhysX and cuMotion recognize obstacles
        for p in stage.Traverse():
            if str(p.GetPath()).startswith(prim_path) and p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(p)
                mesh_api.CreateApproximationAttr().Set("convexDecomposition")

        # Create and bind yellow plastic material to bin mesh
        mat_path = Sdf.Path(f"{prim_path}/Looks/BinMaterial")
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.965, 0.706, 0.180))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        mesh_prim = stage.GetPrimAtPath(f"{prim_path}/part_00/mesh_node")
        if mesh_prim.IsValid():
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

        # Save an XformPrim handle to directly update the USD stage transforms when the bin moves
        self.bin_xform = XformPrim(prim_path, reset_xform_op_properties=True)
        self.bin_xform.set_world_poses(positions=[position], orientations=[orientation])
        self.bin_xform.set_local_scales(scales=[scale])

        bin_rigid = RigidPrim(
            paths=prim_path,
            scales=[scale],
            positions=[position],
            orientations=[orientation],
            reset_xform_op_properties=True
        )
        self.update()
        return bin_rigid

    def setup_scene(self):
        """Creates the stage, lights, workspace, walls, bin, robot, and Lego bricks."""
        stage_utils.create_new_stage()
        
        GroundPlane(GROUND_PLANE_PRIM_PATH)   
        
        self.light = DomeLight(DOME_LIGHT_PRIM_PATH)
        self.light.set_intensities(1000)

        # Spawn Workspace (Base + Mat), Walls, and Bin using constants
        self.workspace_base_obj, self.workspace_mat_obj, self.wall_back_obj, self.wall_right_obj, self.wall_front_obj = self.spawn_workspace()
        self.bin_obj = self.spawn_bin(prim_path=self._BIN_PRIM_PATH)
        
        # Spawn SO-101 Robot Articulation
        self._articulation, self.robot_xform = self.spawn_robot(
            prim_path=self._ROBOT_PRIM_PATH,
            usd_path=ROBOT_USD_PATH,
            position=[0.05, 0.0, WORKSPACE_BASE_HEIGHT],
            orientation=[1.0, 0.0, 0.0, 0.0]
        )
        
        # Spawn Red Lego Brick only (single brick)
        self.spawn_lego_brick(
            prim_path=self._RED_BRICK_PATH,
            usd_path=LEGO_RED_USD_PATH,
            initial_pos=LEGO_RED_INIT_POS
        )
        
        # Setup Camera Rig if enabled
        if self._camera_rig is not None:
            self._camera_rig.setup_cameras(
                robot_prim_path=self._ROBOT_PRIM_PATH,
                robot_articulation_path=self._ARTICULATION_PATH,
            )

        # Position viewport camera view
        set_camera_view(
            eye=[-0.35, 0.0, 0.40], 
            target=[0.22, 0.0, 0.08], 
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def initialize_after_play(self):
        """Initializes joint positions, world interface, and cameras after simulation play."""
        dof_count = self._articulation.num_dofs
        init_pos = [0.0] * dof_count
        self._articulation.set_dof_positions(init_pos)
        self._articulation.set_dof_position_targets(init_pos)
        self._articulation.set_dof_velocities([0.0] * dof_count)

        # Initialize injected WorldInterface with robot base pose
        if self.world_interface is not None:
            robot_pos, robot_ori = self._articulation.get_world_poses()
            self.world_interface.initialize(robot_pos, robot_ori)

        # Initialize Camera Rig annotators
        if self._camera_rig is not None:
            self._camera_rig.initialize()

    def get_observations(self) -> dict:
        """
        Retrieves synchronized multimodal observations:
          - 'joint_positions': np.ndarray of current robot joint angles
          - 'joint_velocities': np.ndarray of current robot joint velocities
          - 'overhead_rgb', 'overhead_depth'
          - 'wrist_rgb', 'wrist_depth'
        """
        obs = {}
        if self._articulation is not None:
            d_pos = self._articulation.get_dof_positions()
            d_vel = self._articulation.get_dof_velocities()
            obs["joint_positions"] = d_pos.numpy() if hasattr(d_pos, "numpy") else np.asarray(d_pos)
            obs["joint_velocities"] = d_vel.numpy() if hasattr(d_vel, "numpy") else np.asarray(d_vel)
        if self._camera_rig is not None:
            obs.update(self._camera_rig.get_observations())
        return obs

    def reset_domain(self, episode_id: int | None = None):
        """Domain randomization applied at each episode reset for single red brick."""
        if episode_id is not None:
            self._episode_id = episode_id
        else:
            self._episode_id += 1

        print(f"\n>>> [Episode {self._episode_id}] [RESET] Randomizing Workspace & Domain (Single Red Lego Brick)...")
        
        # Step each replicator domain randomizer in the injected list (brick, bin, etc.)
        for r in self.randomizers:
            r.step()

        # Log final Bin position
        if self.bin_obj is not None:
            b_pos, b_ori = self.bin_obj.get_world_poses()
            bp = b_pos.numpy()[0] if hasattr(b_pos, "numpy") else np.asarray(b_pos)[0]
            bo = b_ori.numpy()[0] if hasattr(b_ori, "numpy") else np.asarray(b_ori)[0]
            print(f"  [Episode {self._episode_id}] [Reset] Bin Pos: [{bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f}], Ori: [{bo[0]:.4f}, {bo[1]:.4f}, {bo[2]:.4f}, {bo[3]:.4f}]")

        # Reset Robot Arm joints to home with slight noise (+- 2 degrees)
        dof_count = self._articulation.num_dofs
        noisy_pose = [random.uniform(-0.035, 0.035) for _ in range(dof_count)]
        if dof_count > 5:
            # Keep gripper fully open on reset (+100 deg = 1.7453 rad)
            noisy_pose[5] = 1.7453
        self._articulation.set_dof_positions(noisy_pose)
        self._articulation.set_dof_position_targets(noisy_pose)
        self._articulation.set_dof_velocities([0.0] * dof_count)
        
        # Randomize Lighting Intensity
        self.light.set_intensities(random.uniform(700, 1500))

        # Synchronize cuMotion world interface when obstacles change position
        if self.world_interface is not None:
            robot_pos, robot_ori = self._articulation.get_world_poses()
            self.world_interface.update_robot_transform_and_sync(robot_pos, robot_ori)

        # Print final block position
        pos, _ = self.lego_prims[0].get_world_poses()
        p = pos.numpy()[0] if hasattr(pos, "numpy") else np.asarray(pos)[0]
        print(f"  [Episode {self._episode_id}] [Reset] Lego Brick Red Pos: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")

