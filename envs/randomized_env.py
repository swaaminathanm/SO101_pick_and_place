import os
import random
import numpy as np

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane, DomeLight, Cube
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim, GeomPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdPhysics, UsdGeom
from isaacsim.sensors.experimental.physics import Contact, ContactSensor
from envs.env_protocol import SO101EnvProtocol

# ==================== PRIM PATH CONSTANTS ====================
ROBOT_PRIM_PATH = "/World/so101"
ROBOT_ARTICULATION_PATH = "/World/so101/Geometry/base_link"

TABLE_PRIM_PATH = "/World/Table"
BIN_PRIM_PATH = "/World/Bin"
RED_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Red"
BLUE_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Blue"

GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
# ==============================================================

# ==================== ASSET & WORKSPACE CONFIGURATION ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Path to assembled SO-101 USD
ROBOT_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "so101_new_calib", "so101_new_calib.usda")
ROBOT_BASE_OFFSET_Z = 0.016

TABLE_HEIGHT = 0.08
TABLE_LENGTH = 0.40 
TABLE_WIDTH = 0.30 

# Lego brick assets
LEGO_RED_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "lego_brick_red.usd")
LEGO_BLUE_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "lego_brick_blue.usd")
if not os.path.exists(LEGO_BLUE_USD_PATH):
    LEGO_BLUE_USD_PATH = LEGO_RED_USD_PATH

LEGO_RED_INIT_POS = [0.18, -0.10, TABLE_HEIGHT + 0.001]
LEGO_BLUE_INIT_POS = [0.24, -0.18, TABLE_HEIGHT + 0.001]

BIN_USD_PATH = os.path.join(PROJECT_ROOT, "assets", "D00935", "simready_usd", "D00935.usd")
BIN_SCALE = 0.4

LEGO_SCALE = [0.25, 0.25, 0.25]
LEGO_BRICK_RADIUS_RANGE = (0.19, 0.27)
LEGO_BRICK_ANGLE_RANGE = (-0.48, -0.10)

BIN_RADIUS_RANGE = (0.24, 0.30)
BIN_ANGLE_RANGE = (0.30, 0.50)
# =========================================================================


class LegoBrickPickPlaceEnv(SO101EnvProtocol):
    """
    Randomized pick-and-place environment for the SO-101 manipulator
    and a single red Lego brick with domain randomization (brick pose, bin pose, lighting).
    """
    def __init__(self, simulation_app=None, world_interface=None):
        self.simulation_app = simulation_app
        self.world_interface = world_interface

        # Prim paths bound from constants
        self._ROBOT_PRIM_PATH = ROBOT_PRIM_PATH
        self._ARTICULATION_PATH = ROBOT_ARTICULATION_PATH
        self._TABLE_PRIM_PATH = TABLE_PRIM_PATH
        self._BIN_PRIM_PATH = BIN_PRIM_PATH
        self._RED_BRICK_PATH = RED_BRICK_PRIM_PATH

        self._articulation: Articulation | None = None
        self.robot_xform: XformPrim | None = None
        
        self._lego_prims: list[RigidPrim] = []
        self.table_obj = None
        self.bin_obj = None
        self.bin_xform: XformPrim | None = None
        self.red_contact = None
        self._episode_id: int = 0

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
    def brick_center_offset(self) -> np.ndarray:
        """Scaled Lego brick centroid offset from top-left origin (scale 0.25)."""
        return np.array([0.0197, 0.0397, 0.0145], dtype=np.float32)

    @property
    def half_width(self) -> float:
        """Half-width of the Lego brick along grasp axis."""
        return 0.0197

    @property
    def closed_pos(self) -> float:
        """Grasp target angle for Lego brick."""
        return 0.45

    def update(self):
        """Helper to step Isaac Sim application updates."""
        if self.simulation_app is not None:
            self.simulation_app.update()
        else:
            app_utils.update()

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

    def spawn_table(
        self,
        prim_path: str = TABLE_PRIM_PATH,
        position: list[float] | None = None,
        scale: list[float] | None = None,
        color: str = "#A77B51",
    ) -> Cube:
        """Spawns and configures a collision-enabled tabletop."""
        if position is None:
            position = [TABLE_LENGTH / 2, 0.0, TABLE_HEIGHT / 2]
        if scale is None:
            scale = [TABLE_LENGTH / 2, TABLE_WIDTH, TABLE_HEIGHT / 2]

        table = Cube(
            paths=prim_path,
            positions=[position],
            scales=[scale],
            colors=color
        )
        GeomPrim(paths=table.paths, apply_collision_apis=True)
        return table

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
            position = [0.25, 0.12, TABLE_HEIGHT]
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
        """Creates the stage, lights, table, bin, robot, and Lego bricks."""
        stage_utils.create_new_stage()
        
        GroundPlane(GROUND_PLANE_PRIM_PATH)   
        
        self.light = DomeLight(DOME_LIGHT_PRIM_PATH)
        self.light.set_intensities(1000)

        # Spawn Table and Bin using constants
        self.table_obj = self.spawn_table(prim_path=self._TABLE_PRIM_PATH)
        self.bin_obj = self.spawn_bin(prim_path=self._BIN_PRIM_PATH)
        
        # Spawn SO-101 Robot Articulation
        self._articulation, self.robot_xform = self.spawn_robot(
            prim_path=self._ROBOT_PRIM_PATH,
            usd_path=ROBOT_USD_PATH,
            position=[0.0, 0.0, TABLE_HEIGHT],
            orientation=[1.0, 0.0, 0.0, 0.0]
        )
        
        # Spawn Red Lego Brick only (single brick)
        self.spawn_lego_brick(
            prim_path=self._RED_BRICK_PATH,
            usd_path=LEGO_RED_USD_PATH,
            initial_pos=LEGO_RED_INIT_POS
        )
        
        # Setup Contact sensor to monitor red brick-to-bin collisions
        self.red_contact_authoring = Contact.create(
            path=f"{self._RED_BRICK_PATH}/ContactSensor",
            min_threshold=0.0,
            max_threshold=1e5,
            radius=-1.0,
        )
        self.red_contact = ContactSensor(self.red_contact_authoring)
        
        # Position camera view
        set_camera_view(
            eye=[0.8, 0.8, 0.6], 
            target=[0.20, 0.0, 0.0], 
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def initialize_after_play(self):
        """Initializes joint positions and contact sensors after simulation play."""
        dof_count = self._articulation.num_dofs
        init_pos = [0.0] * dof_count
        self._articulation.set_dof_positions(init_pos)
        self._articulation.set_dof_position_targets(init_pos)
        self._articulation.set_dof_velocities([0.0] * dof_count)
        
        self.red_contact.add_raw_contact_data_to_frame()

        # Initialize injected WorldInterface with robot base pose
        if self.world_interface is not None:
            robot_pos, robot_ori = self._articulation.get_world_poses()
            self.world_interface.initialize(robot_pos, robot_ori)
        
    def _check_collision(self, sensor, target_keywords):
        data = sensor.get_data()
        in_contact = data.get("in_contact", False)
        contacts = data.get("contacts", [])
        if in_contact and contacts:
            for c in contacts:
                b0 = str(c.get("body0", ""))
                b1 = str(c.get("body1", ""))
                for kw in target_keywords:
                    if kw.lower() in b0.lower() or kw.lower() in b1.lower():
                        return True
        return False

    def reset_domain(self, episode_id: int | None = None):
        """Domain randomization applied at each episode reset for single red brick."""
        if episode_id is not None:
            self._episode_id = episode_id
        else:
            self._episode_id += 1

        print(f"\n>>> [Episode {self._episode_id}] [RESET] Randomizing Workspace & Domain (Single Red Lego Brick)...")
        
        cz = TABLE_HEIGHT + 0.001
        
        # Randomize Bin position within workspace
        bin_dist = random.uniform(*BIN_RADIUS_RANGE)
        bin_angle = random.uniform(*BIN_ANGLE_RANGE)
        bin_pos = np.array([
            bin_dist * np.cos(bin_angle),
            bin_dist * np.sin(bin_angle),
            TABLE_HEIGHT
        ])
        # Update USD Stage transform so cuMotion's SceneQuery / WorldBinding reads the new position
        if self.bin_xform is not None:
            self.bin_xform.set_world_poses(
                positions=[bin_pos],
                orientations=[[1.0, 0.0, 0.0, 0.0]]
            )

        # Update PhysX simulation tensor buffer
        if self.bin_obj is not None:
            self.bin_obj.set_world_poses(
                positions=[bin_pos],
                orientations=[[1.0, 0.0, 0.0, 0.0]]
            )
            self.bin_obj.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )
            print(f"  [Episode {self._episode_id}] [Reset] Bin Pos: [{bin_pos[0]:.4f}, {bin_pos[1]:.4f}, {bin_pos[2]:.4f}]")
        
        default_pos = np.array(LEGO_RED_INIT_POS)
        
        # Sample randomized position for the single red brick
        d0 = random.uniform(*LEGO_BRICK_RADIUS_RANGE)
        a0 = random.uniform(*LEGO_BRICK_ANGLE_RANGE)
        brick_pos = np.array([d0 * np.cos(a0), d0 * np.sin(a0), cz])
        
        yaw = random.uniform(0, 2 * np.pi)
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        
        self.lego_prims[0].set_world_poses(positions=[brick_pos], orientations=[q])
        self.lego_prims[0].set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )

        self.update()
        
        red_hit = self._check_collision(self.red_contact, ["Bin"])

        if red_hit:
            print(f"  [Episode {self._episode_id}] [CONTACT SENSOR ALERT] Red brick collision with Bin detected! Defaulting to safe position...")
            self.lego_prims[0].set_world_poses(
                positions=[default_pos],
                orientations=[[1.0, 0.0, 0.0, 0.0]]
            )
            self.lego_prims[0].set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )
            self.update()
        
        # Reset Robot Arm joints to home with slight noise (+- 2 degrees)
        dof_count = self._articulation.num_dofs
        noisy_pose = [random.uniform(-0.035, 0.035) for _ in range(dof_count)]
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


# Backward compatibility alias
RandomizedSO101Env = LegoBrickPickPlaceEnv
