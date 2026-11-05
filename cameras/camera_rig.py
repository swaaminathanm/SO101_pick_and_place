"""
Camera rig module for SO-101 pick-and-place task in Isaac Sim.
Uses clean, fixed poses for both Overhead and Wrist cameras.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation


def compute_overhead_camera_pose() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns position, quaternion [w, x, y, z], and Euler angles (XYZ degrees)
    for the overhead camera.
    """
    pos = np.array([-0.01, 0.24125, 0.35], dtype=np.float32)
    euler_deg = np.array([-40, -55, -133], dtype=np.float32)

    rot = Rotation.from_euler("XYZ", euler_deg, degrees=True)
    quat = rot.as_quat(scalar_first=True).astype(np.float32)

    return pos, quat, euler_deg

def compute_wrist_camera_pose() -> Tuple[np.ndarray, np.ndarray]:
    """     
Computes the local pose of the wrist camera relative to the gripper link.

    Args:
        translation: np.ndarray shape (3,) -> [x, y, z] in meters
        rotation: np.ndarray shape (4,) -> quaternion [w, x, y, z]

    Returns:
        translation: np.ndarray shape (3,)
        rotation: np.ndarray shape (4,)
    """
    translation = np.array([0.0, -0.05, 0.03], dtype=np.float32)  # Applied on wrist_roll_follower_so101_v1
    rot = Rotation.from_euler("xyz", [150.0, 0.0, 0.0], degrees=True)
    rotation = rot.as_quat(scalar_first=True).astype(np.float32)
    return translation, rotation


# ==================== FIXED CAMERA POSES ====================
OVERHEAD_CAM_POS, OVERHEAD_CAM_QUAT, OVERHEAD_CAM_EULER = compute_overhead_camera_pose()
WRIST_CAM_POS, WRIST_CAM_QUAT = compute_wrist_camera_pose()
# ============================================================


class SO101CameraRig:
    """
    Manages camera sensors for the SO-101 workspace:
      1. Overhead Camera: Fixed overview of the workspace and bin.
      2. Wrist Camera: Mounted on gripper link, moves with the hand.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        frequency: int = 30,
        enable_overhead: bool = True,
        enable_wrist: bool = True,
        overhead_focal_length: float = 1.85,
        overhead_horizontal_aperture: float = 3.84,
        overhead_vertical_aperture: float = 2.16,
        near_clipping_plane: float = 0.01,
        far_clipping_plane: float = 10000.0,
    ):
        self.resolution = resolution
        self.frequency = frequency
        self.enable_overhead = enable_overhead
        self.enable_wrist = enable_wrist

        # Seeed Studio OV9732 (1MP, 100° FOV) physical optical parameters
        self.overhead_focal_length = overhead_focal_length
        self.overhead_horizontal_aperture = overhead_horizontal_aperture
        self.overhead_vertical_aperture = overhead_vertical_aperture
        self.near_clipping_plane = near_clipping_plane
        self.far_clipping_plane = far_clipping_plane

        self.overhead_cam: Optional[Any] = None
        self.wrist_cam: Optional[Any] = None

    def setup_cameras(
        self,
        robot_prim_path: str = "/World/so101",
        robot_articulation_path: str = "/World/so101/Geometry/base_link",
    ) -> None:
        """Creates the camera prims on the USD stage with fixed poses."""
        from pxr import Gf
        from isaacsim.sensors.camera import Camera

        # Static Overhead Camera (OV9732 1MP, 100° FOV)
        if self.enable_overhead:
            self.overhead_cam = Camera(
                prim_path="/World/Camera_Overhead",
                name="overhead_camera",
                resolution=self.resolution,
                frequency=self.frequency,
            )
            # Set pose using camera_axes="usd" so Isaac Sim doesn't add the default +90° world conversion
            self.overhead_cam.set_world_pose(
                position=OVERHEAD_CAM_POS,
                orientation=OVERHEAD_CAM_QUAT,
                camera_axes="usd",
            )
            overhead_prim = self.overhead_cam.prim
            overhead_prim.GetAttribute("horizontalAperture").Set(self.overhead_horizontal_aperture)
            overhead_prim.GetAttribute("verticalAperture").Set(self.overhead_vertical_aperture)
            overhead_prim.GetAttribute("focalLength").Set(self.overhead_focal_length)
            overhead_prim.GetAttribute("clippingRange").Set(
                Gf.Vec2f(self.near_clipping_plane, self.far_clipping_plane)
            )

            print(f"Overhead Camera Pose: Position={OVERHEAD_CAM_POS}, Quaternion={OVERHEAD_CAM_QUAT}, Euler={OVERHEAD_CAM_EULER}")

        # Moving Wrist Camera (child of gripper_link)
        if self.enable_wrist:
            wrist_prim_path = (
                f"{robot_articulation_path}/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/wrist_roll_follower_so101_v1/wrist_roll_follower_so101_v1/wrist_camera"
            )
            self.wrist_cam = Camera(
                prim_path=wrist_prim_path,
                name="wrist_camera",
                resolution=self.resolution,
                frequency=self.frequency,
            )
            self.wrist_cam.set_local_pose(
                translation=WRIST_CAM_POS,
                orientation=WRIST_CAM_QUAT,
                camera_axes="usd",
            )
            wrist_prim = self.wrist_cam.prim
            wrist_prim.GetAttribute("horizontalAperture").Set(self.overhead_horizontal_aperture)
            wrist_prim.GetAttribute("verticalAperture").Set(self.overhead_vertical_aperture)
            wrist_prim.GetAttribute("focalLength").Set(self.overhead_focal_length)
            wrist_prim.GetAttribute("clippingRange").Set(
                Gf.Vec2f(self.near_clipping_plane, self.far_clipping_plane)
            )

    def initialize(self) -> None:
        """Initializes sensors and attaches RGB & Depth annotators."""
        for cam in [self.overhead_cam, self.wrist_cam]:
            if cam is not None:
                cam.initialize()
                cam.add_rgb_to_frame()
                cam.add_distance_to_image_plane_to_frame()

    def get_observations(self) -> Dict[str, Optional[np.ndarray]]:
        """Retrieves the latest RGB and Depth frames from both cameras."""
        obs = {
            "overhead_rgb": None,
            "overhead_depth": None,
            "wrist_rgb": None,
            "wrist_depth": None,
        }

        if self.overhead_cam is not None:
            self.overhead_cam.get_current_frame()
            obs["overhead_rgb"] = self.overhead_cam.get_rgb()
            obs["overhead_depth"] = self.overhead_cam.get_depth()

        if self.wrist_cam is not None:
            self.wrist_cam.get_current_frame()
            obs["wrist_rgb"] = self.wrist_cam.get_rgb()
            obs["wrist_depth"] = self.wrist_cam.get_depth()

        return obs
