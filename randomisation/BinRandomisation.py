"""
Direct Math and Prim Domain Randomizer for Bin in SO-101 Pick and Place.
Samples tabletop yaw rotation (0-360 deg) and position in the +Y quadrant workspace
using pure NumPy/SciPy math, updating USD stage attributes, PhysX RigidPrim, and cuMotion
XformPrim obstacle tracking without Omniverse Replicator.
"""

from typing import Tuple, Any, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdGeom, Gf
import isaacsim.core.experimental.utils.stage as stage_utils
from envs.constants import (
    BIN_PRIM_PATH,
    WORKSPACE_HEIGHT,
    DEBUG_DRAW_BOUNDING_BOX,
    BIN_HALF_EXTENTS,
    BIN_HALF_LENGTH,
    BIN_HALF_WIDTH,
    BIN_HALF_HEIGHT,
)

# Continuous tabletop yaw rotation range around Z-axis (degrees)
DEFAULT_BIN_Z_ROT_RANGE: Tuple[float, float] = (0.0, 360.0)

DEFAULT_BIN_POS_MIN: Tuple[float, float, float] = (0.22, 0.13, WORKSPACE_HEIGHT)
DEFAULT_BIN_POS_MAX: Tuple[float, float, float] = (0.30, 0.18, WORKSPACE_HEIGHT)


class BinRandomisation:
    """
    Randomizes the Bin position within the +Y workspace quadrant and orientation
    around the Z-axis (tabletop yaw: 0-360 deg) using direct math.
    Synchronizes the authored USD transform into the PhysX simulation buffer
    and cuMotion obstacle tracking directly without Omniverse Replicator.
    """

    def __init__(
        self,
        bin_prim_path: str = BIN_PRIM_PATH,
        rigid_prim: Any = None,
        xform_prim: Any = None,
        pos_min: Tuple[float, float, float] = DEFAULT_BIN_POS_MIN,
        pos_max: Tuple[float, float, float] = DEFAULT_BIN_POS_MAX,
        z_rot_range: Tuple[float, float] = DEFAULT_BIN_Z_ROT_RANGE,
        table_z: float = WORKSPACE_HEIGHT + 0.001,
        half_extents: Optional[np.ndarray] = None,
        debug_draw: bool = DEBUG_DRAW_BOUNDING_BOX,
    ):
        self.bin_prim_path = bin_prim_path
        self.rigid_prim = rigid_prim
        self.xform_prim = xform_prim
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.z_rot_range = z_rot_range
        self.table_z = float(table_z)
        self.half_extents = np.asarray(
            half_extents if half_extents is not None else BIN_HALF_EXTENTS,
            dtype=np.float32,
        )
        self.debug_draw = debug_draw

        # Centroid planar bounds match pos bounds directly
        self._center_min = self.pos_min
        self._center_max = self.pos_max

        # Cached last state
        self.last_pos: Optional[np.ndarray] = None
        self.last_quat_wxyz: Optional[list] = None

        stage = stage_utils.get_current_stage()
        if stage and stage.GetPrimAtPath(self.bin_prim_path).IsValid():
            if self.debug_draw:
                self.draw_bounding_box()

    def step(self) -> None:
        """
        Samples Bin position in +Y quadrant and Z tabletop yaw orientation with math,
        and synchronizes directly to USD stage, PhysX simulation buffer, and cuMotion XformPrim.
        """
        stage = stage_utils.get_current_stage()
        if stage is None:
            return

        bin_prim = stage.GetPrimAtPath(self.bin_prim_path)
        if not (bin_prim and bin_prim.IsValid()):
            return

        # 1. Sample planar position (X, Y) within +Y quadrant
        x = float(np.random.uniform(self.pos_min[0], self.pos_max[0]))
        y = float(np.random.uniform(self.pos_min[1], self.pos_max[1]))
        center_z = float(self.table_z)
        center_pos = np.array([x, y, center_z], dtype=np.float32)

        # 2. Sample tabletop yaw rotation around Z (Roll=0, Pitch=0)
        yaw_z = float(np.random.uniform(self.z_rot_range[0], self.z_rot_range[1]))
        rot = R.from_euler("xyz", [0.0, 0.0, yaw_z], degrees=True)
        quat_xyzw = rot.as_quat()
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

        self.last_pos = center_pos
        self.last_quat_wxyz = quat_wxyz

        # 3. Lazy acquire RigidPrim and XformPrim handles if not yet bound
        if self.rigid_prim is None:
            try:
                from isaacsim.core.experimental.prims import RigidPrim
                self.rigid_prim = RigidPrim(self.bin_prim_path)
            except Exception:
                raise RuntimeError(f"[BinRandomisation] Could not acquire RigidPrim for {self.bin_prim_path}")

        if self.xform_prim is None:
            try:
                from isaacsim.core.experimental.prims import XformPrim
                self.xform_prim = XformPrim(self.bin_prim_path)
            except Exception:
                raise RuntimeError(f"[BinRandomisation] Could not acquire XformPrim for {self.bin_prim_path}")

        # 4. Synchronize to PhysX simulation tensor buffer
        if self.rigid_prim is not None:
            self.rigid_prim.set_world_poses(
                positions=[center_pos.tolist()],
                orientations=[quat_wxyz],
            )
            self.rigid_prim.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )

        # 5. Synchronize XformPrim so cuMotion's SceneQuery / WorldBinding tracks the obstacle
        if self.xform_prim is not None:
            self.xform_prim.set_world_poses(
                positions=[center_pos.tolist()],
                orientations=[quat_wxyz],
            )

        # 6. Update USD stage attributes so USD stage matches PhysX exactly
        translate_attr = bin_prim.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            t_type = str(translate_attr.GetTypeName())
            vec = Gf.Vec3d(float(x), float(y), float(center_z)) if ("Vec3d" in t_type or "double" in t_type) else Gf.Vec3f(float(x), float(y), float(center_z))
            translate_attr.Set(vec)

        orient_attr = bin_prim.GetAttribute("xformOp:orient")
        if orient_attr and orient_attr.IsValid():
            o_type = str(orient_attr.GetTypeName())
            if "Quatd" in o_type:
                orient_attr.Set(Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))
            else:
                orient_attr.Set(Gf.Quatf(quat_wxyz[0], Gf.Vec3f(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))

        print(
            f"[BinRandomisation] Center=[{center_pos[0]:.4f}, {center_pos[1]:.4f}, {center_pos[2]:.4f}], "
            f"Yaw={yaw_z:.1f}°, Quaternion=[{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]"
        )

    def draw_bounding_box(
        self,
        color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        width: float = 3.0,
        z_offset: float = 0.001,
    ) -> None:
        """
        Visualizes the randomization bounding box in the Isaac Sim viewport
        using the debug draw line interface. Draws the green outer bounding box
        expanded by the half distance from center [pos_min - half_extents, pos_max + half_extents].
        """
        try:
            from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            print(f"[BinRandomisation] Could not acquire debug draw interface: {e}")
            return

        x_min, y_min = self.pos_min[0], self.pos_min[1]
        x_max, y_max = self.pos_max[0], self.pos_max[1]
        z_draw = self.table_z + z_offset

        hx = float(self.half_extents[0])
        hy = float(self.half_extents[1])
        ox_min, ox_max = x_min - hx, x_max + hx
        oy_min, oy_max = y_min - hy, y_max + hy

        corners = [
            (ox_min, oy_min, z_draw),
            (ox_max, oy_min, z_draw),
            (ox_max, oy_max, z_draw),
            (ox_min, oy_max, z_draw),
        ]
        starts = [corners[i] for i in range(4)]
        ends = [corners[(i + 1) % 4] for i in range(4)]
        colors = [color] * 4
        widths = [width] * 4

        draw.draw_lines(starts, ends, colors, widths)
        print(
            f"[BinRandomisation] Drew Outer Bounding Box: "
            f"X=[{ox_min:.4f}, {ox_max:.4f}], Y=[{oy_min:.4f}, {oy_max:.4f}], Z={z_draw:.4f} "
            f"(Half-Distance Extents: [hx={hx:.4f}, hy={hy:.4f}])"
        )

    def clear_bounding_box(self) -> None:
        """Clears debug draw lines from the viewport."""
        try:
            from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
            draw.clear_lines()
        except Exception as e:
            print(f"[BinRandomisation] Could not clear debug draw lines: {e}")


# Backwards compatibility aliases
BinRandomizer = BinRandomisation
BinReplicatorRandomizer = BinRandomisation
