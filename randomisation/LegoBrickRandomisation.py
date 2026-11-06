"""
Direct Math and Prim Domain Randomizer for SO-101 Pick and Place Lego Brick.
Samples poses and resting face orientations directly using NumPy/SciPy math,
computes dynamic table-flush elevations, and updates USD stage attributes and
PhysX simulation tensor buffers without Omniverse Replicator.
"""

from typing import Tuple, Any, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdGeom, Gf
import isaacsim.core.experimental.utils.stage as stage_utils
from envs.constants import RED_BRICK_PRIM_PATH, WORKSPACE_HEIGHT, DEBUG_DRAW_BOUNDING_BOX

# Lego brick half-dimensions at scale 0.25: [half_width, half_length, half_height] (meters)
LEGO_BRICK_HALF_EXTENTS: np.ndarray = np.array([0.019375, 0.039375, 0.0141875], dtype=np.float32)
LEGO_BRICK_HALF_HEIGHT: float = 0.0142

# Discrete pitch (Y) rotation choices corresponding to stable resting faces (degrees):
# 0° (flat up, studs +Z), 90° (right side, studs +X), 180° (flat upside down, studs -Z), 270° (left side, studs -X)
DEFAULT_Y_ROT_CHOICES: List[float] = [0.0, 90.0, 180.0, 270.0]
DEFAULT_Z_ROT_RANGE: Tuple[float, float] = (0.0, 360.0)

# Standard Lego brick RGB colors for discrete albedo sampling
DEFAULT_LEGO_COLORS: List[Tuple[float, float, float]] = [
    (0.85, 0.10, 0.10),  # Red
    (0.10, 0.35, 0.85),  # Blue
    (0.10, 0.75, 0.20),  # Green
    (0.95, 0.85, 0.10),  # Yellow
    (0.95, 0.50, 0.10),  # Orange
    (0.60, 0.15, 0.70),  # Purple
    (0.10, 0.80, 0.80),  # Cyan
    (0.90, 0.90, 0.90),  # White
]


class LegoBrickRandomisation:
    """
    Randomizes the Lego brick position, orientation (resting face pitch and tabletop yaw),
    and material albedo (diffuse color) using direct math and updates the USD prim
    and PhysX simulation buffer without Omniverse Replicator.
    """

    def __init__(
        self,
        brick_prim_path: str = RED_BRICK_PRIM_PATH,
        rigid_prim: Any = None,
        pos_min: Tuple[float, float, float] = (0.17, -0.20, WORKSPACE_HEIGHT),
        pos_max: Tuple[float, float, float] = (0.35, -0.01, WORKSPACE_HEIGHT),
        y_rot_choices: Optional[List[float]] = None,
        z_rot_range: Tuple[float, float] = DEFAULT_Z_ROT_RANGE,
        table_z: float = WORKSPACE_HEIGHT + 0.003,
        half_extents: Optional[np.ndarray] = None,
        rot_min: Optional[Tuple[float, float, float]] = None,
        rot_max: Optional[Tuple[float, float, float]] = None,
        randomize_color: bool = True,
        color_choices: Optional[List[Tuple[float, float, float]]] = DEFAULT_LEGO_COLORS,
        color_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        color_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        shader_prim_path: Optional[str] = None,
        debug_draw: bool = DEBUG_DRAW_BOUNDING_BOX
    ):
        self.brick_prim_path = brick_prim_path
        self.rigid_prim = rigid_prim
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.table_z = float(table_z)
        self.half_extents = np.asarray(
            half_extents if half_extents is not None else LEGO_BRICK_HALF_EXTENTS,
            dtype=np.float32,
        )

        # Discrete pitch Y choices [0, 90, 180, 270 deg] and continuous yaw Z range [0-360 deg]
        self.y_rot_choices = list(y_rot_choices) if y_rot_choices is not None else list(DEFAULT_Y_ROT_CHOICES)
        self.z_rot_range = z_rot_range
        self.rot_min = rot_min
        self.rot_max = rot_max

        self.randomize_color = randomize_color
        self.color_choices = color_choices
        self.color_min = color_min
        self.color_max = color_max
        self.shader_prim_path = shader_prim_path
        self.debug_draw = debug_draw

        # Centroid planar bounds match pos bounds directly:
        self._center_min = self.pos_min
        self._center_max = self.pos_max

        # Cached state of last randomized parameters
        self.last_center_pos: Optional[np.ndarray] = None
        self.last_quat_wxyz: Optional[List[float]] = None
        self.last_color: Optional[Tuple[float, float, float]] = None

        stage = stage_utils.get_current_stage()
        if stage and stage.GetPrimAtPath(self.brick_prim_path).IsValid():
            self._resolve_shader_path()
            if self.debug_draw:
                self.draw_bounding_box()

    def _resolve_shader_path(self) -> None:
        """Finds the OmniPBR Shader prim containing inputs:diffuse_color_constant."""
        if self.shader_prim_path is not None:
            return

        stage = stage_utils.get_current_stage()
        if stage is None:
            return

        root_prim = stage.GetPrimAtPath(self.brick_prim_path)
        if root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                if prim.GetTypeName() == "Shader" and prim.HasAttribute("inputs:diffuse_color_constant"):
                    self.shader_prim_path = str(prim.GetPath())
                    break

    def _get_shader_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        """Returns the Shader prim responsible for the brick's diffuse albedo color."""
        if self.shader_prim_path:
            prim = stage.GetPrimAtPath(self.shader_prim_path)
            if prim and prim.IsValid():
                return prim

        root_prim = stage.GetPrimAtPath(self.brick_prim_path)
        if root_prim and root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                if prim.GetTypeName() == "Shader" and prim.HasAttribute("inputs:diffuse_color_constant"):
                    self.shader_prim_path = str(prim.GetPath())
                    return prim
        return None

    def set_brick_color(self, rgb: Tuple[float, float, float]) -> None:
        """Directly sets the brick's diffuse albedo color in USD."""
        stage = stage_utils.get_current_stage()
        if stage is None:
            return
        shader_prim = self._get_shader_prim(stage)
        if shader_prim and shader_prim.IsValid():
            attr = shader_prim.GetAttribute("inputs:diffuse_color_constant")
            if attr and attr.IsValid():
                attr.Set(Gf.Vec3f(*rgb))
                self.last_color = rgb
                print(f"[LegoBrickRandomisation] Set Brick Color (Albedo): RGB=[{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}]")

    def get_brick_color(self) -> Optional[Tuple[float, float, float]]:
        """Returns the current diffuse albedo color RGB tuple from USD."""
        stage = stage_utils.get_current_stage()
        if stage is None:
            return self.last_color
        shader_prim = self._get_shader_prim(stage)
        if shader_prim and shader_prim.IsValid():
            attr = shader_prim.GetAttribute("inputs:diffuse_color_constant")
            if attr and attr.IsValid():
                c = attr.Get()
                if c is not None:
                    return (float(c[0]), float(c[1]), float(c[2]))
        return self.last_color

    def step(self) -> None:
        """
        Performs domain randomization using direct math and prim updates:
        1. Samples planar position (X, Y) within [pos_min, pos_max].
        2. Samples pitch (resting faces: 0, 90, 180, 270 deg) and yaw (0-360 deg).
        3. Computes dynamic resting elevation (Z = table_z + half_h) flush on table.
        4. Updates PhysX simulation buffer directly and resets velocities to zero.
        5. Updates USD stage transform attributes (xformOp:translate, xformOp:orient).
        6. Samples and applies discrete or continuous albedo color directly to shader prim.
        """
        stage = stage_utils.get_current_stage()
        if stage is None:
            return

        brick_prim = stage.GetPrimAtPath(self.brick_prim_path)
        if not (brick_prim and brick_prim.IsValid()):
            return

        self._resolve_shader_path()

        # 1. Sample planar position (X, Y)
        x = float(np.random.uniform(self.pos_min[0], self.pos_max[0]))
        y = float(np.random.uniform(self.pos_min[1], self.pos_max[1]))

        # 2. Sample orientation using pitch Y resting faces and tabletop yaw Z
        roll_x = 0.0
        pitch_y = float(np.random.choice(self.y_rot_choices))
        yaw_z = float(np.random.uniform(self.z_rot_range[0], self.z_rot_range[1]))

        # Euler angles in degrees (roll X, pitch Y, yaw Z)
        rot = R.from_euler("xyz", [roll_x, pitch_y, yaw_z], degrees=True)
        quat_xyzw = rot.as_quat()  # [qx, qy, qz, qw]
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]

        # 3. Dynamic half-height keeps bottom face resting flush on table for ANY orientation
        rot_mat = rot.as_matrix()
        half_h = float(np.abs(rot_mat[2, :]) @ self.half_extents)
        center_z = float(self.table_z + half_h)
        center_pos = np.array([x, y, center_z], dtype=np.float32)

        self.last_center_pos = center_pos
        self.last_quat_wxyz = quat_wxyz

        # 4. Synchronize to PhysX simulation tensor buffer
        if self.rigid_prim is None:
            try:
                from isaacsim.core.experimental.prims import RigidPrim
                self.rigid_prim = RigidPrim(self.brick_prim_path)
            except Exception:
                pass

        if self.rigid_prim is not None:
            self.rigid_prim.set_world_poses(
                positions=[center_pos.tolist()],
                orientations=[quat_wxyz],
            )
            self.rigid_prim.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )

        # 5. Synchronize USD stage transform attributes
        translate_attr = brick_prim.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            t_type = str(translate_attr.GetTypeName())
            vec = Gf.Vec3d(float(x), float(y), float(center_z)) if ("Vec3d" in t_type or "double" in t_type) else Gf.Vec3f(float(x), float(y), float(center_z))
            translate_attr.Set(vec)

        orient_attr = brick_prim.GetAttribute("xformOp:orient")
        if orient_attr and orient_attr.IsValid():
            o_type = str(orient_attr.GetTypeName())
            if "Quatd" in o_type:
                orient_attr.Set(Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))
            else:
                orient_attr.Set(Gf.Quatf(quat_wxyz[0], Gf.Vec3f(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))

        # 6. Randomize material diffuse albedo directly
        if self.randomize_color:
            if self.color_choices is not None and len(self.color_choices) > 0:
                color_idx = int(np.random.choice(len(self.color_choices)))
                chosen_color = self.color_choices[color_idx]
            else:
                chosen_color = (
                    float(np.random.uniform(self.color_min[0], self.color_max[0])),
                    float(np.random.uniform(self.color_min[1], self.color_max[1])),
                    float(np.random.uniform(self.color_min[2], self.color_max[2])),
                )
            self.set_brick_color(chosen_color)

        print(
            f"[LegoBrickRandomisation] Center=[{center_pos[0]:.4f}, {center_pos[1]:.4f}, {center_pos[2]:.4f}], "
            f"Half-H={half_h*1000:.1f}mm, Yaw={yaw_z:.1f}°, Pitch={pitch_y:.1f}°, "
            f"Quaternion=[{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]"
        )

    def draw_bounding_box(
        self,
        color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        width: float = 3.0,
        z_offset: float = 0.003,
    ) -> None:
        """
        Visualizes the randomization bounding box in the Isaac Sim viewport
        using the debug draw line interface. Draws the outer bounding box
        expanded by the half-distance extents when draw_half_distance_extents is True.
        """
        try:
            from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            print(f"[LegoBrickRandomisation] Could not acquire debug draw interface: {e}")
            return

        x_min, y_min = self.pos_min[0], self.pos_min[1]
        x_max, y_max = self.pos_max[0], self.pos_max[1]
        z_draw = self.table_z + z_offset

        max_extent = float(max(self.half_extents[0], self.half_extents[1]))
        ox_min, ox_max = x_min - max_extent, x_max + max_extent
        oy_min, oy_max = y_min - max_extent, y_max + max_extent
        corners = [
            (ox_min, oy_min, z_draw),
            (ox_max, oy_min, z_draw),
            (ox_max, oy_max, z_draw),
            (ox_min, oy_max, z_draw),
        ]
        print(
            f"[LegoBrickRandomisation] Drew Outer Bounding Box: "
            f"X=[{ox_min:.4f}, {ox_max:.4f}], Y=[{oy_min:.4f}, {oy_max:.4f}], Z={z_draw:.4f} "
            f"(Half-Distance Extents: [max_h={max_extent:.4f}])"
        )

        starts = [corners[i] for i in range(4)]
        ends = [corners[(i + 1) % 4] for i in range(4)]
        colors = [color] * 4
        widths = [width] * 4

        draw.draw_lines(starts, ends, colors, widths)

    def clear_bounding_box(self) -> None:
        """Clears debug draw lines from the viewport."""
        try:
            from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
            draw.clear_lines()
        except Exception as e:
            print(f"[LegoBrickRandomisation] Could not clear debug draw lines: {e}")

