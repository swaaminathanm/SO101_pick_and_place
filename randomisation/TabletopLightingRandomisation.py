"""
Direct USD Lux Tabletop Lighting Domain Randomizer for SO-101 Pick and Place.

Accurately models domestic indoor lighting physics for Sim2Real transfer:
- Primary top white ceiling light (UsdLuxRectLight / UsdLuxDiskLight) with realistic
  Kelvin color temperatures (4000K-6000K) and ceiling position jitter.
- Window sunlight / diffuse daylight (UsdLuxDistantLight) with omnidirectional
  azimuth sampling (0-360 deg) and low-to-mid elevation angles (15-50 deg).
- Ambient room bounce (UsdLuxDomeLight) preventing unrecoverable black shadows.

Samples across 3 physically realistic domestic states:
  - State A (Night / Blinds Closed, ~30%): Ceiling ON, Sun OFF.
  - State B (Overcast / Indirect Daylight, ~40%): Ceiling ON, Sun soft daylight fill.
  - State C (Direct Sunny Day, ~30%): Ceiling OFF, Sun intense directional beam.

Supports both direct episodic reset stepping (env.reset_domain()) and
Omniverse Replicator custom randomizer registration (@rep.randomizer.register).
"""

from typing import Tuple, List, Optional, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf

try:
    import isaacsim.core.experimental.utils.stage as stage_utils
except ImportError:
    import omni.usd as stage_utils

from envs.constants import (
    CEILING_LIGHT_PRIM_PATH,
    WINDOW_LIGHT_PRIM_PATH,
    DOME_LIGHT_PRIM_PATH,
    WORKSPACE_HEIGHT,
)

# Default geometric positions (meters)
# Table center corresponds to the middle of the D00620 work mat between robot and bin
DEFAULT_TABLE_CENTER: Tuple[float, float, float] = (0.25, 0.0, WORKSPACE_HEIGHT)
DEFAULT_CEILING_HEIGHT: float = 2.0
DEFAULT_CEILING_JITTER_XY: float = 0.15

# Default physical lighting properties
DEFAULT_CEILING_INTENSITY_RANGE: Tuple[float, float] = (15000.0, 45000.0)
DEFAULT_CEILING_TEMP_RANGE: Tuple[float, float] = (4000.0, 6000.0)      # White LED range (Kelvin)
DEFAULT_CEILING_SIZE: Tuple[float, float] = (0.40, 0.40)                # 40cm x 40cm panel

DEFAULT_SUN_DIRECT_INTENSITY_RANGE: Tuple[float, float] = (60000.0, 160000.0)
DEFAULT_SUN_OVERCAST_INTENSITY_RANGE: Tuple[float, float] = (5000.0, 15000.0)
DEFAULT_SUN_DIRECT_TEMP_RANGE: Tuple[float, float] = (5000.0, 5600.0)   # Direct sun (Kelvin)
DEFAULT_SUN_OVERCAST_TEMP_RANGE: Tuple[float, float] = (6200.0, 7200.0) # Diffuse sky (Kelvin)

DEFAULT_SUN_AZIMUTH_RANGE: Tuple[float, float] = (0.0, 360.0)          # Omnidirectional (degrees)
DEFAULT_SUN_ELEVATION_RANGE: Tuple[float, float] = (15.0, 50.0)         # Angle above horizon (degrees)

DEFAULT_AMBIENT_INTENSITY_RANGE: Tuple[float, float] = (300.0, 800.0)   # Indirect room bounce
DEFAULT_STATE_PROBABILITIES: Tuple[float, float, float] = (0.30, 0.40, 0.30)  # [A: Night, B: Overcast, C: Sun]


class TabletopLightingRandomisation:
    """
    Randomizes scene lighting for tabletop pick-and-place manipulation
    using physically grounded USD Lux primitives.
    """

    def __init__(
        self,
        ceiling_prim_path: str = CEILING_LIGHT_PRIM_PATH,
        window_prim_path: str = WINDOW_LIGHT_PRIM_PATH,
        dome_prim_path: str = DOME_LIGHT_PRIM_PATH,
        table_center: Tuple[float, float, float] = DEFAULT_TABLE_CENTER,
        ceiling_height: float = DEFAULT_CEILING_HEIGHT,
        ceiling_jitter_xy: float = DEFAULT_CEILING_JITTER_XY,
        ceiling_intensity_range: Tuple[float, float] = DEFAULT_CEILING_INTENSITY_RANGE,
        ceiling_temp_range: Tuple[float, float] = DEFAULT_CEILING_TEMP_RANGE,
        ceiling_size: Tuple[float, float] = DEFAULT_CEILING_SIZE,
        sun_direct_intensity_range: Tuple[float, float] = DEFAULT_SUN_DIRECT_INTENSITY_RANGE,
        sun_overcast_intensity_range: Tuple[float, float] = DEFAULT_SUN_OVERCAST_INTENSITY_RANGE,
        sun_azimuth_range: Tuple[float, float] = DEFAULT_SUN_AZIMUTH_RANGE,
        sun_elevation_range: Tuple[float, float] = DEFAULT_SUN_ELEVATION_RANGE,
        ambient_intensity_range: Tuple[float, float] = DEFAULT_AMBIENT_INTENSITY_RANGE,
        state_probabilities: Tuple[float, float, float] = DEFAULT_STATE_PROBABILITIES,
        auto_setup: bool = True,
    ):
        self.ceiling_prim_path = ceiling_prim_path
        self.window_prim_path = window_prim_path
        self.dome_prim_path = dome_prim_path

        self.table_center = np.asarray(table_center, dtype=np.float32)
        self.ceiling_height = float(ceiling_height)
        self.ceiling_jitter_xy = float(ceiling_jitter_xy)

        self.ceiling_intensity_range = ceiling_intensity_range
        self.ceiling_temp_range = ceiling_temp_range
        self.ceiling_size = ceiling_size

        self.sun_direct_intensity_range = sun_direct_intensity_range
        self.sun_overcast_intensity_range = sun_overcast_intensity_range
        self.sun_direct_temp_range = DEFAULT_SUN_DIRECT_TEMP_RANGE
        self.sun_overcast_temp_range = DEFAULT_SUN_OVERCAST_TEMP_RANGE

        self.sun_azimuth_range = sun_azimuth_range
        self.sun_elevation_range = sun_elevation_range

        self.ambient_intensity_range = ambient_intensity_range
        self.state_probabilities = state_probabilities
        self.auto_setup = auto_setup

        self._is_rig_setup: bool = False
        self._current_state: str = "init"
        self._last_params: Dict[str, Any] = {}

    def _get_stage(self) -> Optional[Usd.Stage]:
        """Resolves the current active USD stage."""
        if hasattr(stage_utils, "get_current_stage"):
            return stage_utils.get_current_stage()
        elif hasattr(stage_utils, "get_context"):
            return stage_utils.get_context().get_stage()
        return None

    def setup_rig(self, stage: Optional[Usd.Stage] = None) -> bool:
        """
        Instantiates or binds the 3 USD Lux lighting prims on the USD stage:
        - RectLight at CEILING_LIGHT_PRIM_PATH
        - DistantLight at WINDOW_LIGHT_PRIM_PATH
        - DomeLight at DOME_LIGHT_PRIM_PATH
        """
        if stage is None:
            stage = self._get_stage()
        if stage is None:
            print("[TabletopLightingRandomisation] [WARN] Stage not yet initialized. Rig setup deferred.")
            return False

        # Setup Ceiling Light (UsdLuxRectLight)
        ceiling_path = Sdf.Path(self.ceiling_prim_path)
        ceiling_prim = stage.GetPrimAtPath(ceiling_path)
        if not ceiling_prim.IsValid():
            ceiling_light = UsdLux.RectLight.Define(stage, ceiling_path)
        else:
            ceiling_light = UsdLux.RectLight(ceiling_prim)

        ceiling_light.CreateWidthAttr().Set(float(self.ceiling_size[0]))
        ceiling_light.CreateHeightAttr().Set(float(self.ceiling_size[1]))
        ceiling_light.CreateEnableColorTemperatureAttr().Set(True)
        ceiling_light.CreateColorTemperatureAttr().Set(5000.0)
        ceiling_light.CreateIntensityAttr().Set(float(self.ceiling_intensity_range[0]))

        # Ensure transform ops exist for ceiling light
        xform_ceiling = UsdGeom.Xformable(ceiling_light.GetPrim())
        translate_attr = xform_ceiling.GetPrim().GetAttribute("xformOp:translate")
        if not translate_attr or not translate_attr.IsValid():
            xform_ceiling.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)
        
        # Orient facing straight down (-Z emission in USD Lux)
        orient_attr = xform_ceiling.GetPrim().GetAttribute("xformOp:orient")
        if not orient_attr or not orient_attr.IsValid():
            xform_ceiling.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

        # Setup Window Sun Light (UsdLuxDistantLight)
        window_path = Sdf.Path(self.window_prim_path)
        window_prim = stage.GetPrimAtPath(window_path)
        if not window_prim.IsValid():
            distant_light = UsdLux.DistantLight.Define(stage, window_path)
        else:
            distant_light = UsdLux.DistantLight(window_prim)

        distant_light.CreateEnableColorTemperatureAttr().Set(True)
        distant_light.CreateColorTemperatureAttr().Set(5500.0)
        distant_light.CreateIntensityAttr().Set(0.0)  # Default off

        xform_distant = UsdGeom.Xformable(distant_light.GetPrim())
        orient_attr_sun = xform_distant.GetPrim().GetAttribute("xformOp:orient")
        if not orient_attr_sun or not orient_attr_sun.IsValid():
            xform_distant.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

        # Setup Ambient Dome Light (UsdLuxDomeLight)
        dome_path = Sdf.Path(self.dome_prim_path)
        dome_prim = stage.GetPrimAtPath(dome_path)
        if not dome_prim.IsValid():
            dome_light = UsdLux.DomeLight.Define(stage, dome_path)
        else:
            dome_light = UsdLux.DomeLight(dome_prim)

        dome_light.CreateIntensityAttr().Set(float(self.ambient_intensity_range[0]))

        self._is_rig_setup = True
        print(f"[TabletopLightingRandomisation] Lighting rig successfully configured on stage:")
        print(f"  • Ceiling: {self.ceiling_prim_path}")
        print(f"  • Window:  {self.window_prim_path}")
        print(f"  • Dome:    {self.dome_prim_path}")
        return True

    def _sample_state(self) -> str:
        """Samples one of the 3 physical lighting states based on configured probabilities."""
        states = ["night", "overcast", "sunny"]
        probs = np.array(self.state_probabilities, dtype=np.float32)
        probs = probs / np.sum(probs)
        return str(np.random.choice(states, p=probs))

    def _compute_ceiling_position(self) -> Tuple[float, float, float]:
        """Computes ceiling light position with random XY jitter around table center."""
        jx = float(np.random.uniform(-self.ceiling_jitter_xy, self.ceiling_jitter_xy))
        jy = float(np.random.uniform(-self.ceiling_jitter_xy, self.ceiling_jitter_xy))
        return (
            float(self.table_center[0] + jx),
            float(self.table_center[1] + jy),
            float(self.ceiling_height),
        )

    def _apply_lighting_to_stage(
        self,
        stage: Usd.Stage,
        state: str,
        ceil_pos: Tuple[float, float, float],
        c_intensity: float,
        c_temp: float,
        s_intensity: float,
        s_temp: float,
        yaw_z: float,
        s_elevation: float,
        d_intensity: float,
    ) -> Dict[str, Any]:
        """Applies evaluated light parameters to the USD stage prims and logs."""
        # Compute Sun Rotation (Euler to Quaternion)
        # Pitch around Y tilts default downward (-Z) beam UP toward horizon
        # Yaw around Z swings beam around compass to face the window
        roll_x = 0.0
        pitch_y = float(90.0 - s_elevation)
        
        rot = R.from_euler("z", yaw_z, degrees=True) * R.from_euler("y", pitch_y, degrees=True)
        rot = R.from_euler("xyz", [roll_x, pitch_y, yaw_z], degrees=True)
        sun_quat_xyzw = rot.as_quat()  # [qx, qy, qz, qw]
        sun_quat_wxyz = [float(sun_quat_xyzw[3]), float(sun_quat_xyzw[0]), float(sun_quat_xyzw[1]), float(sun_quat_xyzw[2])]

        # Update Ceiling Light (RectLight)
        ceiling_prim = stage.GetPrimAtPath(Sdf.Path(self.ceiling_prim_path))
        if ceiling_prim.IsValid():
            c_light = UsdLux.RectLight(ceiling_prim)
            c_light.GetIntensityAttr().Set(float(c_intensity))
            c_light.GetColorTemperatureAttr().Set(float(c_temp))

            trans_attr = ceiling_prim.GetAttribute("xformOp:translate")
            if trans_attr and trans_attr.IsValid():
                t_type = str(trans_attr.GetTypeName())
                vec = Gf.Vec3d(*ceil_pos) if ("Vec3d" in t_type or "double" in t_type) else Gf.Vec3f(*ceil_pos)
                trans_attr.Set(vec)

            orient_attr = ceiling_prim.GetAttribute("xformOp:orient")
            if orient_attr and orient_attr.IsValid():
                orient_attr.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        # Update Window Sun Light (DistantLight)
        window_prim = stage.GetPrimAtPath(Sdf.Path(self.window_prim_path))
        if window_prim.IsValid():
            s_light = UsdLux.DistantLight(window_prim)
            s_light.GetIntensityAttr().Set(float(s_intensity))
            s_light.GetColorTemperatureAttr().Set(float(s_temp))

            sun_orient_attr = window_prim.GetAttribute("xformOp:orient")
            if sun_orient_attr and sun_orient_attr.IsValid():
                o_type = str(sun_orient_attr.GetTypeName())
                if "Quatd" in o_type:
                    sun_orient_attr.Set(Gf.Quatd(sun_quat_wxyz[0], Gf.Vec3d(sun_quat_wxyz[1], sun_quat_wxyz[2], sun_quat_wxyz[3])))
                else:
                    sun_orient_attr.Set(Gf.Quatf(sun_quat_wxyz[0], Gf.Vec3f(sun_quat_wxyz[1], sun_quat_wxyz[2], sun_quat_wxyz[3])))

        # Update Ambient Dome Light (DomeLight)
        dome_prim = stage.GetPrimAtPath(Sdf.Path(self.dome_prim_path))
        if dome_prim.IsValid():
            d_light = UsdLux.DomeLight(dome_prim)
            d_light.GetIntensityAttr().Set(float(d_intensity))

        self._last_params = {
            "state": state,
            "ceiling_intensity": float(c_intensity),
            "ceiling_temperature": float(c_temp),
            "ceiling_pos": ceil_pos,
            "sun_intensity": float(s_intensity),
            "sun_temperature": float(s_temp),
            "sun_yaw_z": float(yaw_z),
            "sun_azimuth": float(yaw_z),
            "sun_elevation": float(s_elevation),
            "dome_intensity": float(d_intensity),
        }

        print(
            f"[TabletopLightingRandomisation] State={state.upper():<8} | "
            f"Ceiling: Int={c_intensity:>6.0f}, Temp={c_temp:>4.0f}K | "
            f"Sun: Int={s_intensity:>6.0f}, Temp={s_temp:>4.0f}K, Yaw={yaw_z:>5.1f}°, El={s_elevation:>4.1f}° | "
            f"Dome: Int={d_intensity:>4.0f}"
        )
        return self._last_params

    def _step_night_stage(self, stage: Usd.Stage, ceil_pos: Tuple[float, float, float]) -> Dict[str, Any]:
        """State A: Pure artificial ceiling light, window completely dark (night/blinds closed)."""
        c_intensity = float(np.random.uniform(self.ceiling_intensity_range[0], self.ceiling_intensity_range[1]))
        c_temp = float(np.random.uniform(self.ceiling_temp_range[0], self.ceiling_temp_range[1]))
        s_intensity = 0.0
        s_temp = 5500.0
        yaw_z = float(np.random.uniform(self.sun_azimuth_range[0], self.sun_azimuth_range[1]))
        s_elevation = float(np.random.uniform(self.sun_elevation_range[0], self.sun_elevation_range[1]))
        d_intensity = float(np.random.uniform(self.ambient_intensity_range[0], self.ambient_intensity_range[0] * 1.3))

        return self._apply_lighting_to_stage(
            stage=stage,
            state="night",
            ceil_pos=ceil_pos,
            c_intensity=c_intensity,
            c_temp=c_temp,
            s_intensity=s_intensity,
            s_temp=s_temp,
            yaw_z=yaw_z,
            s_elevation=s_elevation,
            d_intensity=d_intensity,
        )

    def _step_overcast_stage(self, stage: Usd.Stage, ceil_pos: Tuple[float, float, float]) -> Dict[str, Any]:
        """State B: Ceiling light on + soft diffuse daylight fill from window."""
        c_intensity = float(np.random.uniform(self.ceiling_intensity_range[0], self.ceiling_intensity_range[1]))
        c_temp = float(np.random.uniform(4500.0, 5200.0))
        s_intensity = float(np.random.uniform(self.sun_overcast_intensity_range[0], self.sun_overcast_intensity_range[1]))
        s_temp = float(np.random.uniform(self.sun_overcast_temp_range[0], self.sun_overcast_temp_range[1]))
        yaw_z = float(np.random.uniform(self.sun_azimuth_range[0], self.sun_azimuth_range[1]))
        s_elevation = float(np.random.uniform(self.sun_elevation_range[0], self.sun_elevation_range[1]))
        d_intensity = float(np.random.uniform(self.ambient_intensity_range[0] * 1.2, self.ambient_intensity_range[1]))

        return self._apply_lighting_to_stage(
            stage=stage,
            state="overcast",
            ceil_pos=ceil_pos,
            c_intensity=c_intensity,
            c_temp=c_temp,
            s_intensity=s_intensity,
            s_temp=s_temp,
            yaw_z=yaw_z,
            s_elevation=s_elevation,
            d_intensity=d_intensity,
        )

    def _step_sunny_stage(self, stage: Usd.Stage, ceil_pos: Tuple[float, float, float]) -> Dict[str, Any]:
        """State C: Pure direct natural sunlight beam, ceiling light OFF."""
        c_intensity = 0.0
        c_temp = 5000.0
        s_intensity = float(np.random.uniform(self.sun_direct_intensity_range[0], self.sun_direct_intensity_range[1]))
        s_temp = float(np.random.uniform(self.sun_direct_temp_range[0], self.sun_direct_temp_range[1]))
        yaw_z = float(np.random.uniform(self.sun_azimuth_range[0], self.sun_azimuth_range[1]))
        s_elevation = float(np.random.uniform(self.sun_elevation_range[0], self.sun_elevation_range[1]))
        d_intensity = float(np.random.uniform(self.ambient_intensity_range[0] * 1.3, self.ambient_intensity_range[1]))

        return self._apply_lighting_to_stage(
            stage=stage,
            state="sunny",
            ceil_pos=ceil_pos,
            c_intensity=c_intensity,
            c_temp=c_temp,
            s_intensity=s_intensity,
            s_temp=s_temp,
            yaw_z=yaw_z,
            s_elevation=s_elevation,
            d_intensity=d_intensity,
        )

    def step(self, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes one domain randomization step for scene lighting.
        Dispatches to clean, dedicated stage step functions based on active state:
          - _step_night_stage (State A: Night / Blinds Closed)
          - _step_overcast_stage (State B: Overcast / Diffuse Daylight)
          - _step_sunny_stage (State C: Direct Sun, Ceiling OFF)
        """
        stage = self._get_stage()
        if stage is None:
            return {}

        if not self._is_rig_setup:
            if not self.setup_rig(stage):
                return {}

        if state is None:
            state = self._sample_state()
        self._current_state = state

        ceil_pos = self._compute_ceiling_position()

        # Clean dispatch to separate stage functions per state
        if state == "night":
            return self._step_night_stage(stage, ceil_pos)
        elif state == "overcast":
            return self._step_overcast_stage(stage, ceil_pos)
        elif state == "sunny":
            return self._step_sunny_stage(stage, ceil_pos)
        else:
            raise ValueError(f"Unknown lighting state: '{state}'. Allowed states are 'night', 'overcast', 'sunny'.")

    def draw_bounding_box(self, *args, **kwargs) -> None:
        """Safe visualizer no-op interface for compatibility with other randomizers."""
        pass

    def randomize(self) -> Dict[str, Any]:
        """Alias for step() for declarative Replicator graph loops."""
        return self.step()

    def register_replicator(self):
        """Registers a custom Replicator randomizer for SDG pipelines."""
        try:
            import omni.replicator.core as rep

            @rep.randomizer.register
            def randomize_tabletop_lighting():
                self.step()

            return randomize_tabletop_lighting
        except Exception as e:
            print(f"[TabletopLightingRandomisation] Could not register Replicator randomizer: {e}")
            return None
