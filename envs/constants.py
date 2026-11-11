"""
Constants for SO-101 pick-and-place simulation.
Pure Python definitions without Isaac Sim runtime dependencies to avoid circular imports.
"""

import numpy as np

# Prim Path Constants
ROBOT_PRIM_PATH = "/World/so101"
ROBOT_ARTICULATION_PATH = "/World/so101/Geometry/base_link"
WORKSPACE_PRIM_PATH = "/World/Workspace"
WORKSPACE_BASE_PRIM_PATH = "/World/Workspace/Base"
WORKSPACE_MAT_PRIM_PATH = "/World/Workspace/Mat"
WALLS_PRIM_PATH = "/World/Walls"
WALL_BACK_PRIM_PATH = "/World/Walls/Wall_Back"
WALL_FRONT_PRIM_PATH = "/World/Walls/Wall_Front"
WALL_RIGHT_PRIM_PATH = "/World/Walls/Wall_Right"
WALL_CEILING_PRIM_PATH = "/World/Walls/Wall_Ceiling"
BIN_PRIM_PATH = "/World/Bin"
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
CEILING_LIGHT_PRIM_PATH = "/World/Lighting/CeilingLight"
WINDOW_LIGHT_PRIM_PATH = "/World/Lighting/WindowSun"
RED_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Red"
BLUE_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Blue"

# Workspace & Mat Dimensions (meters)
# Note: MAT_HEIGHT + WORKSPACE_BASE_HEIGHT == WORKSPACE_HEIGHT
WORKSPACE_HEIGHT: float = 0.05       # Total working surface elevation (5 cm)
WORKSPACE_LENGTH: float = 0.42       # Work mat depth along X axis
WORKSPACE_WIDTH: float = 0.56        # Work mat width along Y axis

MAT_HEIGHT: float = 0.0092           # Thickness of D00620 silicone work mat
WORKSPACE_BASE_HEIGHT: float = WORKSPACE_HEIGHT - MAT_HEIGHT

# Wall Dimensions & Appearance (Cardboard Box with Front Side Open)
WALL_HEIGHT: float = 0.55            # Cardboard box height (elevated slightly above arm standing straight up)
WALL_THICKNESS: float = 0.015        # Realistic cardboard thickness (1.5 cm)
WALL_COLOR: str = "#C19A6B"          # Classic kraft cardboard box brown

# Lego Brick Dimensions at scale 0.10: [half_width (X), half_length (Y), half_height (Z)] (meters)
LEGO_BRICK_HALF_EXTENTS: np.ndarray = np.array([0.00775, 0.01575, 0.005675], dtype=np.float32)
LEGO_BRICK_HALF_HEIGHT: float = 0.0057

# Bin half-dimensions at scale 0.4: [half_length (X), half_width (Y), half_height (Z)] (meters)
# Mesh base size: [0.4281, 0.2905, 0.0936] -> scaled at 0.4: [0.1712, 0.1162, 0.0375]
BIN_HALF_EXTENTS: np.ndarray = np.array([0.0856, 0.0581, 0.0187], dtype=np.float32)
BIN_HALF_LENGTH: float = 0.0856
BIN_HALF_WIDTH: float = 0.0581
BIN_HALF_HEIGHT: float = 0.0187

# Debug Visualization Flags
DEBUG_DRAW: bool = False

