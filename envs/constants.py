"""
Constants for SO-101 pick-and-place simulation.
Pure Python definitions without Isaac Sim runtime dependencies to avoid circular imports.
"""

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
BIN_PRIM_PATH = "/World/Bin"
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
RED_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Red"
BLUE_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Blue"

# Workspace & Mat Dimensions (meters)
# Note: MAT_HEIGHT + WORKSPACE_BASE_HEIGHT == WORKSPACE_HEIGHT
WORKSPACE_HEIGHT: float = 0.05       # Total working surface elevation (5 cm)
WORKSPACE_LENGTH: float = 0.42       # Work mat depth along X axis
WORKSPACE_WIDTH: float = 0.56        # Work mat width along Y axis

MAT_HEIGHT: float = 0.0092           # Thickness of D00620 silicone work mat
WORKSPACE_BASE_HEIGHT: float = WORKSPACE_HEIGHT - MAT_HEIGHT

# Debug Visualization Flags
DEBUG_DRAW_BOUNDING_BOX: bool = True

