"""
Constants for SO-101 pick-and-place simulation.
Pure Python definitions without Isaac Sim runtime dependencies to avoid circular imports.
"""

# Prim Path Constants
ROBOT_PRIM_PATH = "/World/so101"
ROBOT_ARTICULATION_PATH = "/World/so101/Geometry/base_link"
TABLE_PRIM_PATH = "/World/Table"
BIN_PRIM_PATH = "/World/Bin"
GROUND_PLANE_PRIM_PATH = "/World/GroundPlane"
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
RED_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Red"
BLUE_BRICK_PRIM_PATH = "/World/Legos/LegoBrick_Blue"

# Table Dimensions (meters)
TABLE_HEIGHT: float = 0.08
TABLE_LENGTH: float = 0.40
TABLE_WIDTH: float = 0.60
