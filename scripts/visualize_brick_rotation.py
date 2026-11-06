"""
Standalone visualizer script for inspecting Lego brick combined Y-axis (tilt/resting face pitch: 0, 90, 180, 270 deg)
and Z-axis (tabletop yaw: 0-360 deg) rotations, position randomization in the -Y quadrant, and color variations
using LegoBrickRandomisation in Isaac Sim without robot movement.

Run with:
  ./run.sh scripts/visualize_brick_rotation.py
"""

import os
import sys
import numpy as np

# Initialize Isaac Sim SimulationApp
from isaacsim import SimulationApp

HEADLESS = "--headless" in sys.argv
simulation_app = SimulationApp({"headless": HEADLESS})

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from envs import LegoBrickPickPlaceEnv
from envs.constants import RED_BRICK_PRIM_PATH, WORKSPACE_HEIGHT
from randomisation import LegoBrickRandomisation


def main():
    print("\n[Visualizer] Launching Lego brick randomizer inspector (using LegoBrickRandomisation)...")

    # Setup physics simulation
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cuda")

    # Instantiate Lego brick randomizer with direct math and prim updates
    brick_randomizer = LegoBrickRandomisation(
        brick_prim_path=RED_BRICK_PRIM_PATH,
        pos_min=(0.17, -0.20, WORKSPACE_HEIGHT),
        pos_max=(0.38, -0.01, WORKSPACE_HEIGHT),
        y_rot_choices=[0.0, 90.0, 180.0, 270.0],
        z_rot_range=(0.0, 360.0),
        table_z=WORKSPACE_HEIGHT,
        randomize_color=True,
        debug_draw=True
    )

    # Instantiate the pick-and-place environment
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=None,
        enable_cameras=False,
        randomizers=[brick_randomizer],
    )
    env.setup_scene()

    # Focus viewport camera closely onto the brick on the workspace mat (-Y quadrant)
    set_camera_view(
        eye=[0.48, -0.10, WORKSPACE_HEIGHT + 0.25],
        target=[0.27, -0.10, WORKSPACE_HEIGHT],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initialize handles
    env.initialize_after_play()

    # Draw the green bounding box and half-distance extents
    brick_randomizer.draw_bounding_box()

    step_count = 0
    steps_per_pose = 90  # 90 physics steps at 60Hz = 1.5 seconds per sample
    sample_idx = 0

    print("\n" + "=" * 80)
    print(">>> Lego Brick Randomizer Visualizer Running (Direct Math)")
    print(">>> Sampling new random pose (Pitch [0, 90, 180, 270]°, Yaw 0-360°, Pos in -Y quadrant) & color every 1.5s.")
    print(">>> Close the viewport window to exit.")
    print("=" * 80 + "\n")

    # Initial sample
    brick_randomizer.step()

    while simulation_app.is_running():
        simulation_app.update()

        if app_utils.is_playing() and SimulationManager.is_simulating():
            step_count += 1
            if step_count >= steps_per_pose:
                step_count = 0
                sample_idx += 1
                print(f"\n--- [Sample #{sample_idx}] ---")
                brick_randomizer.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
