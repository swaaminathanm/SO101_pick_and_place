"""
Standalone visualizer script for inspecting Bin Z-axis yaw rotation and random positioning
in the +Y quadrant with debug bounding box and half-distance from center lines.

Run with:
  ./run.sh scripts/visualize_bin_randomizer.py
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
from envs.constants import BIN_PRIM_PATH, WORKSPACE_HEIGHT
from randomisation import BinRandomisation


def main():
    print("\n[Visualizer] Launching Bin Z-axis rotation & +Y quadrant randomizer inspector...")

    # Setup physics simulation
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cuda")

    # Instantiate Bin randomizer with direct math and prim updates
    bin_randomizer = BinRandomisation(
        bin_prim_path=BIN_PRIM_PATH,
        debug_draw=True,
    )

    # Instantiate the pick-and-place environment
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=None,
        enable_cameras=False,
        randomizers=[bin_randomizer],
    )
    env.setup_scene()

    # Focus viewport camera closely onto the +Y quadrant bin area
    set_camera_view(
        eye=[0.42, 0.22, WORKSPACE_HEIGHT + 0.22],
        target=[0.25, 0.12, WORKSPACE_HEIGHT],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initialize handles
    env.initialize_after_play()

    # Draw the green bounding box and half-distance extents
    bin_randomizer.draw_bounding_box()

    step_count = 0
    steps_per_pose = 90  # 90 physics steps at 60Hz = 1.5 seconds per sample
    sample_idx = 0

    print("\n" + "=" * 80)
    print(">>> Bin Z-axis Rotation & +Y Quadrant Randomizer Visualizer Running")
    print(">>> Sampling new random pose (Yaw around Z in 0-360°, Pos in +Y quadrant) every 1.5s.")
    print(">>> Close the viewport window to exit.")
    print("=" * 80 + "\n")

    # Initial sample
    bin_randomizer.step()

    while simulation_app.is_running():
        simulation_app.update()

        if app_utils.is_playing() and SimulationManager.is_simulating():
            step_count += 1
            if step_count >= steps_per_pose:
                step_count = 0
                sample_idx += 1
                print(f"\n--- [Sample #{sample_idx}] ---")
                bin_randomizer.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
