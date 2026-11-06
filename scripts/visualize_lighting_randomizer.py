"""
Standalone visualizer script for inspecting Tabletop Lighting Domain Randomization
in Isaac Sim for the SO-101 pick-and-place workspace.

Cycles through the 3 real-world domestic lighting states:
  1. Night / Blinds Closed (Top white ceiling light ON, Sun OFF)
  2. Overcast / Indirect Daylight (Ceiling light ON + diffuse cool daylight from window)
  3. Direct Sunny Day (Ceiling light OFF, intense angled sunbeam casting long shadows)

Run with:
  ./run.sh scripts/visualize_lighting_randomizer.py
  (Add --headless to run without viewport)
"""

import os
import sys
import numpy as np

# Initialize Isaac Sim SimulationApp
from isaacsim import SimulationApp

HEADLESS = "--headless" in sys.argv
NUM_FRAMES = 20 if "--test" in sys.argv else None
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
from envs.constants import WORKSPACE_HEIGHT
from randomisation import TabletopLightingRandomisation


def main():
    print("\n" + "=" * 80)
    print(">>> [Visualizer] Launching Tabletop Lighting Randomizer Inspector...")
    print("=" * 80 + "\n")

    # Setup physics simulation
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cuda")

    # Instantiate the lighting randomizer
    lighting_randomizer = TabletopLightingRandomisation(
        table_center=(0.25, 0.0, WORKSPACE_HEIGHT),
        ceiling_height=2.0,
        ceiling_jitter_xy=0.15,
    )

    # Instantiate the pick-and-place environment
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=None,
        enable_cameras=False,
        randomizers=[lighting_randomizer],
    )
    env.setup_scene()

    # Position viewport camera to view entire tabletop, robot arm, and cast shadows
    set_camera_view(
        eye=[-0.25, -0.40, 0.45],
        target=[0.22, 0.0, 0.08],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initialize handles
    env.initialize_after_play()

    # Sequence of states to demonstrate
    demo_states = ["night", "overcast", "sunny"]
    step_count = 0
    steps_per_state = 5 if NUM_FRAMES is not None else 120  # Fast cycle for test mode
    state_idx = 0

    print("\n" + "=" * 80)
    print(">>> Tabletop Lighting Visualizer Running")
    print(">>> Cycling through States [NIGHT -> OVERCAST -> SUNNY] every 2.0s.")
    print(">>> Notice:")
    print("      • NIGHT:    Pure white ceiling light, downward shadows, no sun.")
    print("      • OVERCAST: Soft diffuse window daylight + ceiling light.")
    print("      • SUNNY:    Ceiling light OFF, dramatic angled sunbeam & sharp shadows.")
    print(">>> Close the viewport window to exit.")
    print("=" * 80 + "\n")

    # Initial sample
    current_state = demo_states[state_idx]
    lighting_randomizer.step(state=current_state)

    total_steps = 0
    while simulation_app.is_running():
        simulation_app.update()

        if app_utils.is_playing() and SimulationManager.is_simulating():
            step_count += 1
            total_steps += 1

            if NUM_FRAMES is not None and total_steps >= NUM_FRAMES:
                print(f"[Visualizer] Completed test run of {NUM_FRAMES} frames.")
                break

            if step_count >= steps_per_state:
                step_count = 0
                state_idx = (state_idx + 1) % len(demo_states)
                current_state = demo_states[state_idx]
                lighting_randomizer.step(state=current_state)

    print("\n[Visualizer] Exiting...")
    simulation_app.close()


if __name__ == "__main__":
    main()
