"""
Standalone visualizer script for the SO-101 pick-and-place manipulator and controller.
Initializes LegoBrickPickPlaceEnv with LegoBrickRandomisation (0 Y rotation, resting flat)
and default bin from inside the env.

Run with:
  ./run.sh scripts/visualize_controller.py
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
from envs.constants import (
    ROBOT_PRIM_PATH,
    RED_BRICK_PRIM_PATH,
    BLUE_BRICK_PRIM_PATH,
    BIN_PRIM_PATH,
    WALLS_PRIM_PATH,
    WALL_BACK_PRIM_PATH,
    WALL_RIGHT_PRIM_PATH,
    WALL_FRONT_PRIM_PATH,
    WALL_CEILING_PRIM_PATH,
    DEBUG_DRAW,
)
from motion import SO101WorldInterface, SO101PickPlaceController
from randomisation import LegoBrickRandomisation


def main():
    print("\n[Visualizer] Launching SO-101 Controller Visualizer...")

    # Setup physics simulation
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cuda")

    # Create cuMotion World Interface instance
    world_interface = SO101WorldInterface(
        robot_prim_path=ROBOT_PRIM_PATH,
        exclude_prim_paths=[
            ROBOT_PRIM_PATH,
            RED_BRICK_PRIM_PATH,
            BLUE_BRICK_PRIM_PATH,
            WALLS_PRIM_PATH,
            WALL_BACK_PRIM_PATH,
            WALL_RIGHT_PRIM_PATH,
            WALL_FRONT_PRIM_PATH,
            WALL_CEILING_PRIM_PATH,
        ],
        search_box_min=[-10.0, -10.0, -10.0],
        search_box_max=[10.0, 10.0, 10.0],
        safety_margin=0.01,
    )

    # Lego brick randomizer with 0 Y rotation (flat on table, no tilt)
    brick_randomizer = LegoBrickRandomisation(
        brick_prim_path=RED_BRICK_PRIM_PATH,
        y_rot_choices=[0, 90, 180, 270],
        x_rot_choices=[0, 90, 180, 270],
        z_rot_range=(0.0, 360.0),
        randomize_color=False,
    )

    # Instantiate the pick-and-place environment (bin etc. default from inside env)
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=world_interface,
        enable_cameras=False,
        randomizers=[brick_randomizer],
        show_walls=False,
    )
    env.setup_scene()
    env.set_walls_visibility(False)

    # Default perspective camera
    set_camera_view(
        eye=[-0.35, 0.0, 0.40],
        target=[0.22, 0.0, 0.08],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initialize joint handles and world interface
    env.initialize_after_play()

    if getattr(brick_randomizer, "debug_draw", False):
        brick_randomizer.draw_bounding_box()
    else:
        brick_randomizer.clear_bounding_box()

    # Create and initialize the controller
    controller = SO101PickPlaceController(
        articulation=env._articulation,
        world_interface=world_interface,
        robot_prim_path=ROBOT_PRIM_PATH,
        half_width=getattr(env, "half_width", None),
        gripper_max_effort=0.5,
    )
    controller.initialize()

    # Initial workspace and domain reset
    env.reset_domain()
    controller.reset(episode_id=env.episode_id)
    step = 0

    print("\n" + "=" * 80)
    print(">>> SO-101 Controller Visualizer Running")
    print(">>> Close the viewport window to exit.")
    print("=" * 80 + "\n")

    while simulation_app.is_running():
        simulation_app.update()

        if app_utils.is_playing() and SimulationManager.is_simulating():
            step += 1

            # Keep cuMotion synchronized with all obstacle transforms every physics frame
            world_interface.sync()

            # Read live pose of the target Lego Brick
            brick_pos, brick_ori = env.lego_prims[0].get_world_poses()
            b_pos = brick_pos.numpy()[0] if hasattr(brick_pos, "numpy") else np.asarray(brick_pos)[0]
            b_ori = brick_ori.numpy()[0] if hasattr(brick_ori, "numpy") else np.asarray(brick_ori)[0]

            # Read live pose of the bin
            bin_pos, bin_ori = env.bin_prim.get_world_poses()
            b_bin = bin_pos.numpy()[0] if hasattr(bin_pos, "numpy") else np.asarray(bin_pos)[0]
            b_bin_ori = bin_ori.numpy()[0] if hasattr(bin_ori, "numpy") else np.asarray(bin_ori)[0]

            # Step the controller
            controller.step(b_pos, b_ori, b_bin)

            # Check if task succeeded (brick physically inside bin) and if arm finished cycle
            is_success = env.is_brick_in_bin(brick_pos=b_pos, bin_pos=b_bin, bin_ori=b_bin_ori)
            arm_cycle_complete = controller.is_phase_converged(controller.phase.RETRACT_HOME)

            # Periodically reset domain (default: 300 steps or task converged)
            if (arm_cycle_complete and is_success) or step % 300 == 0:
                if is_success:
                    print(f"\n>>> [SO-101] Pick & Place task completed successfully (brick inside bin)! Resetting domain for next episode...\n")
                else:
                    print(f"\n>>> [SO-101] Episode ended without successful placement. Resetting domain for next episode...\n")

                env.reset_domain()
                controller.reset(episode_id=env.episode_id)
                step = 0

    simulation_app.close()


if __name__ == "__main__":
    main()
