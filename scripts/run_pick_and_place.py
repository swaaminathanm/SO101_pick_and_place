"""
Main execution script for SO-101 Pick and Place environment.
Integrates SO101WorldInterface, LegoBrickPickPlaceEnv (with Camera Rig),
and SO101PickPlaceController for cuMotion RMPflow pick-and-place execution.
"""

import os
import sys
import numpy as np

# Parse command line flags before SimulationApp startup
HEADLESS = "--headless" in sys.argv
ENABLE_CAMERAS = "--no-cameras" not in sys.argv

# Initialize Isaac Sim SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": HEADLESS})

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from envs import LegoBrickPickPlaceEnv
from envs.constants import (
    ROBOT_PRIM_PATH,
    RED_BRICK_PRIM_PATH,
    BLUE_BRICK_PRIM_PATH,
    WALLS_PRIM_PATH,
    WALL_BACK_PRIM_PATH,
    WALL_RIGHT_PRIM_PATH,
)
from motion import SO101WorldInterface, SO101PickPlaceController, PickPlacePhase

SAVE_LIVE_PREVIEW: bool = True


def save_preview(image_array: np.ndarray, file_path: str):
    """Saves uint8 RGB image to disk."""
    if image_array is None:
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    Image.fromarray(image_array).save(file_path)


def main():
    print(f"\n[SO-101] Initializing simulation (Headless={HEADLESS}, Cameras={ENABLE_CAMERAS})...")

    # Setup GPU physics simulation
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cuda")

    # Create the World Interface instance using shared prim path constants
    world_interface = SO101WorldInterface(
        robot_prim_path=ROBOT_PRIM_PATH,
        exclude_prim_paths=[
            ROBOT_PRIM_PATH,
            RED_BRICK_PRIM_PATH,
            BLUE_BRICK_PRIM_PATH,
            WALLS_PRIM_PATH,
            WALL_BACK_PRIM_PATH,
            WALL_RIGHT_PRIM_PATH,
        ],
        search_box_min=[-10.0, -10.0, -10.0],
        search_box_max=[10.0, 10.0, 10.0],
        safety_margin=0.01,
    )

    # Inject world_interface and simulation_app into LegoBrickPickPlaceEnv
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=world_interface,
        enable_cameras=ENABLE_CAMERAS,
    )
    env.setup_scene()

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initializes joint handles, contact sensors, world interface, and cameras
    env.initialize_after_play()

    # Create and initialize the Pick and Place Controller
    controller = SO101PickPlaceController(
        articulation=env._articulation,
        world_interface=world_interface,
        robot_prim_path=ROBOT_PRIM_PATH,
        brick_center_offset=getattr(env, "brick_center_offset", None),
        half_width=getattr(env, "half_width", None),
        gripper_max_effort=0.5,
    )
    controller.initialize()

    # Initial workspace and domain reset
    env.reset_domain()
    controller.reset(episode_id=env.episode_id)
    step = 0
    preview_dir = os.path.join(PROJECT_ROOT, "camera_previews")

    print(
        f"\n>>> [SO-101] Running Pick & Place State Machine "
        f"({PickPlacePhase.ORIENT_DOWNWARD.name} -> {PickPlacePhase.PRE_GRASP_HOVER.name} -> "
        f"{PickPlacePhase.DESCEND_TO_GRASP.name} -> {PickPlacePhase.GRASP.name} -> "
        f"{PickPlacePhase.MOVE_UP.name} -> {PickPlacePhase.MOVE_TO_BIN.name} -> "
        f"{PickPlacePhase.DROP.name} -> {PickPlacePhase.POST_DROP_LIFT.name} -> "
        f"{PickPlacePhase.RETRACT_HOME.name}). Close window to exit.\n"
    )

    while simulation_app.is_running():
        simulation_app.update()

        # Step physics only while actively playing
        if app_utils.is_playing() and SimulationManager.is_simulating():
            step += 1

            # Keep cuMotion synchronized with all obstacle transforms every physics frame
            world_interface.sync()

            # Read live pose of the target Lego Brick
            brick_pos, brick_ori = env.lego_prims[0].get_world_poses()
            b_pos = brick_pos.numpy()[0] if hasattr(brick_pos, "numpy") else np.asarray(brick_pos)[0]
            b_ori = brick_ori.numpy()[0] if hasattr(brick_ori, "numpy") else np.asarray(brick_ori)[0]

            # Read live pose of the bin
            bin_pos, _ = env.bin_prim.get_world_poses()
            b_bin = bin_pos.numpy()[0] if hasattr(bin_pos, "numpy") else np.asarray(bin_pos)[0]

            # Step the controller (executing state machine phases)
            controller.step(b_pos, b_ori, b_bin)

            # Retrieve synchronized observations (joint states + camera images)
            if ENABLE_CAMERAS and step % 60 == 0:
                obs = env.get_observations()
                if SAVE_LIVE_PREVIEW:
                    save_preview(obs.get("overhead_rgb"), os.path.join(preview_dir, "live_overhead.png"))
                    save_preview(obs.get("wrist_rgb"), os.path.join(preview_dir, "live_wrist.png"))

            # Periodically reset domain to verify adaptation across randomized brick poses
            if step % 300 == 0 or controller.is_converged:
                if controller.is_converged:
                    print(f"\n>>> [SO-101] Pick & Place task completed successfully! Resetting domain for next episode...\n")
                else:
                    print(f"\n>>> [SO-101] Resetting domain for next episode...\n")

                env.reset_domain()
                controller.reset(episode_id=env.episode_id)
                step = 0

    simulation_app.close()


if __name__ == "__main__":
    main()
