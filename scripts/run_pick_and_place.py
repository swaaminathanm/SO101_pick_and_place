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
    BIN_PRIM_PATH,
    WALLS_PRIM_PATH,
    WALL_BACK_PRIM_PATH,
    WALL_RIGHT_PRIM_PATH,
    WALL_FRONT_PRIM_PATH,
    WALL_CEILING_PRIM_PATH,
)
from motion import SO101WorldInterface, SO101PickPlaceController, PickPlacePhase
from randomisation import (
    LegoBrickRandomisation,
    BinRandomisation,
    TabletopLightingRandomisation,
)

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
            WALL_FRONT_PRIM_PATH,
            WALL_CEILING_PRIM_PATH,
        ],
        search_box_min=[-10.0, -10.0, -10.0],
        search_box_max=[10.0, 10.0, 10.0],
        safety_margin=0.01,
    )

    # Domain Randomizers created from outside (pure math + direct prim updates)
    brick_randomizer = LegoBrickRandomisation(
        brick_prim_path=RED_BRICK_PRIM_PATH,
    )
    bin_randomizer = BinRandomisation(
        bin_prim_path=BIN_PRIM_PATH,
    )
    lighting_randomizer = TabletopLightingRandomisation()

    randomizers = [brick_randomizer, bin_randomizer, lighting_randomizer]

    # Inject world_interface, simulation_app, and randomizers list into LegoBrickPickPlaceEnv
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=world_interface,
        enable_cameras=ENABLE_CAMERAS,
        randomizers=randomizers,
    )
    env.setup_scene()

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initializes joint handles, contact sensors, world interface, and cameras
    env.initialize_after_play()

    for r in randomizers:
        r.draw_bounding_box()  # Draw debug visualization for each randomizer

    # Create and initialize the Pick and Place Controller
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
    preview_dir = os.path.join(PROJECT_ROOT, "camera_previews")

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
            bin_pos, bin_ori = env.bin_prim.get_world_poses()
            b_bin = bin_pos.numpy()[0] if hasattr(bin_pos, "numpy") else np.asarray(bin_pos)[0]
            b_bin_ori = bin_ori.numpy()[0] if hasattr(bin_ori, "numpy") else np.asarray(bin_ori)[0]

            # Step the controller (executing state machine phases)
            controller.step(b_pos, b_ori, b_bin)

            # Retrieve synchronized observations (joint states + camera images)
            if ENABLE_CAMERAS and step % 60 == 0:
                obs = env.get_observations()
                if SAVE_LIVE_PREVIEW:
                    save_preview(obs.get("overhead_rgb"), os.path.join(preview_dir, "live_overhead.png"))
                    save_preview(obs.get("wrist_rgb"), os.path.join(preview_dir, "live_wrist.png"))

            # Check if task succeeded (brick physically inside bin) and if arm finished cycle
            is_success = env.is_brick_in_bin(brick_pos=b_pos, bin_pos=b_bin, bin_ori=b_bin_ori)
            arm_cycle_complete = controller.is_phase_converged(controller.phase.RETRACT_HOME)

            # Periodically reset domain to verify adaptation across randomized brick poses
            if (arm_cycle_complete and is_success) or step % 300 == 0:
                if is_success:
                    print(f"\n>>> [SO-101] Pick & Place task completed successfully (brick inside bin)! Resetting domain for next episode...\n")
                else:
                    print(f"\n>>> [SO-101] Episode finished without successful placement. Resetting domain for next episode...\n")

                env.reset_domain()
                controller.reset(episode_id=env.episode_id)
                step = 0

    simulation_app.close()


if __name__ == "__main__":
    main()
