"""
Main execution script for SO-101 Pick and Place environment.
Integrates SO101WorldInterface, RandomizedSO101Env, and SO101PickPlaceController
for cuMotion RMPflow top-down reaching (Phase 0: Pre-Grasp).
"""

import os
import sys
import numpy as np

# Initialize Isaac Sim SimulationApp first
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import Isaac Sim, Motion, and Environment modules
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.experimental.utils.stage as stage_utils
from envs import LegoBrickPickPlaceEnv, SimpleSO101Env
from envs.simple_pick_place_env import (
    ROBOT_PRIM_PATH,
    RED_BRICK_PRIM_PATH,
    BLUE_BRICK_PRIM_PATH,
)
from motion import SO101WorldInterface, SO101PickPlaceController, PickPlacePhase

# Environment selection: True for deterministic simple cube environment, False for randomized Lego brick
USE_SIMPLE_ENV: bool = False

def main():
    # Setup GPU physics simulation
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cuda")

    # Create the World Interface instance using shared prim path constants
    world_interface = SO101WorldInterface(
        robot_prim_path=ROBOT_PRIM_PATH,
        exclude_prim_paths=[
            ROBOT_PRIM_PATH,
            RED_BRICK_PRIM_PATH,
            BLUE_BRICK_PRIM_PATH,
        ],
        search_box_min=[-10.0, -10.0, -10.0],
        search_box_max=[10.0, 10.0, 10.0],
        safety_margin=0.01,
    )

    # Inject world_interface and simulation_app into chosen Environment
    EnvClass = SimpleSO101Env if USE_SIMPLE_ENV else LegoBrickPickPlaceEnv
    env = EnvClass(
        simulation_app=simulation_app,
        world_interface=world_interface
    )
    env.setup_scene()

    # Start simulation playback
    app_utils.play()
    simulation_app.update()

    # Initializes joint handles and world_interface with the robot's base pose
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

    print(f"\n>>> [SO-101] Running Pick & Place State Machine ({PickPlacePhase.ORIENT_DOWNWARD.name} -> {PickPlacePhase.PRE_GRASP_HOVER.name} -> {PickPlacePhase.DESCEND_TO_GRASP.name} -> {PickPlacePhase.GRASP.name} -> {PickPlacePhase.MOVE_UP.name} -> {PickPlacePhase.HOLD.name}). Close window to exit.\n")

    while simulation_app.is_running():
        simulation_app.update()

        # Step physics only while actively playing
        if app_utils.is_playing() and SimulationManager.is_simulating():
            step += 1

            # Keep cuMotion synchronized with all obstacle transforms every physics frame
            world_interface.sync()

            # Read live pose of the target Red Lego Brick
            brick_pos, brick_ori = env.lego_prims[0].get_world_poses()
            b_pos = brick_pos.numpy()[0] if hasattr(brick_pos, "numpy") else np.asarray(brick_pos)[0]
            b_ori = brick_ori.numpy()[0] if hasattr(brick_ori, "numpy") else np.asarray(brick_ori)[0]

            # Step the controller (executing 4-phase approach)
            controller.step(b_pos, b_ori)

            # Periodically reset domain to verify adaptation across randomized brick poses
            if step % 240 == 0:
                env.reset_domain()
                controller.reset(episode_id=env.episode_id)

    simulation_app.close()


if __name__ == "__main__":
    main()
