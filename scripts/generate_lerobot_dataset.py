"""
Generate Demonstration Dataset in Hugging Face LeRobot v2.0 Parquet Format
for SO-101 Pick-and-Place task in Isaac Sim.

Dataset Structure Created:
  <output_dir>/
  ├── meta/
  │   ├── info.json
  │   └── episodes.jsonl
  ├── data/
  │   └── chunk-000/
  │       ├── episode_000000.parquet
  │       ├── episode_000001.parquet
  │       └── ...
  └── videos/
      ├── observation.images.overhead/
      │   └── episode_000000.mp4
      └── observation.images.wrist/
          └── episode_000000.mp4
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import imageio.v2 as imageio

# Parse known CLI flags before SimulationApp startup
parser = argparse.ArgumentParser(description="SO-101 LeRobot Demonstration Dataset Generator")
parser.add_argument("--headless", action="store_true", default=True, help="Run simulation headlessly")
parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for parallel collection")
parser.add_argument("--num-episodes", type=int, default=10, help="Number of successful episodes to collect")
parser.add_argument("--max-steps", type=int, default=800, help="Max physics steps per episode")
parser.add_argument("--record-interval", type=int, default=2, help="Subsample rate (2 = 30Hz from 60Hz physics)")
parser.add_argument("--output-dir", type=str, default="./lerobot_dataset", help="Output directory for this worker")
args, _ = parser.parse_known_args()

# Initialize Isaac Sim SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if "scripts" in CURRENT_DIR else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pyarrow as pa
import pyarrow.parquet as pq
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
from motion import SO101WorldInterface, SO101PickPlaceController
from randomisation import (
    LegoBrickRandomisation,
    BinRandomisation,
    TabletopLightingRandomisation,
)


def write_video(frames: list[np.ndarray], output_mp4_path: str, fps: int = 30):
    """Encodes a list of RGB frames into an MP4 video using imageio (H.264)."""
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    if not frames:
        print(f"[Warning] No frames to write for {output_mp4_path}")
        return

    try:
        writer = imageio.get_writer(
            output_mp4_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-g", "15"],  # Fast random-access seeking for VLA dataloaders
        )
        for frame in frames:
            writer.append_data(frame.astype(np.uint8))
        writer.close()
        file_size_kb = os.path.getsize(output_mp4_path) / 1024.0
        print(f"  [Video] Written: {output_mp4_path} ({len(frames)} frames, {file_size_kb:.1f} KB)")
    except Exception as e:
        print(f"[Error writing video {output_mp4_path}]: {e}")


def main():
    # Distinct random seed per worker to guarantee diverse data
    worker_seed = 42 + args.worker_id * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    print(f"\n=======================================================")
    print(f"  [Worker {args.worker_id}] SO-101 LeRobot Dataset Generator")
    print(f"  Target Successful Episodes: {args.num_episodes}")
    print(f"  Random Seed:               {worker_seed}")
    print(f"  Output Directory:          {os.path.abspath(args.output_dir)}")
    print(f"=======================================================\n")

    # Physics rate is 60Hz
    SIM_DT = 1.0 / 60.0
    SimulationManager.setup_simulation(dt=SIM_DT, device="cuda")

    # Recording rate: 60Hz / record_interval (default 2 -> 30 FPS)
    FPS = int(1.0 / (SIM_DT * args.record_interval))

    # Initialize cuMotion World Interface
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

    # Domain randomizers
    brick_randomizer = LegoBrickRandomisation(brick_prim_path=RED_BRICK_PRIM_PATH)
    bin_randomizer = BinRandomisation(bin_prim_path=BIN_PRIM_PATH)
    lighting_randomizer = TabletopLightingRandomisation()
    randomizers = [brick_randomizer, bin_randomizer, lighting_randomizer]

    # Environment setup
    env = LegoBrickPickPlaceEnv(
        simulation_app=simulation_app,
        world_interface=world_interface,
        enable_cameras=True,
        randomizers=randomizers,
    )
    env.setup_scene()

    app_utils.play()
    simulation_app.update()
    env.initialize_after_play()

    # Warm up camera render textures so annotators have valid frames from step 1
    for _ in range(6):
        simulation_app.update()
        env.get_observations()

    # Controller setup
    controller = SO101PickPlaceController(
        articulation=env._articulation,
        world_interface=world_interface,
        robot_prim_path=ROBOT_PRIM_PATH,
        half_width=getattr(env, "half_width", None),
        gripper_max_effort=0.5,
    )
    controller.initialize()

    # Output directory layout
    data_dir = os.path.join(args.output_dir, "data", "chunk-000")
    overhead_video_dir = os.path.join(args.output_dir, "videos", "observation.images.overhead")
    wrist_video_dir = os.path.join(args.output_dir, "videos", "observation.images.wrist")
    meta_dir = os.path.join(args.output_dir, "meta")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(overhead_video_dir, exist_ok=True)
    os.makedirs(wrist_video_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # LeRobot Arrow Schema for Parquet
    parquet_schema = pa.schema([
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("observation.state", pa.list_(pa.float32())),
        ("observation.velocity", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("next.reward", pa.float32()),
        ("next.done", pa.bool_()),
        ("next.success", pa.bool_()),
    ])

    global_frame_index = 0
    successful_episodes = 0
    episodes_meta = []


    try:
        while successful_episodes < args.num_episodes and simulation_app.is_running():
            print(f"\n[Worker {args.worker_id}] >>> Starting Episode {successful_episodes + 1}/{args.num_episodes}...")
            env.reset_domain()
            controller.reset(episode_id=successful_episodes)

            ep_overhead_frames = []
            ep_wrist_frames = []
            ep_states = []
            ep_vels = []
            ep_actions = []
            ep_timestamps = []

            step = 0
            is_success = False

            while simulation_app.is_running() and step < args.max_steps:
                simulation_app.update()

                if not (app_utils.is_playing() and SimulationManager.is_simulating()):
                    continue

                step += 1
                world_interface.sync()

                # Live poses
                b_pos_raw, b_ori_raw = env.lego_prims[0].get_world_poses()
                bin_pos_raw, bin_ori_raw = env.bin_prim.get_world_poses()
                b_pos = b_pos_raw.numpy()[0] if hasattr(b_pos_raw, "numpy") else np.asarray(b_pos_raw)[0]
                b_ori = b_ori_raw.numpy()[0] if hasattr(b_ori_raw, "numpy") else np.asarray(b_ori_raw)[0]
                b_bin = bin_pos_raw.numpy()[0] if hasattr(bin_pos_raw, "numpy") else np.asarray(bin_pos_raw)[0]
                b_bin_ori = bin_ori_raw.numpy()[0] if hasattr(bin_ori_raw, "numpy") else np.asarray(bin_ori_raw)[0]

                # Step controller
                controller.step(b_pos, b_ori, b_bin)

                # Downsampled recording
                if step % args.record_interval == 0:
                    obs = env.get_observations()
                    joint_pos = obs.get("joint_positions")
                    joint_vel = obs.get("joint_velocities")
                    ov_rgb = obs.get("overhead_rgb")
                    wr_rgb = obs.get("wrist_rgb")

                    # Only record when robot state AND both cameras produced valid frames
                    if joint_pos is None or ov_rgb is None or wr_rgb is None:
                        continue

                    # Commanded targets
                    target_joints = env._articulation.get_dof_position_targets()
                    if target_joints is None:
                        target_joints = joint_pos

                    t_pos = target_joints.numpy() if hasattr(target_joints, "numpy") else np.asarray(target_joints)
                    q_pos = joint_pos.flatten().tolist()
                    q_vel = joint_vel.flatten().tolist() if joint_vel is not None else [0.0] * len(q_pos)
                    act = t_pos.flatten().tolist()

                    # Append all synchronized modalities together
                    ep_states.append(q_pos)
                    ep_vels.append(q_vel)
                    ep_actions.append(act)
                    ep_timestamps.append(float(len(ep_timestamps) * (1.0 / FPS)))
                    ep_overhead_frames.append(ov_rgb[:, :, :3])
                    ep_wrist_frames.append(wr_rgb[:, :, :3])

                # Check task completion
                is_in_bin = env.is_brick_in_bin(brick_pos=b_pos, bin_pos=b_bin, bin_ori=b_bin_ori)
                arm_home = controller.is_phase_converged(controller.phase.RETRACT_HOME)

                if is_in_bin and arm_home:
                    is_success = True
                    break

            # Filter: Only keep successful episodes
            if is_success and len(ep_states) > 10:
                ep_idx = successful_episodes
                ep_length = len(ep_states)

                timestamps = np.array(ep_timestamps, dtype=np.float32)
                frame_indices = np.arange(ep_length, dtype=np.int64)
                episode_indices = np.full(ep_length, ep_idx, dtype=np.int64)
                global_indices = np.arange(global_frame_index, global_frame_index + ep_length, dtype=np.int64)

                rewards = np.zeros(ep_length, dtype=np.float32)
                rewards[-1] = 1.0

                dones = np.zeros(ep_length, dtype=bool)
                dones[-1] = True

                successes = np.ones(ep_length, dtype=bool)

                table_dict = {
                    "timestamp": timestamps,
                    "frame_index": frame_indices,
                    "episode_index": episode_indices,
                    "index": global_indices,
                    "observation.state": ep_states,
                    "observation.velocity": ep_vels,
                    "action": ep_actions,
                    "next.reward": rewards,
                    "next.done": dones,
                    "next.success": successes,
                }

                # 1. Save Parquet
                parquet_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
                table = pa.Table.from_pydict(table_dict, schema=parquet_schema)
                pq.write_table(table, parquet_path)

                # 2. Save Videos (H.264 via imageio)
                overhead_video_path = os.path.join(overhead_video_dir, f"episode_{ep_idx:06d}.mp4")
                wrist_video_path = os.path.join(wrist_video_dir, f"episode_{ep_idx:06d}.mp4")
                write_video(ep_overhead_frames, overhead_video_path, fps=FPS)
                write_video(ep_wrist_frames, wrist_video_path, fps=FPS)

                # 3. Record Metadata
                episodes_meta.append({
                    "episode_index": ep_idx,
                    "tasks": ["Pick up the red Lego brick and place it into the bin."],
                    "length": ep_length,
                })

                global_frame_index += ep_length
                successful_episodes += 1
                print(f"  [Worker {args.worker_id}] [SUCCESS] Episode {ep_idx:06d} saved ({ep_length} frames) -> {parquet_path}")
            else:
                print(f"  [Worker {args.worker_id}] [DISCARDED] Task incomplete within step limit. Discarded.")

    finally:
        # Write worker metadata
        if episodes_meta:
            with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
                for meta in episodes_meta:
                    f.write(json.dumps(meta) + "\n")

            info = {
                "codebase_version": "v2.0",
                "robot_type": "so101",
                "total_episodes": successful_episodes,
                "total_frames": global_frame_index,
                "total_tasks": 1,
                "fps": FPS,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [6],
                        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                    },
                    "observation.velocity": {
                        "dtype": "float32",
                        "shape": [6],
                        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [6],
                        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                    },
                    "observation.images.overhead": {
                        "dtype": "video",
                        "shape": [720, 1280, 3],
                        "names": ["height", "width", "channels"],
                    },
                    "observation.images.wrist": {
                        "dtype": "video",
                        "shape": [720, 1280, 3],
                        "names": ["height", "width", "channels"],
                    },
                },
            }
            with open(os.path.join(meta_dir, "info.json"), "w") as f:
                json.dump(info, f, indent=2)

            print(f"\n[Worker {args.worker_id}] Finished! Total Successful Episodes: {successful_episodes}")

        simulation_app.close()


if __name__ == "__main__":
    main()
