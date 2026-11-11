"""
Consolidates parallel worker outputs into a single unified Hugging Face LeRobot dataset.
Renurs episodes sequentially, ensures continuous global frame indexing,
merges metadata, and places files into standard LeRobot v2.0 structure.
"""

import os
import sys
import json
import shutil
import argparse
import glob
import pyarrow as pa
import pyarrow.parquet as pq


def consolidate(output_base: str, keep_worker_dirs: bool = False):
    output_base = os.path.abspath(output_base)
    worker_dirs = sorted(glob.glob(os.path.join(output_base, "worker_*")))

    if not worker_dirs:
        print(f"[Consolidate] No worker directories found in {output_base}. Nothing to consolidate.")
        return

    print(f"\n=======================================================")
    print(f"  Consolidating {len(worker_dirs)} Parallel Workers")
    print(f"  Target Unified Directory: {output_base}")
    print(f"=======================================================\n")

    # Final unified directories
    final_data_dir = os.path.join(output_base, "data", "chunk-000")
    final_overhead_dir = os.path.join(output_base, "videos", "observation.images.overhead")
    final_wrist_dir = os.path.join(output_base, "videos", "observation.images.wrist")
    final_meta_dir = os.path.join(output_base, "meta")

    os.makedirs(final_data_dir, exist_ok=True)
    os.makedirs(final_overhead_dir, exist_ok=True)
    os.makedirs(final_wrist_dir, exist_ok=True)
    os.makedirs(final_meta_dir, exist_ok=True)

    global_ep_idx = 0
    global_frame_idx = 0
    merged_episodes_meta = []
    fps = 30

    for w_dir in worker_dirs:
        w_name = os.path.basename(w_dir)
        w_data_dir = os.path.join(w_dir, "data", "chunk-000")
        w_meta_file = os.path.join(w_dir, "meta", "episodes.jsonl")
        w_info_file = os.path.join(w_dir, "meta", "info.json")

        if os.path.exists(w_info_file):
            try:
                with open(w_info_file, "r") as f:
                    w_info = json.load(f)
                    fps = w_info.get("fps", fps)
            except Exception:
                pass

        parquet_files = sorted(glob.glob(os.path.join(w_data_dir, "episode_*.parquet")))
        print(f"[{w_name}] Found {len(parquet_files)} episodes.")

        for p_file in parquet_files:
            orig_ep_name = os.path.basename(p_file) # e.g. episode_000000.parquet
            ep_stem = os.path.splitext(orig_ep_name)[0]

            # Read parquet table to update episode_index and global index
            table = pq.read_table(p_file)
            pydict = table.to_pydict()
            ep_len = len(pydict["timestamp"])

            # Renumber episode_index and global index
            pydict["episode_index"] = [global_ep_idx] * ep_len
            pydict["index"] = list(range(global_frame_idx, global_frame_idx + ep_len))

            new_table = pa.Table.from_pydict(pydict, schema=table.schema)
            new_p_name = f"episode_{global_ep_idx:06d}.parquet"
            new_p_path = os.path.join(final_data_dir, new_p_name)
            pq.write_table(new_table, new_p_path)

            # Move/Copy videos
            for cam_key, target_dir in [
                ("observation.images.overhead", final_overhead_dir),
                ("observation.images.wrist", final_wrist_dir),
            ]:
                src_vid = os.path.join(w_dir, "videos", cam_key, f"{ep_stem}.mp4")
                dst_vid = os.path.join(target_dir, f"episode_{global_ep_idx:06d}.mp4")
                if os.path.exists(src_vid):
                    shutil.move(src_vid, dst_vid)

            # Track metadata
            merged_episodes_meta.append({
                "episode_index": global_ep_idx,
                "tasks": ["Pick up the red Lego brick and place it into the bin."],
                "length": ep_len,
            })

            global_frame_idx += ep_len
            global_ep_idx += 1

    # Write merged metadata
    with open(os.path.join(final_meta_dir, "episodes.jsonl"), "w") as f:
        for meta in merged_episodes_meta:
            f.write(json.dumps(meta) + "\n")

    info = {
        "codebase_version": "v2.0",
        "robot_type": "so101",
        "total_episodes": global_ep_idx,
        "total_frames": global_frame_idx,
        "total_tasks": 1,
        "fps": fps,
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
    with open(os.path.join(final_meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Clean up worker directories if requested
    if not keep_worker_dirs:
        for w_dir in worker_dirs:
            shutil.rmtree(w_dir, ignore_errors=True)

    print(f"\n=======================================================")
    print(f"  [SUCCESS] Dataset Consolidation Finished!")
    print(f"  Total Episodes Consolidated: {global_ep_idx}")
    print(f"  Total Frames Indexed:        {global_frame_idx}")
    print(f"  Unified Dataset Directory:   {output_base}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate parallel worker datasets into LeRobot format")
    parser.add_argument("--output-base", type=str, default="./lerobot_dataset", help="Dataset root directory")
    parser.add_argument("--keep-workers", action="store_true", help="Keep temporary worker directories")
    args = parser.parse_args()

    consolidate(args.output_base, keep_worker_dirs=args.keep_workers)
