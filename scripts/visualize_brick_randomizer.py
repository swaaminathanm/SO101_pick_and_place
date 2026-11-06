"""
Standalone visualizer script for inspecting Lego brick combined Y-axis (tilt/resting face pitch: 0, 90, 180, 270 deg)
and Z-axis (tabletop yaw: 0-360 deg) rotations, position randomization in the -Y quadrant, and color variations
using LegoBrickRandomisation in Isaac Sim without robot movement.

Run with:
  ./run.sh scripts/visualize_brick_randomizer.py
"""

from scripts.visualize_brick_rotation import main

if __name__ == "__main__":
    main()
