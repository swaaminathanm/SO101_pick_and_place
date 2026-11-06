"""scripts/recenter_brick_usd.py
Recenters Lego brick USD assets so (0,0,0) is at the geometric centroid.
"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import shutil
from pxr import Usd, UsdGeom, Gf

USD_FILES = ["assets/lego_brick_blue.usd", "assets/lego_brick_red.usd"]

for path in USD_FILES:
    # Backup original
    shutil.copyfile(path, path + ".bak")

    # Open stage
    stage = Usd.Stage.Open(path)
    brick_prim = stage.GetPrimAtPath("/World/LegoBrick")

    # Compute centroid
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = bbox_cache.ComputeLocalBound(brick_prim).GetBox()
    center = (box.GetMin() + box.GetMax()) / 2.0

    # Shift /World/LegoBrick by -center
    xf = UsdGeom.Xformable(brick_prim)
    xf.GetTranslateOp().Set(-center)

    # Save modified USD
    stage.GetRootLayer().Save()
    print(f"[✓] Centered {path} (Shifted by {-center})")

app.close()