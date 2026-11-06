"""
Domain Randomization module for SO-101 Pick and Place.
Provides pure math and direct prim-based randomizers without Omniverse Replicator:
- LegoBrickRandomisation (for Lego brick)
- BinRandomisation (for Bin)
"""

from .LegoBrickRandomisation import LegoBrickRandomisation
from .BinRandomisation import BinRandomisation, BinRandomizer
from .TabletopLightingRandomisation import TabletopLightingRandomisation

# Backwards compatibility aliases
LegoReplicatorRandomizer = LegoBrickRandomisation
BinReplicatorRandomizer = BinRandomisation

__all__ = [
    "LegoBrickRandomisation",
    "BinRandomisation",
    "BinRandomizer",
    "TabletopLightingRandomisation",
    "LegoReplicatorRandomizer",
    "BinReplicatorRandomizer",
]
