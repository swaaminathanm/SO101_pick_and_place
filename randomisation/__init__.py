"""
Domain Randomization module for SO-101 Pick and Place.
Provides pure math and direct prim-based randomizers without Omniverse Replicator:
- LegoBrickRandomisation (for Lego brick)
- BinRandomisation (for Bin)
"""

from .LegoBrickRandomisation import LegoBrickRandomisation, LegoBrickRandomizer
from .BinRandomisation import BinRandomisation, BinRandomizer

# Backwards compatibility aliases
LegoReplicatorRandomizer = LegoBrickRandomisation
BinReplicatorRandomizer = BinRandomisation

__all__ = [
    "LegoBrickRandomisation",
    "LegoBrickRandomizer",
    "BinRandomisation",
    "BinRandomizer",
    "LegoReplicatorRandomizer",
    "BinReplicatorRandomizer",
]
