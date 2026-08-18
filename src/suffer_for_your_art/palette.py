"""Named pigment palette for deterministic rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pigment:
    """Simple named pigment definition."""

    label: str
    rgb: tuple[int, int, int]


# Limited, explicit palette used by all painters.
PALETTE = {
    "carbon": Pigment("carbon", (17, 17, 17)),
    "umber": Pigment("umber", (97, 64, 39)),
    "ink": Pigment("ink", (48, 56, 67)),
    "alizarin": Pigment("alizarin", (180, 42, 45)),
    "emerald": Pigment("emerald", (26, 123, 73)),
    "azure": Pigment("azure", (43, 102, 175)),
    "ochre": Pigment("ochre", (188, 134, 37)),
}

PALETTE_ORDER = list(PALETTE.keys())

__all__ = ["Pigment", "PALETTE", "PALETTE_ORDER"]
