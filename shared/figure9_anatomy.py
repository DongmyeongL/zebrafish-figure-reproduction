"""Canonical zebrafish root-area grouping for Figure 9."""

from __future__ import annotations


ANATOMY_GROUP_ORDER = ("Tel", "Di", "Mes", "Hind")

_ANATOMY_GROUPS = {
    "Tel": frozenset({"P", "SP", "OB", "OE", "OG", "PO"}),
    "Di": frozenset({"HB", "HC", "HI", "HR", "TH", "PT", "PRT"}),
    "Mes": frozenset({"R", "TEO", "TL", "TS", "T"}),
    "Hind": frozenset(
        {
            "CB", "IPN", "IO", "MON", "MOS1", "MOS2", "MOS3", "MOS4",
            "MOS5", "RA", "ARF", "PRF", "IMRF", "VR", "NX", "TG",
        }
    ),
}

_GROUP_BY_ROOT_AREA = {
    root_area: group
    for group, root_areas in _ANATOMY_GROUPS.items()
    for root_area in root_areas
}

_SIDE_ALIASES = {
    "raRF": "aRF",
    "rimRF": "imRF",
    "rpRF": "pRF",
}


def strip_side(node: str) -> str:
    """Return the side-independent root-area label used for grouping."""
    name = str(node).strip()
    if name in _SIDE_ALIASES:
        return _SIDE_ALIASES[name]
    if len(name) > 1 and name[0] in {"l", "r"} and name[1].isupper():
        return name[1:]
    return name


def anatomy_group(node: str) -> str:
    """Map a zebrafish root-area label to Tel, Di, Mes, or Hind."""
    root_area = strip_side(node).upper()
    try:
        return _GROUP_BY_ROOT_AREA[root_area]
    except KeyError as exc:
        raise ValueError(f"Unrecognized zebrafish root-area label: {node!r}") from exc

