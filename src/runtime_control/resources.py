"""Stable access to terrain resources shipped inside the Python package."""

from pathlib import Path

from .map_manager import MapSpec


PACKAGE_DIR = Path(__file__).resolve().parent
MAPS_DIR = PACKAGE_DIR / "maps"

# Keep file names and import-specific exclusions private so downstream players
# only depend on stable terrain names.
_BUNDLED_MAPS = {
    "rc26_track": ("26rc_track.xml", ("trunk",)),
    "race_track": ("race_track.xml", ()),
    "stairs": ("stairs.xml", ()),
    "cross_stairs": ("cross_stairs.xml", ()),
    "cross_slope": ("cross_slope.xml", ()),
    "google_barkour": ("google_barkour.xml", ()),
    "gap_jump": ("gap_jump.xml", ()),
    "hurdles": ("hurdles.xml", ()),
    "suspended_steps": ("suspended_steps.xml", ()),
    "perlin_rough": ("perlin_rough.xml", ()),
    "dynamic_obstacles": ("dynamic_obstacles.xml", ()),
}


def available_bundled_maps():
    """Return the stable names of terrains included in the distribution."""
    return tuple(_BUNDLED_MAPS)


def bundled_map_path(name):
    """Return an installed terrain path, raising a useful error for bad names."""
    try:
        file_name = _BUNDLED_MAPS[str(name)][0]
    except KeyError as exc:
        choices = ", ".join(available_bundled_maps())
        raise KeyError("unknown bundled map %r; choose from: %s" % (name, choices)) from exc
    path = MAPS_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(
            "bundled map resource is missing: %s; reinstall mujoco-runtime-control"
            % path
        )
    return path


def bundled_map_spec(name, **overrides):
    """Create a :class:`MapSpec` using package-owned resource metadata."""
    name = str(name)
    try:
        _, exclusions = _BUNDLED_MAPS[name]
    except KeyError:
        bundled_map_path(name)  # raises the detailed public error
        raise
    values = {"exclude_bodies": exclusions}
    values.update(overrides)
    return MapSpec(bundled_map_path(name), **values)


def bundled_map_specs(names=None):
    """Return ``{name: MapSpec}`` for all or a selected set of bundled maps."""
    selected = available_bundled_maps() if names is None else tuple(names)
    return {name: bundled_map_spec(name) for name in selected}
