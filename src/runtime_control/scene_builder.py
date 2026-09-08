"""Serializable, deterministic terrain and obstacle scene builder.

The builder deliberately produces terrain-only MJCF.  Robot composition,
policy execution and runtime controls remain owned by the existing runtime.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


TERRAIN_KINDS = ("flat", "slope", "stairs", "noise")
OBSTACLE_KINDS = (
    "platform", "wall", "stairs", "gap", "stepping_stones", "slalom",
    "ramp", "side_slope", "speed_bumps", "hurdles", "narrow_bridge",
    "wave_ground", "uneven_stairs", "random_blocks", "seesaw",
    "rotating_bar",
)


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % label)
    return value


def _positive(params, key, default, allow_zero=False):
    value = _finite(params.get(key, default), key)
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        raise ValueError("%s must be %s" % (key, "non-negative" if allow_zero else "positive"))
    return value


@dataclass
class TerrainSpec:
    """Reproducible base terrain configuration."""

    kind: str = "flat"
    length: float = 12.0
    width: float = 8.0
    height: float = 0.35
    rows: int = 128
    cols: int = 128
    seed: int = 0
    roughness: float = 0.18
    smoothness: int = 8
    stair_count: int = 8

    def __post_init__(self):
        self.kind = str(self.kind).strip().lower()
        if self.kind not in TERRAIN_KINDS:
            raise ValueError("terrain kind must be one of %s" % (TERRAIN_KINDS,))
        self.length = _positive(vars(self), "length", 12.0)
        self.width = _positive(vars(self), "width", 8.0)
        self.height = _positive(vars(self), "height", 0.35, allow_zero=True)
        self.roughness = _positive(vars(self), "roughness", 0.18, allow_zero=True)
        self.rows, self.cols = int(self.rows), int(self.cols)
        self.smoothness, self.stair_count = int(self.smoothness), int(self.stair_count)
        if self.rows < 2 or self.cols < 2:
            raise ValueError("terrain rows and cols must be at least 2")
        if self.smoothness < 1 or self.stair_count < 1:
            raise ValueError("smoothness and stair_count must be positive")


@dataclass
class ObstacleSpec:
    """One parameterized obstacle placed in world coordinates."""

    kind: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        self.kind = str(self.kind).strip().lower()
        if self.kind not in OBSTACLE_KINDS:
            raise ValueError("obstacle kind must be one of %s" % (OBSTACLE_KINDS,))
        for key in ("x", "y", "z", "yaw"):
            setattr(self, key, _finite(getattr(self, key), "obstacle %s" % key))
        if not isinstance(self.params, dict):
            raise TypeError("obstacle params must be a dictionary")


@dataclass
class SceneSpec:
    """Portable source representation for a generated runtime map."""

    name: str = "generated_scene"
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    obstacles: List[ObstacleSpec] = field(default_factory=list)
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.45)
    spawn_quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.terrain, dict):
            self.terrain = TerrainSpec(**self.terrain)
        self.obstacles = [
            item if isinstance(item, ObstacleSpec) else ObstacleSpec(**item)
            for item in self.obstacles
        ]
        if not self.name or not str(self.name).replace("_", "").replace("-", "").isalnum():
            raise ValueError("scene name must contain only letters, digits, '_' or '-'")
        if len(self.spawn_position) != 3 or len(self.spawn_quaternion) != 4:
            raise ValueError("spawn position/quaternion must have 3/4 values")
        self.spawn_position = tuple(_finite(v, "spawn position") for v in self.spawn_position)
        self.spawn_quaternion = tuple(_finite(v, "spawn quaternion") for v in self.spawn_quaternion)
        norm = math.sqrt(sum(value * value for value in self.spawn_quaternion))
        if norm < 1e-8:
            raise ValueError("spawn quaternion cannot be zero")
        self.spawn_quaternion = tuple(value / norm for value in self.spawn_quaternion)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))


def save_scene_spec(scene, path):
    """Save a scene source document that can be edited and regenerated."""
    if not isinstance(scene, SceneSpec):
        scene = SceneSpec.from_dict(scene)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scene.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_scene_spec(path):
    path = Path(path).expanduser().resolve()
    return SceneSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _smooth_noise(rows, cols, rng, smoothness):
    grid_rows = max(2, rows // smoothness + 2)
    grid_cols = max(2, cols // smoothness + 2)
    coarse = rng.random((grid_rows, grid_cols))
    old_x, new_x = np.linspace(0, 1, grid_cols), np.linspace(0, 1, cols)
    horizontal = np.stack([np.interp(new_x, old_x, row) for row in coarse])
    old_y, new_y = np.linspace(0, 1, grid_rows), np.linspace(0, 1, rows)
    return np.stack([np.interp(new_y, old_y, horizontal[:, col]) for col in range(cols)], axis=1)


def generate_heightfield(spec):
    """Return normalized float32 heights; identical specs produce identical data."""
    if isinstance(spec, dict):
        spec = TerrainSpec(**spec)
    rows, cols = spec.rows, spec.cols
    progress = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    if spec.kind == "flat":
        heights = np.zeros((rows, cols), dtype=np.float32)
    elif spec.kind == "slope":
        heights = np.broadcast_to(progress, (rows, cols)).copy()
    elif spec.kind == "stairs":
        steps = np.floor(progress * spec.stair_count) / max(spec.stair_count - 1, 1)
        heights = np.broadcast_to(np.clip(steps, 0, 1), (rows, cols)).copy()
    else:
        rng = np.random.default_rng(spec.seed)
        broad = _smooth_noise(rows, cols, rng, spec.smoothness)
        detail = _smooth_noise(rows, cols, rng, max(1, spec.smoothness // 3))
        weight = min(1.0, spec.roughness)
        heights = (1.0 - weight) * broad + weight * detail
        span = float(np.ptp(heights))
        heights = np.zeros_like(heights) if span < 1e-9 else (heights - heights.min()) / span
    return np.asarray(np.clip(heights, 0, 1), dtype=np.float32)


def _fmt(values):
    return " ".join("%.8g" % float(value) for value in values)


def _box(parent, name, center, half_size, yaw, rgba="0.32 0.48 0.68 1"):
    return ET.SubElement(parent, "geom", {
        "name": name, "type": "box", "pos": _fmt(center),
        "size": _fmt(half_size), "euler": _fmt((0, 0, yaw)),
        "rgba": rgba,
    })


def _local_xy(obstacle, forward, lateral=0.0):
    cosine, sine = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
    return (
        obstacle.x + forward * cosine - lateral * sine,
        obstacle.y + forward * sine + lateral * cosine,
    )


def _axis_angle(axis, angle):
    half = angle / 2.0
    sine = math.sin(half)
    return (math.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine)


def _quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw*rw - lx*rx - ly*ry - lz*rz,
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
    )


def _yaw_tilt_quat(yaw, axis, tilt):
    return _quat_multiply(_axis_angle((0, 0, 1), yaw), _axis_angle(axis, tilt))


def _append_obstacle(worldbody, obstacle, index):
    prefix = obstacle.name or "%s_%d" % (obstacle.kind, index)
    params = obstacle.params
    if obstacle.kind in {"platform", "wall"}:
        defaults = (1.0, 1.0, 0.3) if obstacle.kind == "platform" else (0.15, 2.0, 0.8)
        length = _positive(params, "length", defaults[0])
        width = _positive(params, "width", defaults[1])
        height = _positive(params, "height", defaults[2])
        _box(worldbody, prefix, (obstacle.x, obstacle.y, obstacle.z + height / 2),
             (length / 2, width / 2, height / 2), obstacle.yaw)
    elif obstacle.kind == "stairs":
        count = int(params.get("count", 5))
        run = _positive(params, "run", 0.3)
        width = _positive(params, "width", 1.2)
        rise = _positive(params, "rise", 0.12)
        if count < 1:
            raise ValueError("stairs count must be positive")
        for step in range(count):
            height = rise * (step + 1)
            x, y = _local_xy(obstacle, run * (step + 0.5))
            _box(worldbody, "%s_step_%d" % (prefix, step),
                 (x, y, obstacle.z + height / 2), (run / 2, width / 2, height / 2), obstacle.yaw)
    elif obstacle.kind == "gap":
        gap = _positive(params, "gap", 0.35)
        length = _positive(params, "bank_length", 1.5)
        width = _positive(params, "width", 1.5)
        height = _positive(params, "height", 0.25)
        for side in (-1, 1):
            x, y = _local_xy(obstacle, side * (gap + length) / 2)
            _box(worldbody, "%s_bank_%s" % (prefix, "a" if side < 0 else "b"),
                 (x, y, obstacle.z + height / 2), (length / 2, width / 2, height / 2), obstacle.yaw)
    elif obstacle.kind == "stepping_stones":
        count = int(params.get("count", 7))
        size = _positive(params, "size", 0.22)
        spacing = _positive(params, "spacing", 0.45)
        height = _positive(params, "height", 0.2)
        stagger = _positive(params, "stagger", 0.25, allow_zero=True)
        if count < 1:
            raise ValueError("stepping_stones count must be positive")
        for item in range(count):
            lateral = stagger if item % 2 else -stagger
            x, y = _local_xy(obstacle, item * spacing, lateral)
            _box(worldbody, "%s_stone_%d" % (prefix, item),
                 (x, y, obstacle.z + height / 2), (size / 2, size / 2, height / 2), obstacle.yaw)
    elif obstacle.kind == "slalom":
        count = int(params.get("count", 6))
        spacing = _positive(params, "spacing", 0.8)
        offset = _positive(params, "offset", 0.5)
        radius = _positive(params, "radius", 0.04)
        height = _positive(params, "height", 0.9)
        if count < 1:
            raise ValueError("slalom count must be positive")
        for item in range(count):
            lateral = offset if item % 2 else -offset
            x, y = _local_xy(obstacle, item * spacing, lateral)
            ET.SubElement(worldbody, "geom", {
                "name": "%s_pole_%d" % (prefix, item), "type": "cylinder",
                "pos": _fmt((x, y, obstacle.z + height / 2)),
                "size": _fmt((radius, height / 2)), "rgba": "0.92 0.35 0.18 1",
            })
    elif obstacle.kind in {"ramp", "side_slope"}:
        length = _positive(params, "length", 2.0)
        width = _positive(params, "width", 1.5)
        angle = math.radians(_positive(params, "angle", 20.0))
        thickness = _positive(params, "thickness", 0.08)
        axis = (0, 1, 0) if obstacle.kind == "ramp" else (1, 0, 0)
        span = length if obstacle.kind == "ramp" else width
        lift = span * math.sin(angle)
        quat = _yaw_tilt_quat(obstacle.yaw, axis, -angle)
        geom = _box(
            worldbody, prefix,
            (obstacle.x, obstacle.y, obstacle.z + lift / 2 + thickness / 2),
            (length / 2, width / 2, thickness / 2), 0,
        )
        geom.attrib.pop("euler", None)
        geom.set("quat", _fmt(quat))
    elif obstacle.kind == "speed_bumps":
        count = int(params.get("count", 6))
        spacing = _positive(params, "spacing", 0.45)
        width = _positive(params, "width", 1.5)
        radius = _positive(params, "radius", 0.07)
        if count < 1:
            raise ValueError("speed_bumps count must be positive")
        for item in range(count):
            x, y = _local_xy(obstacle, item * spacing)
            lateral_x, lateral_y = -math.sin(obstacle.yaw), math.cos(obstacle.yaw)
            half = width / 2
            ET.SubElement(worldbody, "geom", {
                "name": "%s_bump_%d" % (prefix, item), "type": "capsule",
                "fromto": _fmt((x-lateral_x*half, y-lateral_y*half, obstacle.z+radius,
                                 x+lateral_x*half, y+lateral_y*half, obstacle.z+radius)),
                "size": _fmt((radius,)), "rgba": "0.78 0.55 0.18 1",
            })
    elif obstacle.kind == "hurdles":
        count = int(params.get("count", 4))
        spacing = _positive(params, "spacing", 1.0)
        width = _positive(params, "width", 1.4)
        height = _positive(params, "height", 0.28)
        thickness = _positive(params, "thickness", 0.045)
        if count < 1:
            raise ValueError("hurdles count must be positive")
        for item in range(count):
            x, y = _local_xy(obstacle, item * spacing)
            _box(worldbody, "%s_bar_%d" % (prefix, item), (x, y, obstacle.z + height),
                 (thickness / 2, width / 2, thickness / 2), obstacle.yaw,
                 "0.88 0.32 0.18 1")
    elif obstacle.kind == "narrow_bridge":
        length = _positive(params, "length", 3.0)
        width = _positive(params, "width", 0.45)
        height = _positive(params, "height", 0.35)
        _box(worldbody, prefix, (obstacle.x, obstacle.y, obstacle.z + height / 2),
             (length / 2, width / 2, height / 2), obstacle.yaw, "0.35 0.50 0.62 1")
    elif obstacle.kind == "wave_ground":
        length = _positive(params, "length", 4.0)
        width = _positive(params, "width", 1.5)
        amplitude = _positive(params, "amplitude", 0.12)
        waves = _positive(params, "waves", 3.0)
        segments = int(params.get("segments", 32))
        if segments < 4 or segments > 256:
            raise ValueError("wave_ground segments must be between 4 and 256")
        run = length / segments
        for item in range(segments):
            forward = -length / 2 + (item + 0.5) * run
            height = amplitude * (1.0 + math.sin(2 * math.pi * waves * (item + 0.5) / segments))
            x, y = _local_xy(obstacle, forward)
            _box(worldbody, "%s_wave_%d" % (prefix, item),
                 (x, y, obstacle.z + height / 2), (run / 2, width / 2, max(0.005, height / 2)),
                 obstacle.yaw, "0.30 0.50 0.46 1")
    elif obstacle.kind == "uneven_stairs":
        count = int(params.get("count", 7))
        run = _positive(params, "run", 0.32)
        width = _positive(params, "width", 1.3)
        min_rise = _positive(params, "min_rise", 0.07)
        max_rise = _positive(params, "max_rise", 0.17)
        seed = int(params.get("seed", 0))
        if count < 1 or max_rise < min_rise:
            raise ValueError("uneven_stairs needs count > 0 and max_rise >= min_rise")
        rng = np.random.default_rng(seed)
        level = 0.0
        for item, rise in enumerate(rng.uniform(min_rise, max_rise, count)):
            level += float(rise)
            x, y = _local_xy(obstacle, run * (item + 0.5))
            _box(worldbody, "%s_step_%d" % (prefix, item),
                 (x, y, obstacle.z + level / 2), (run / 2, width / 2, level / 2), obstacle.yaw)
    elif obstacle.kind == "random_blocks":
        length = _positive(params, "length", 4.0)
        width = _positive(params, "width", 2.0)
        if "density" in params:
            density = _positive(params, "density", 3.0)
            count = max(1, round(length * width * density))
        else:
            count = int(params.get("count", 24))
        min_size = _positive(params, "min_size", 0.12)
        max_size = _positive(params, "max_size", 0.32)
        max_height = _positive(params, "max_height", 0.18)
        seed = int(params.get("seed", 0))
        if count < 1 or count > 500 or max_size < min_size:
            raise ValueError("random_blocks count/size range is invalid")
        rng = np.random.default_rng(seed)
        for item in range(count):
            forward = float(rng.uniform(-length / 2, length / 2))
            lateral = float(rng.uniform(-width / 2, width / 2))
            size_x, size_y = rng.uniform(min_size, max_size, 2)
            height = float(rng.uniform(0.03, max_height))
            x, y = _local_xy(obstacle, forward, lateral)
            _box(worldbody, "%s_block_%d" % (prefix, item),
                 (x, y, obstacle.z + height / 2), (size_x / 2, size_y / 2, height / 2),
                 obstacle.yaw + float(rng.uniform(-0.5, 0.5)), "0.42 0.38 0.32 1")
    elif obstacle.kind == "seesaw":
        length = _positive(params, "length", 2.2)
        width = _positive(params, "width", 0.8)
        thickness = _positive(params, "thickness", 0.08)
        pivot_height = _positive(params, "pivot_height", 0.28)
        amplitude = math.radians(_positive(params, "amplitude", 18.0))
        speed = _positive(params, "speed", 1.0)
        body = ET.SubElement(worldbody, "body", {
            "name": "dynamic_seesaw_%d_a%d_s%d" % (
                index, round(amplitude * 1000), round(speed * 1000)
            ), "mocap": "true",
            "pos": _fmt((obstacle.x, obstacle.y, obstacle.z + pivot_height)),
            "quat": _fmt(_axis_angle((0, 0, 1), obstacle.yaw)),
        })
        ET.SubElement(body, "geom", {
            "name": prefix, "type": "box", "size": _fmt((length/2, width/2, thickness/2)),
            "rgba": "0.80 0.52 0.16 1",
        })
        _box(worldbody, prefix + "_pivot", (obstacle.x, obstacle.y, obstacle.z + pivot_height/2),
             (0.08, width/2, pivot_height/2), obstacle.yaw, "0.28 0.30 0.34 1")
    elif obstacle.kind == "rotating_bar":
        length = _positive(params, "length", 2.5)
        height = _positive(params, "height", 0.35)
        thickness = _positive(params, "thickness", 0.055)
        speed = _positive(params, "speed", 1.2)
        body = ET.SubElement(worldbody, "body", {
            "name": "dynamic_rotating_bar_%d_s%d" % (index, round(speed * 1000)),
            "mocap": "true",
            "pos": _fmt((obstacle.x, obstacle.y, obstacle.z + height)),
            "quat": _fmt(_axis_angle((0, 0, 1), obstacle.yaw)),
        })
        ET.SubElement(body, "geom", {
            "name": prefix, "type": "capsule", "fromto": _fmt((-length/2, 0, 0, length/2, 0, 0)),
            "size": _fmt((thickness,)), "rgba": "0.90 0.22 0.16 1",
        })
        ET.SubElement(worldbody, "geom", {
            "name": prefix + "_post", "type": "cylinder",
            "pos": _fmt((obstacle.x, obstacle.y, obstacle.z + height/2)),
            "size": _fmt((0.07, height/2)), "rgba": "0.25 0.28 0.32 1",
        })


def export_scene_map(scene, output_dir):
    """Export ``scene.json``, optional height PNG, and terrain-only ``terrain.xml``."""
    if not isinstance(scene, SceneSpec):
        scene = SceneSpec.from_dict(scene)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = save_scene_spec(scene, output_dir / "scene.json")

    root = ET.Element("mujoco", {"model": scene.name})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    terrain = scene.terrain
    image_path = None
    if terrain.kind == "flat" or terrain.height == 0:
        ET.SubElement(worldbody, "geom", {
            "name": "ground", "type": "plane", "size": _fmt((terrain.length / 2, terrain.width / 2, 0.1)),
            "rgba": "0.24 0.29 0.34 1",
        })
    else:
        image_path = output_dir / "terrain.png"
        pixels = np.rint(generate_heightfield(terrain) * 65535).astype(np.uint16)
        Image.fromarray(pixels, mode="I;16").save(image_path)
        ET.SubElement(asset, "hfield", {
            "name": "generated_heightfield", "file": image_path.name,
            "size": _fmt((terrain.length / 2, terrain.width / 2, terrain.height, 0.05)),
        })
        ET.SubElement(worldbody, "geom", {
            "name": "ground", "type": "hfield", "hfield": "generated_heightfield",
            "rgba": "0.24 0.29 0.34 1",
        })
    for index, obstacle in enumerate(scene.obstacles):
        _append_obstacle(worldbody, obstacle, index)

    try:
        ET.indent(root, space="  ")
    except AttributeError:  # Python 3.8
        pass
    xml_path = output_dir / "terrain.xml"
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    result = {"xml": xml_path, "scene": scene_path}
    if image_path is not None:
        result["heightfield"] = image_path
    return result


def scene_map_spec(scene, output_dir, **map_overrides):
    """Export a scene and return a MapSpec ready for RuntimeScene."""
    from .map_manager import MapSpec

    paths = export_scene_map(scene, output_dir)
    return MapSpec(paths["xml"], **map_overrides)
