"""Robot-independent MJCF terrain composition and runtime map switching."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


MAP_GEOM_PREFIX = "map_"


@dataclass(frozen=True)
class MapSpec:
    """A terrain-only MJCF file and optional bodies to omit while importing."""

    path: Path
    exclude_bodies: tuple = ()
    minimum_box_half_thickness: float = None
    contact_params: dict = field(default_factory=dict)
    strict: bool = True

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        object.__setattr__(self, "exclude_bodies", tuple(self.exclude_bodies))
        thickness = self.minimum_box_half_thickness
        if thickness is not None and float(thickness) < 0:
            raise ValueError("minimum_box_half_thickness must be >= 0")
        allowed = {
            "condim", "friction", "solref", "solimp", "margin", "gap",
            "priority", "contype", "conaffinity",
        }
        contact_params = {
            str(name): str(value) for name, value in self.contact_params.items()
        }
        unknown = sorted(set(contact_params) - allowed)
        if unknown:
            raise ValueError("unsupported contact parameters: " + ", ".join(unknown))
        object.__setattr__(self, "contact_params", contact_params)


def _vector(text):
    return np.asarray([float(item) for item in text.split()], dtype=np.float64)


def _format_vector(values):
    return " ".join(f"{value:.6g}" for value in values)


def _quat_rotation(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def thicken_thin_collision_boxes(root, minimum_half_thickness=0.05):
    """Prevent fast feet tunnelling through very thin static box obstacles."""
    for geom in root.findall(".//geom[@type='box']"):
        size_text, pos_text = geom.get("size"), geom.get("pos")
        if size_text is None or pos_text is None:
            continue
        if geom.get("contype") == "0" or geom.get("conaffinity") == "0":
            continue
        size = _vector(size_text)
        if size.size != 3 or size[2] >= minimum_half_thickness:
            continue
        old_half_thickness = size[2]
        size[2] = minimum_half_thickness
        geom.set("size", _format_vector(size))
        quat = _vector(geom.get("quat", "1 0 0 0"))
        pos = _vector(pos_text)
        pos -= (size[2] - old_half_thickness) * _quat_rotation(quat)[:, 2]
        geom.set("pos", _format_vector(pos))


def _asset_source_name(element):
    name = element.get("name")
    if name is None and element.get("file"):
        name = Path(element.get("file")).stem
    return name


def _asset_file_path(map_spec, map_root, element):
    file_name = element.get("file")
    if not file_name:
        return None
    path = Path(file_name)
    if path.is_absolute():
        return path
    compiler = map_root.find("compiler")
    asset_dir = ""
    if compiler is not None:
        asset_dir = compiler.get("assetdir", "")
        if element.tag == "mesh":
            asset_dir = compiler.get("meshdir", asset_dir)
        elif element.tag in {"texture", "hfield"}:
            asset_dir = compiler.get("texturedir", asset_dir)
    return (map_spec.path.parent / asset_dir / path).resolve()


def _normalize_specs(map_specs):
    normalized = {}
    for name, spec in map_specs.items():
        normalized[name] = spec if isinstance(spec, MapSpec) else MapSpec(spec)
    return normalized


def _imported_world_elements(map_name, map_root, map_spec):
    source_world = map_root.find("worldbody")
    if source_world is None:
        raise ValueError("Map %r has no worldbody: %s" % (map_name, map_spec.path))
    excluded = set(map_spec.exclude_bodies)
    imported = []
    for element in source_world:
        if element.tag not in {"geom", "body"}:
            continue
        if element.tag == "body" and element.get("name") in excluded:
            continue
        imported.append(element)
    if not imported:
        raise ValueError("Map %r has no terrain elements after exclusions" % map_name)
    if map_spec.strict:
        for element in imported:
            if element.tag == "body" and (
                element.findall(".//joint") or element.findall(".//freejoint")
            ):
                raise ValueError(
                    "Map %r is not terrain-only: imported body contains a joint" % map_name
                )
    return imported


def validate_model_dimensions(model, expected, label="composed scene"):
    """Raise when a compiled map changes the robot's nq/nv/nu dimensions."""
    if hasattr(expected, "nq"):
        dimensions = (int(expected.nq), int(expected.nv), int(expected.nu))
    else:
        dimensions = tuple(int(value) for value in expected)
    actual = (int(model.nq), int(model.nv), int(model.nu))
    if len(dimensions) != 3:
        raise ValueError("expected dimensions must be (nq, nv, nu)")
    if actual != dimensions:
        raise ValueError(
            "%s changed robot dimensions: expected nq/nv/nu=%s, got %s"
            % (label, dimensions, actual)
        )
    return actual


def compose_scene(
    robot_xml,
    map_specs,
    output_path,
    robot_body_name="base_link",
    robot_cameras=(),
    minimum_box_half_thickness=0.05,
):
    """Combine one robot MJCF with multiple static terrain-only MJCF files.

    Imported assets and geoms receive a map-specific prefix. At runtime this
    prefix lets :class:`MapManager` hide and disable every inactive map without
    recompiling the MuJoCo model.
    """
    robot_xml = Path(robot_xml).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    specs = _normalize_specs(map_specs)
    robot_root = ET.parse(robot_xml).getroot()
    map_roots = {name: ET.parse(spec.path).getroot() for name, spec in specs.items()}
    imported_elements = {}
    for name, root in map_roots.items():
        imported_elements[name] = _imported_world_elements(
            name, root, specs[name]
        )
        thickness = specs[name].minimum_box_half_thickness
        if thickness is None:
            thickness = minimum_box_half_thickness
        if thickness:
            thicken_thin_collision_boxes(root, float(thickness))

    compiler = robot_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        robot_root.insert(0, compiler)
    for attr in ("meshdir", "texturedir", "assetdir"):
        directory = compiler.get(attr)
        if directory and not Path(directory).is_absolute():
            compiler.set(attr, str((robot_xml.parent / directory).resolve()))

    robot_asset = robot_root.find("asset")
    if robot_asset is None:
        robot_asset = ET.Element("asset")
        robot_root.insert(1, robot_asset)

    asset_maps = {}
    for map_name, map_root in map_roots.items():
        name_maps = {"texture": {}, "material": {}, "mesh": {}, "hfield": {}}
        asset_maps[map_name] = name_maps
        source_asset = map_root.find("asset")
        if source_asset is None:
            continue
        source_geoms = []
        for world_element in imported_elements[map_name]:
            if world_element.tag == "geom":
                source_geoms.append(world_element)
            else:
                source_geoms.extend(world_element.findall(".//geom"))
        used_assets = {
            "material": {geom.get("material") for geom in source_geoms if geom.get("material")},
            "mesh": {geom.get("mesh") for geom in source_geoms if geom.get("mesh")},
            "hfield": {geom.get("hfield") for geom in source_geoms if geom.get("hfield")},
            "texture": set(),
        }
        for material in source_asset.findall("material"):
            if material.get("name") in used_assets["material"] and material.get("texture"):
                used_assets["texture"].add(material.get("texture"))
        for element in source_asset:
            source_name = _asset_source_name(element)
            if element.tag in used_assets and source_name not in used_assets[element.tag]:
                continue
            if source_name:
                name_maps.setdefault(element.tag, {})[source_name] = (
                    f"{map_name}_{source_name}"
                )
        for element in source_asset:
            if element.tag == "texture" and element.get("type") == "skybox":
                continue
            copied = deepcopy(element)
            source_name = _asset_source_name(element)
            if element.tag in used_assets and source_name not in used_assets[element.tag]:
                continue
            if source_name:
                copied.set("name", name_maps[element.tag][source_name])
            texture = copied.get("texture")
            if texture:
                copied.set("texture", name_maps["texture"].get(texture, texture))
            file_path = _asset_file_path(specs[map_name], map_root, copied)
            if file_path is not None:
                copied.set("file", str(file_path))
            robot_asset.append(copied)

    worldbody = robot_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Robot MJCF has no worldbody: {robot_xml}")
    robot_body = worldbody.find(f"./body[@name='{robot_body_name}']")
    if robot_body is None:
        raise ValueError(
            f"Robot MJCF has no root body named {robot_body_name!r}: {robot_xml}"
        )
    for camera in robot_cameras:
        camera = dict(camera)
        name = camera.get("name")
        if name and robot_body.find(f"./camera[@name='{name}']") is None:
            ET.SubElement(robot_body, "camera", camera)

    common_world_elements = [
        deepcopy(element)
        for element in worldbody
        if element.tag in {"light", "camera"}
    ]
    for element in list(worldbody):
        worldbody.remove(element)
    for element in common_world_elements:
        worldbody.append(element)

    for map_name, map_root in map_roots.items():
        geom_index = 0
        body_index = 0
        site_index = 0
        camera_index = 0
        for element in imported_elements[map_name]:
            copied = deepcopy(element)
            if copied.tag == "body":
                for body in [copied, *copied.findall(".//body")]:
                    source_name = body.get("name", "body")
                    body.set(
                        "name", f"mapbody_{map_name}_{body_index}_{source_name}"
                    )
                    body_index += 1
                for site in copied.findall(".//site"):
                    site.set("name", f"mapsite_{map_name}_{site_index}")
                    site_index += 1
                for camera in copied.findall(".//camera"):
                    camera.set("name", f"mapcamera_{map_name}_{camera_index}")
                    camera_index += 1
            geoms = [copied] if copied.tag == "geom" else copied.findall(".//geom")
            refs = asset_maps.get(map_name, {})
            for geom in geoms:
                geom.set("name", f"{MAP_GEOM_PREFIX}{map_name}_{geom_index}")
                geom_index += 1
                if geom.get("contype", "1") != "0" and geom.get("conaffinity", "1") != "0":
                    for attr, value in specs[map_name].contact_params.items():
                        geom.set(attr, value)
                for attr, asset_type in (
                    ("material", "material"),
                    ("mesh", "mesh"),
                    ("hfield", "hfield"),
                ):
                    reference = geom.get(attr)
                    if reference:
                        geom.set(attr, refs.get(asset_type, {}).get(reference, reference))
            worldbody.append(copied)
    worldbody.append(robot_body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class MapManager:
    """Toggle precompiled map geom groups and return configured robot spawns."""

    def __init__(self, map_names, spawns=None, randomizers=None):
        self.map_names = tuple(map_names)
        self.spawns = dict(spawns or {})
        self.randomizers = dict(randomizers or {})
        self.groups = None
        self.active_map = None

    def initialize(self, model):
        ids_by_name = {name: [] for name in self.map_names}
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not geom_name:
                continue
            for map_name in self.map_names:
                if geom_name.startswith(f"{MAP_GEOM_PREFIX}{map_name}_"):
                    ids_by_name[map_name].append(geom_id)
                    break
        missing = [name for name, ids in ids_by_name.items() if not ids]
        if missing:
            raise ValueError(f"No compiled geoms found for maps: {', '.join(missing)}")
        self.groups = {
            name: {
                "ids": np.asarray(ids, dtype=np.int32),
                "pos": model.geom_pos[ids].copy(),
                "size": model.geom_size[ids].copy(),
                "quat": model.geom_quat[ids].copy(),
                "contype": model.geom_contype[ids].copy(),
                "conaffinity": model.geom_conaffinity[ids].copy(),
                "rgba": model.geom_rgba[ids].copy(),
            }
            for name, ids in ids_by_name.items()
        }

    def activate(self, model, data, map_name):
        if self.groups is None:
            self.initialize(model)
        if map_name not in self.groups or map_name == self.active_map:
            return None
        for name, group in self.groups.items():
            ids = group["ids"]
            active = name == map_name
            model.geom_pos[ids] = group["pos"]
            model.geom_size[ids] = group["size"]
            model.geom_quat[ids] = group["quat"]
            model.geom_contype[ids] = group["contype"] if active else 0
            model.geom_conaffinity[ids] = group["conaffinity"] if active else 0
            model.geom_rgba[ids] = group["rgba"]
            if not active:
                model.geom_rgba[ids, 3] = 0.0
        randomizer = self.randomizers.get(map_name)
        if randomizer is not None:
            randomizer(model, self.groups[map_name])
        self.active_map = map_name
        mujoco.mj_forward(model, data)
        print(f"[MAP] switched to {map_name}")
        return dict(self.spawns.get(map_name, {}))


def randomize_box_obstacles(model, group, rng=None, skip=1):
    """Default randomizer for a plane followed by static box obstacle geoms."""
    rng = rng or np.random.default_rng()
    obstacle_ids = group["ids"][skip:]
    if obstacle_ids.size == 0:
        return
    count = obstacle_ids.size
    sizes = np.column_stack(
        (
            rng.uniform(0.10, 0.28, count),
            rng.uniform(0.16, 0.48, count),
            rng.uniform(0.08, 0.28, count),
        )
    )
    positions = np.zeros((count, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(1.5, 8.2, count) + rng.uniform(-0.16, 0.16, count)
    positions[:, 1] = rng.uniform(-1.2, 1.2, count)
    positions[:, 2] = sizes[:, 2]
    angles = rng.uniform(-0.65, 0.65, count)
    quats = np.zeros((count, 4), dtype=np.float64)
    quats[:, 0] = np.cos(angles / 2)
    quats[:, 3] = np.sin(angles / 2)
    model.geom_size[obstacle_ids] = sizes
    model.geom_pos[obstacle_ids] = positions
    model.geom_quat[obstacle_ids] = quats
