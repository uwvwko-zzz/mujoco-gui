"""Reusable browser UI, terrain composition and live MuJoCo controls."""

from .adapter import RobotAdapter
from .position_pd_adapter import PositionPDAdapter
from .map_manager import (
    MAP_GEOM_PREFIX,
    MapManager,
    MapSpec,
    compose_scene,
    randomize_box_obstacles,
    thicken_thin_collision_boxes,
    validate_model_dimensions,
)
from .control import MotorCommandDelay, compute_pd_torques, scale_torque_limits
from .integration import (
    BrowserOnlyLoop,
    make_runtime_config,
    make_standard_robot_cameras,
    setup_tracking_camera,
    standard_camera_options,
    viewer_context,
)
from .panel import (
    ActionSpec,
    DEFAULT_SCHEMA,
    ParameterSpec,
    RuntimeControlPanel,
    build_panel_html,
    encode_rgb_jpeg,
    merge_parameter_schema,
)
from .runtime import RuntimeControl, RuntimeKeyboardMixin
from .resources import (
    MAPS_DIR,
    available_bundled_maps,
    bundled_map_path,
    bundled_map_spec,
    bundled_map_specs,
)
from .session import RuntimeScene
from .scene_builder import (
    OBSTACLE_KINDS,
    TERRAIN_KINDS,
    ObstacleSpec,
    SceneSpec,
    TerrainSpec,
    export_scene_map,
    generate_heightfield,
    load_scene_spec,
    save_scene_spec,
    scene_map_spec,
)

__all__ = [
    "ActionSpec",
    "DEFAULT_SCHEMA",
    "BrowserOnlyLoop",
    "MAP_GEOM_PREFIX",
    "MAPS_DIR",
    "MapManager",
    "MapSpec",
    "MotorCommandDelay",
    "ParameterSpec",
    "PositionPDAdapter",
    "RuntimeControl",
    "RobotAdapter",
    "RuntimeScene",
    "RuntimeKeyboardMixin",
    "RuntimeControlPanel",
    "OBSTACLE_KINDS",
    "TERRAIN_KINDS",
    "ObstacleSpec",
    "SceneSpec",
    "TerrainSpec",
    "build_panel_html",
    "available_bundled_maps",
    "bundled_map_path",
    "bundled_map_spec",
    "bundled_map_specs",
    "compose_scene",
    "compute_pd_torques",
    "encode_rgb_jpeg",
    "export_scene_map",
    "generate_heightfield",
    "load_scene_spec",
    "merge_parameter_schema",
    "randomize_box_obstacles",
    "make_runtime_config",
    "make_standard_robot_cameras",
    "scale_torque_limits",
    "save_scene_spec",
    "scene_map_spec",
    "setup_tracking_camera",
    "standard_camera_options",
    "thicken_thin_collision_boxes",
    "validate_model_dimensions",
    "viewer_context",
]
