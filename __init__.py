"""Reusable browser UI, terrain composition and live MuJoCo controls."""

from .map_manager import (
    MAP_GEOM_PREFIX,
    MapManager,
    MapSpec,
    compose_scene,
    randomize_box_obstacles,
    thicken_thin_collision_boxes,
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
    DEFAULT_SCHEMA,
    RuntimeControlPanel,
    build_panel_html,
    encode_rgb_jpeg,
)
from .runtime import RuntimeControl, RuntimeKeyboardMixin

__all__ = [
    "DEFAULT_SCHEMA",
    "BrowserOnlyLoop",
    "MAP_GEOM_PREFIX",
    "MapManager",
    "MapSpec",
    "MotorCommandDelay",
    "RuntimeControl",
    "RuntimeKeyboardMixin",
    "RuntimeControlPanel",
    "build_panel_html",
    "compose_scene",
    "compute_pd_torques",
    "encode_rgb_jpeg",
    "randomize_box_obstacles",
    "make_runtime_config",
    "make_standard_robot_cameras",
    "scale_torque_limits",
    "setup_tracking_camera",
    "standard_camera_options",
    "thicken_thin_collision_boxes",
    "viewer_context",
]
