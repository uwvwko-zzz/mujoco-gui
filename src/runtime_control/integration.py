"""Small adapters shared by MuJoCo policy-player entry points.

This module deliberately contains no policy or joint-order assumptions.  It
only removes repetitive glue needed when connecting a robot player to
``RuntimeControl``.
"""

from copy import deepcopy

import mujoco

from .panel import (
    DEFAULT_SCHEMA,
    ParameterSpec,
    merge_parameter_schema,
    normalize_actions,
)


class BrowserOnlyLoop:
    """Viewer-shaped context used when the browser is the only display."""

    def __init__(self, running=None):
        self._running = running
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._closed = True
        return False

    def is_running(self):
        return not self._closed and (
            self._running is None or bool(self._running())
        )

    def sync(self):
        pass


def viewer_context(
    browser_gui, model, data, key_callback=None, runtime=None, running=None,
):
    """Return mutually exclusive browser-only or native-viewer contexts.

    ``RuntimeControl`` owns browser rendering.  Starting a passive native
    viewer at the same time renders every frame twice and is a common source
    of unexpectedly low frame rates.
    """
    if browser_gui:
        callback = running
        if callback is None and runtime is not None:
            callback = runtime.is_running
        return BrowserOnlyLoop(callback)
    import mujoco.viewer

    return mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    )


def setup_tracking_camera(
    viewer,
    model,
    body_name,
    distance=1.7,
    azimuth=135.0,
    elevation=-20.0,
):
    """Configure a native passive viewer to track a named robot body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo model has no body {body_name!r}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    viewer.cam.distance = float(distance)
    viewer.cam.azimuth = float(azimuth)
    viewer.cam.elevation = float(elevation)


def make_standard_robot_cameras(
    prefix="robot",
    front_position=(0.28, 0.0, 0.16),
    rear_position=(-1.0, 0.0, 0.65),
    top_position=(0.0, 0.0, 1.2),
):
    """Return front, rear-overhead and top cameras for ``compose_scene``.

    Positions are expressed in the root body's local frame.  The convention
    is ``+X`` forward, ``+Y`` left and ``+Z`` up.  Scale the positions for a
    robot substantially larger or smaller than a medium quadruped.
    """
    def vector(values):
        return " ".join(f"{float(value):g}" for value in values)

    return (
        {
            "name": f"{prefix}_front_camera",
            "mode": "fixed",
            "pos": vector(front_position),
            "xyaxes": "0 -1 0 0.242536 0 0.970143",
            "fovy": "80",
        },
        {
            "name": f"{prefix}_rear_overhead_camera",
            "mode": "fixed",
            "pos": vector(rear_position),
            "xyaxes": "0 -1 0 0.342020 0 0.939693",
            "fovy": "85",
        },
        {
            "name": f"{prefix}_top_camera",
            "mode": "fixed",
            "pos": vector(top_position),
            "xyaxes": "1 0 0 0 1 0",
            "fovy": "85",
        },
    )


def standard_camera_options(prefix="robot", tracking_label="第三人称跟随"):
    """Return UI camera names matching ``make_standard_robot_cameras``."""
    return {
        "tracking": tracking_label,
        f"{prefix}_front_camera": "机身前视（第一人称）",
        f"{prefix}_rear_overhead_camera": "机器人后上方跟随",
        f"{prefix}_top_camera": "机器人正上方俯视",
    }


def make_runtime_config(
    *,
    gui,
    title,
    maps,
    map_spawns,
    kp,
    kd,
    torque_limit,
    initial_position,
    initial_quaternion=(1.0, 0.0, 0.0, 0.0),
    gravity_z=-9.81,
    command=(1.0, 1.0, 1.0, 0.45),
    height_range=None,
    cameras=None,
    port=8765,
    render=None,
    tracking_camera=None,
    randomization=None,
    push=None,
    parameters=None,
    actions=None,
    random_seed=None,
    snapshot_path=None,
    stop_on_panel_close=True,
    open_browser=True,
    host="127.0.0.1",
):
    """Build the complete config shape required by ``RuntimeControl``.

    ``maps`` and ``cameras`` map stable internal names to UI labels.  Spawn
    quaternions use MuJoCo's ``wxyz`` order.  Returned containers are copies,
    so RuntimeControl may update commands/spawns without changing the caller's
    dictionaries.
    """
    map_labels = dict(maps)
    spawns = deepcopy(dict(map_spawns))
    missing_spawns = sorted(set(map_labels) - set(spawns))
    if missing_spawns:
        raise ValueError(
            "Missing map_spawns for: " + ", ".join(missing_spawns)
        )
    if len(command) != 4:
        raise ValueError("command must be (linear_x, linear_y, yaw, height)")
    render_cfg = {
        "width": 640,
        "height": 480,
        "fps": 120,
        "jpeg_quality": 70,
    }
    render_cfg.update(render or {})
    tracking_cfg = {
        "camera_distance": 1.7,
        "camera_azimuth": 135.0,
        "camera_elevation": -20.0,
    }
    tracking_cfg.update(tracking_camera or {})
    camera_labels = dict(cameras or {"tracking": "第三人称跟随"})
    parameter_specs = []
    runtime_parameters = {}
    for item in parameters or ():
        spec = item if isinstance(item, ParameterSpec) else ParameterSpec(**dict(item))
        parameter_specs.append(spec)
        runtime_parameters[spec.key] = float(spec.default)
    parameter_schema = merge_parameter_schema(DEFAULT_SCHEMA, parameter_specs)
    custom_actions = normalize_actions(actions)
    config = {
        "_runtime_gui": bool(gui),
        "simulation": {
            "initial_position": list(initial_position),
            "initial_quaternion": list(initial_quaternion),
            "gravity_z": float(gravity_z),
        },
        "control": {
            "kp": float(kp),
            "kd": float(kd),
            "torque_limit": float(torque_limit),
        },
        "command": {
            "linear_x": float(command[0]),
            "linear_y": float(command[1]),
            "yaw": float(command[2]),
            "height": float(command[3]),
        },
        "runtime_ui": {
            "title": str(title),
            "host": str(host),
            "port": int(port),
            "stop_on_panel_close": bool(stop_on_panel_close),
            "open_browser": bool(open_browser),
            "default_map": next(iter(map_labels), None),
            "maps": map_labels,
            "default_camera": next(iter(camera_labels), None),
            "cameras": camera_labels,
            **render_cfg,
            **tracking_cfg,
        },
        "map_spawns": spawns,
        "runtime_parameters": runtime_parameters,
        "runtime_parameter_schema": parameter_schema,
        "runtime_actions": custom_actions,
        "runtime_random_seed": random_seed,
        "runtime_snapshot_path": (
            str(snapshot_path) if snapshot_path is not None else None
        ),
        "runtime_randomization": deepcopy(randomization or {}),
        "stability_test_push": deepcopy(
            push or {"force_range": [80.0, 180.0], "duration_range": [0.10, 0.25]}
        ),
    }
    if height_range is not None:
        config["observation"] = {"height_range": list(height_range)}
    return config
