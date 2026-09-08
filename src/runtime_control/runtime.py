"""Portable runtime control core for MuJoCo locomotion players."""

from copy import deepcopy
import re
import threading
import time

import mujoco
import numpy as np

from .map_manager import MapManager, randomize_box_obstacles
from .panel import DEFAULT_SCHEMA, RuntimeControlPanel, encode_rgb_jpeg


def apply_browser_camera_moves(camera, moves):
    """Apply normalized browser drags to a MuJoCo free camera in place."""
    for action, dx, dy in moves:
        if action == "rotate":
            camera.azimuth = (float(camera.azimuth) - dx * 180.0) % 360.0
            # Match direct-manipulation viewers: drag the mouse down to look up.
            camera.elevation = float(np.clip(camera.elevation - dy * 120.0, -89.0, 89.0))
        elif action == "zoom":
            camera.distance = float(np.clip(
                camera.distance * np.exp(dy * 2.5), 0.1, 100.0
            ))
        elif action == "pan":
            azimuth = np.radians(float(camera.azimuth))
            elevation = np.radians(float(camera.elevation))
            right = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            up = np.array([
                -np.sin(azimuth) * np.sin(elevation),
                np.cos(azimuth) * np.sin(elevation),
                np.cos(elevation),
            ])
            scale = max(0.1, float(camera.distance)) * 1.5
            camera.lookat[:] += scale * (-dx * right + dy * up)


def _quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray([
        lw*rw - lx*rx - ly*ry - lz*rz,
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
    ])


class RuntimeKeyboardMixin:
    """Adapter mixin for host keyboard classes exposing ``pressed`` and ``lock``."""

    def setup_runtime_control(
        self,
        config,
        map_names,
        base_body_name="base_link",
        dynamic_obstacle_map=None,
    ):
        self.runtime = RuntimeControl(
            config,
            map_names=map_names,
            base_body_name=base_body_name,
            dynamic_obstacle_map=dynamic_obstacle_map,
        )

    @property
    def panel(self):
        return self.runtime.panel

    def update_command(self):
        with self.lock:
            pressed = self.pressed.copy()
        if not self.runtime.update_command(pressed):
            super().update_command()

    def consume_reset(self):
        keyboard_requested = super().consume_reset()
        runtime_requested = self.runtime.consume_reset()
        return keyboard_requested or runtime_requested

    def runtime_control(self, model, data):
        return self.runtime.runtime_control(model, data)

    def apply_external_forces(self, model, data):
        self.runtime.apply_external_forces(model, data)

    def reset_simulation_state(self):
        self.runtime.reset_simulation_state()


class RuntimeControl:
    """Own UI state, map switching, model randomization, pushes and rendering.

    The host player remains responsible only for policy inference, PD torque
    application and its physical keyboard callbacks.
    """

    def __init__(
        self,
        config,
        map_names,
        base_body_name="base_link",
        dynamic_obstacle_map=None,
    ):
        self.config = config
        self.base_body_name = base_body_name
        ui_cfg = config.get("runtime_ui", {})
        self.panel = None
        if config.get("_runtime_gui", False):
            self.panel = RuntimeControlPanel(
                self._initial_state(),
                ui_config=ui_cfg,
                randomization=config.get("runtime_randomization", {}),
                schema=self._schema(),
                actions=config.get("runtime_actions", {}),
                random_seed=config.get("runtime_random_seed"),
                snapshot_path=config.get("runtime_snapshot_path"),
            )
            if ui_cfg.get("open_browser", True):
                self.panel.open_browser()

        random_seed = config.get("runtime_random_seed")
        self._rng = np.random.default_rng(
            None if random_seed is None else int(random_seed) + 1
        )
        randomizers = {}
        if dynamic_obstacle_map:
            randomizers[dynamic_obstacle_map] = (
                lambda model, group: randomize_box_obstacles(
                    model, group, rng=self._rng
                )
            )
        self.maps = MapManager(
            map_names,
            spawns=config.get("map_spawns", {}),
            randomizers=randomizers,
        )
        self.default_map = ui_cfg.get("default_map", next(iter(map_names), None))
        self.pending_map = self.default_map
        self.reset_requested = False
        self.pending_actions = set()
        self.stop_event = threading.Event()

        self.push_requested = False
        self.push_force = np.zeros(3, dtype=np.float64)
        self.push_steps_remaining = 0
        self.push_body_id = None

        self._base_mass = None
        self._base_inertia = None
        self._base_friction = None
        self._base_gravity = None
        self._base_body_id = None
        self._const_data = None
        self._applied_mass_scale = None
        self._applied_payload_mass = None
        self._applied_friction_scale = None
        self._applied_gravity_z = None

        self.model_lock = threading.RLock()
        self.render_lock = threading.Lock()
        self.render_qpos = None
        self.render_mocap_pos = None
        self.render_mocap_quat = None
        self.render_thread_started = False
        self.render_thread = None
        self.browser_render_failed = False
        self.browser_camera_mode = ui_cfg.get("default_camera", "tracking")
        self.dynamic_mocap_obstacles = None

    def _initial_state(self):
        command = self.config["command"]
        control = self.config["control"]
        state = {
            "linear_x": abs(float(command.get("linear_x", 0.0))),
            "linear_y": abs(float(command.get("linear_y", 0.0))),
            "yaw": abs(float(command.get("yaw", 0.0))),
            "height": float(command.get("height", 0.45)),
            "kp": float(control["kp"]),
            "kd": float(control["kd"]),
            "torque_limit": float(control["torque_limit"]),
            "motor_strength": 1.0,
            "motor_delay_ms": 0.0,
            "mass_scale": 1.0,
            "payload_mass": 0.0,
            "friction_scale": 1.0,
            "gravity_z": float(self.config["simulation"].get("gravity_z", -9.81)),
        }
        state.update({
            name: float(value)
            for name, value in self.config.get("runtime_parameters", {}).items()
        })
        return state

    def _schema(self):
        schema = deepcopy(
            self.config.get("runtime_parameter_schema", DEFAULT_SCHEMA)
        )
        height_range = self.config.get("observation", {}).get("height_range")
        if height_range and len(height_range) == 2:
            schema["commands"][3][2:4] = height_range
        return schema

    def update_command(self, physical_pressed=()):
        """Update command values; return False when the host should use its fallback."""
        if self.panel is None:
            return False
        state = self.panel.snapshot()
        command = self.config["command"]
        command["height"] = state["height"]
        pressed = set(physical_pressed) | self.panel.keyboard_snapshot()
        command["linear_x"] = (
            state["linear_x"] if "w" in pressed
            else -state["linear_x"] if "s" in pressed else 0.0
        )
        command["linear_y"] = (
            state["linear_y"] if "a" in pressed
            else -state["linear_y"] if "d" in pressed else 0.0
        )
        command["yaw"] = (
            state["yaw"] if "q" in pressed
            else -state["yaw"] if "e" in pressed else 0.0
        )
        actions = self.panel.consume_actions()
        if "push" in actions:
            self.push_requested = True
        if "reset" in actions:
            self.reset_requested = True
        if "randomize" in actions:
            self.panel.randomize()
        if "nominal" in actions:
            self.panel.restore_nominal()
        if "save_snapshot" in actions:
            try:
                target = self.panel.save_snapshot()
                print(f"[UI] runtime snapshot saved: {target}")
            except (OSError, ValueError) as exc:
                print(f"[UI] failed to save runtime snapshot: {exc}")
        if "load_snapshot" in actions:
            try:
                target = self.panel.load_snapshot()
                print(f"[UI] runtime snapshot loaded: {target}")
            except (OSError, ValueError, TypeError) as exc:
                print(f"[UI] failed to load runtime snapshot: {exc}")
        if "stop" in actions:
            with self.panel.lock:
                self.panel.pressed_keys.clear()
            for name in ("linear_x", "linear_y", "yaw"):
                command[name] = 0.0
        builtins = {
            "push", "reset", "randomize", "nominal", "stop",
            "save_snapshot", "load_snapshot",
        }
        self.pending_actions.update(actions - builtins)
        selected_map = self.panel.consume_map_change()
        if selected_map is not None:
            self.pending_map = selected_map
        selected_camera = self.panel.consume_camera_change()
        if selected_camera is not None:
            with self.render_lock:
                self.browser_camera_mode = selected_camera
        return True

    def consume_reset(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def consume_actions(self):
        """Return and clear custom browser actions registered by the host."""
        actions = set(self.pending_actions)
        self.pending_actions.clear()
        return actions

    def is_running(self):
        return not self.stop_event.is_set() and (
            self.panel is None or self.panel.is_running()
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def sync_height_to_panel(self):
        if self.panel is not None:
            with self.panel.lock:
                self.panel.state["height"] = float(self.config["command"]["height"])

    def sync_stop_to_panel(self):
        if self.panel is not None:
            with self.panel.lock:
                self.panel.pressed_keys.clear()
                for name in ("linear_x", "linear_y", "yaw"):
                    self.panel.state[name] = 0.0

    def request_push(self):
        self.push_requested = True

    def runtime_control(self, model, data):
        if self.pending_map is not None:
            with self.model_lock:
                spawn = self.maps.activate(model, data, self.pending_map)
            self.pending_map = None
            if spawn is not None:
                simulation = self.config["simulation"]
                if "position" in spawn:
                    simulation["initial_position"] = list(spawn["position"])
                if "quaternion" in spawn:
                    simulation["initial_quaternion"] = list(spawn["quaternion"])
                self.reset_requested = True

        self._animate_dynamic_obstacles(model, data)
        state = self.panel.snapshot() if self.panel is not None else self._initial_state()
        self._apply_model_parameters(model, data, state)
        if self.panel is not None:
            self._publish_frame(model, data)
        return state

    def _animate_dynamic_obstacles(self, model, data):
        if self.dynamic_mocap_obstacles is None:
            items = []
            for body_id in range(model.nbody):
                mocap_id = int(model.body_mocapid[body_id])
                if mocap_id < 0:
                    continue
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                seesaw = re.search(r"dynamic_seesaw_\d+_a(\d+)_s(\d+)", name)
                rotating = re.search(r"dynamic_rotating_bar_\d+_s(\d+)", name)
                if seesaw:
                    items.append({
                        "kind": "seesaw", "id": mocap_id,
                        "amplitude": int(seesaw.group(1)) / 1000.0,
                        "speed": int(seesaw.group(2)) / 1000.0,
                        "quat": data.mocap_quat[mocap_id].copy(),
                    })
                elif rotating:
                    items.append({
                        "kind": "rotating_bar", "id": mocap_id,
                        "speed": int(rotating.group(1)) / 1000.0,
                        "quat": data.mocap_quat[mocap_id].copy(),
                    })
            self.dynamic_mocap_obstacles = items
        for item in self.dynamic_mocap_obstacles:
            if item["kind"] == "seesaw":
                angle = item["amplitude"] * np.sin(item["speed"] * data.time)
                local = np.asarray([np.cos(angle/2), 0, np.sin(angle/2), 0])
            else:
                angle = item["speed"] * data.time
                local = np.asarray([np.cos(angle/2), 0, 0, np.sin(angle/2)])
            data.mocap_quat[item["id"]] = _quat_multiply(item["quat"], local)

    def _apply_model_parameters(self, model, data, state):
        if self._base_mass is None:
            self._base_mass = model.body_mass.copy()
            self._base_inertia = model.body_inertia.copy()
            self._base_friction = model.geom_friction.copy()
            self._base_gravity = model.opt.gravity.copy()
            self._base_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name
            )
            if self._base_body_id < 0:
                raise ValueError(f"MuJoCo model has no body {self.base_body_name!r}")
            self._const_data = mujoco.MjData(model)
            self._const_data.qpos[:] = model.qpos0
            mujoco.mj_forward(model, self._const_data)
            self._applied_mass_scale = 1.0
            self._applied_payload_mass = 0.0
            self._applied_friction_scale = 1.0
            self._applied_gravity_z = float(model.opt.gravity[2])

        mass_scale = state["mass_scale"]
        payload_mass = state["payload_mass"]
        if (
            mass_scale != self._applied_mass_scale
            or payload_mass != self._applied_payload_mass
        ):
            with self.model_lock:
                model.body_mass[:] = self._base_mass * mass_scale
                model.body_inertia[:] = self._base_inertia * mass_scale
                model.body_mass[self._base_body_id] += payload_mass
                self._const_data.qpos[:] = model.qpos0
                mujoco.mj_forward(model, self._const_data)
                mujoco.mj_setConst(model, self._const_data)
                mujoco.mj_forward(model, data)
            self._applied_mass_scale = mass_scale
            self._applied_payload_mass = payload_mass

        friction_scale = state["friction_scale"]
        if friction_scale != self._applied_friction_scale:
            with self.model_lock:
                model.geom_friction[:, 0] = self._base_friction[:, 0] * friction_scale
            self._applied_friction_scale = friction_scale
        gravity_z = state["gravity_z"]
        if gravity_z != self._applied_gravity_z:
            with self.model_lock:
                model.opt.gravity[:] = self._base_gravity
                model.opt.gravity[2] = gravity_z
            self._applied_gravity_z = gravity_z

    def apply_external_forces(self, model, data):
        if self.push_body_id is None:
            self.push_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name
            )
            if self.push_body_id < 0:
                raise ValueError(f"MuJoCo model has no body {self.base_body_name!r}")
        data.xfrc_applied[self.push_body_id, :] = 0.0
        if self.push_requested:
            push_cfg = self.config["stability_test_push"]
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            magnitude = self._rng.uniform(*push_cfg["force_range"])
            duration = self._rng.uniform(*push_cfg["duration_range"])
            self.push_force[:] = [
                magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0
            ]
            self.push_steps_remaining = max(
                1, int(np.ceil(duration / model.opt.timestep))
            )
            self.push_requested = False
            print(
                f"[PUSH] force={magnitude:.1f} N, "
                f"direction={np.degrees(angle):.1f}°, duration={duration:.3f} s"
            )
        if self.push_steps_remaining > 0:
            data.xfrc_applied[self.push_body_id, :3] = self.push_force
            self.push_steps_remaining -= 1

    def reset_simulation_state(self):
        self.push_requested = False
        self.push_force.fill(0.0)
        self.push_steps_remaining = 0

    def _publish_frame(self, model, data):
        if self.browser_render_failed:
            return
        with self.render_lock:
            self.render_qpos = data.qpos.copy()
            self.render_mocap_pos = data.mocap_pos.copy()
            self.render_mocap_quat = data.mocap_quat.copy()
        if not self.render_thread_started:
            self.render_thread_started = True
            self.render_thread = threading.Thread(
                target=self._render_loop, args=(model,), daemon=True
            )
            self.render_thread.start()

    def _render_loop(self, model):
        ui_cfg = self.config.get("runtime_ui", {})
        period = 1.0 / max(1.0, float(ui_cfg.get("fps", 120.0)))
        renderer = None
        try:
            with self.model_lock:
                render_width = int(ui_cfg.get("width", 640))
                render_height = int(ui_cfg.get("height", 480))
                model.vis.global_.offwidth = max(
                    int(model.vis.global_.offwidth), render_width
                )
                model.vis.global_.offheight = max(
                    int(model.vis.global_.offheight), render_height
                )
                render_data = mujoco.MjData(model)
                renderer = mujoco.Renderer(
                    model,
                    height=render_height,
                    width=render_width,
                )
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(camera)
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                base_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name
                )
            camera.distance = float(ui_cfg.get("camera_distance", 1.7))
            camera.azimuth = float(ui_cfg.get("camera_azimuth", 135))
            camera.elevation = float(ui_cfg.get("camera_elevation", -20))
            while self.is_running():
                started = time.monotonic()
                if self.panel.wants_frames():
                    camera_moves = self.panel.consume_camera_moves()
                    if camera_moves:
                        apply_browser_camera_moves(camera, camera_moves)
                        with self.render_lock:
                            self.browser_camera_mode = "free"
                    with self.render_lock:
                        qpos = None if self.render_qpos is None else self.render_qpos.copy()
                        mocap_pos = None if self.render_mocap_pos is None else self.render_mocap_pos.copy()
                        mocap_quat = None if self.render_mocap_quat is None else self.render_mocap_quat.copy()
                        camera_mode = self.browser_camera_mode
                    if qpos is not None:
                        with self.model_lock:
                            render_data.qpos[:] = qpos
                            if mocap_pos is not None:
                                render_data.mocap_pos[:] = mocap_pos
                                render_data.mocap_quat[:] = mocap_quat
                            mujoco.mj_forward(model, render_data)
                            if camera_mode == "tracking":
                                camera.lookat[:] = render_data.xpos[base_id]
                            renderer.update_scene(
                                render_data,
                                camera=(
                                    camera if camera_mode in {"tracking", "free"}
                                    else camera_mode
                                ),
                            )
                            frame = encode_rgb_jpeg(
                                renderer.render(),
                                quality=int(ui_cfg.get("jpeg_quality", 75)),
                            )
                        self.panel.set_frame(frame)
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as exc:
            if self.is_running():
                self.browser_render_failed = True
                print(f"[UI] browser MuJoCo renderer failed: {exc}")
        finally:
            if renderer is not None:
                renderer.close()

    def close(self):
        self.stop_event.set()
        if self.panel is not None:
            self.panel.close()
        if (
            self.render_thread is not None
            and self.render_thread is not threading.current_thread()
        ):
            self.render_thread.join(timeout=2.0)
