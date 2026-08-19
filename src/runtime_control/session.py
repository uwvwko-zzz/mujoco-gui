"""High-level ownership of a composed MuJoCo scene and runtime controls."""

from pathlib import Path
import tempfile

import mujoco

from .integration import viewer_context
from .map_manager import compose_scene, validate_model_dimensions
from .runtime import RuntimeControl


class RuntimeScene:
    """Compose, compile and clean up a runtime-controlled MuJoCo scene.

    Policy inference, observations and actuator commands deliberately remain in
    the host project.  This class owns only reusable infrastructure and can be
    used as a context manager or through explicit ``open``/``close`` calls.
    """

    def __init__(
        self,
        robot_xml,
        map_specs,
        runtime_config,
        *,
        robot_body_name="base_link",
        robot_cameras=(),
        expected_dimensions=None,
        dimension_context="composed MuJoCo scene",
        dynamic_obstacle_map=None,
        xml_transform=None,
        output_name="runtime_scene.xml",
    ):
        self.robot_xml = Path(robot_xml).expanduser().resolve()
        self.map_specs = dict(map_specs)
        self.runtime_config = runtime_config
        self.robot_body_name = str(robot_body_name)
        self.robot_cameras = tuple(robot_cameras)
        self.expected_dimensions = expected_dimensions
        self.dimension_context = str(dimension_context)
        self.dynamic_obstacle_map = dynamic_obstacle_map
        self.xml_transform = xml_transform
        self.output_name = str(output_name)

        self._temp_dir = None
        self.combined_xml = None
        self.model = None
        self.data = None
        self.runtime = None
        self.adapter = None

    @classmethod
    def for_adapter(
        cls, adapter, robot_xml, map_specs, runtime_config, **kwargs
    ):
        """Construct a scene whose robot contract is supplied by an adapter."""
        forbidden = {"robot_body_name", "expected_dimensions"}.intersection(kwargs)
        if forbidden:
            raise TypeError(
                "for_adapter derives %s from the adapter"
                % ", ".join(sorted(forbidden))
            )
        scene = cls(
            robot_xml,
            map_specs,
            runtime_config,
            robot_body_name=adapter.root_body_name,
            expected_dimensions=adapter.expected_dimensions,
            **kwargs
        )
        scene.adapter = adapter
        return scene

    @property
    def is_open(self):
        return self.runtime is not None

    def open(self):
        """Create the composed model and return this scene."""
        if self.is_open:
            return self
        if not self.robot_xml.is_file():
            raise FileNotFoundError("robot XML not found: %s" % self.robot_xml)
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mujoco_runtime_")
        try:
            self.combined_xml = Path(self._temp_dir.name) / self.output_name
            compose_scene(
                robot_xml=self.robot_xml,
                map_specs=self.map_specs,
                output_path=self.combined_xml,
                robot_body_name=self.robot_body_name,
                robot_cameras=self.robot_cameras,
            )
            if self.xml_transform is not None:
                self.xml_transform(self.combined_xml)
            self.model = mujoco.MjModel.from_xml_path(str(self.combined_xml))
            self.data = mujoco.MjData(self.model)
            if self.expected_dimensions is not None:
                validate_model_dimensions(
                    self.model,
                    self.expected_dimensions,
                    self.dimension_context,
                )
            if self.adapter is not None:
                self.adapter.bind_and_validate(self.model)
            self.runtime = RuntimeControl(
                self.runtime_config,
                map_names=self.map_specs,
                base_body_name=self.robot_body_name,
                dynamic_obstacle_map=self.dynamic_obstacle_map,
            )
            return self
        except Exception:
            self.close()
            raise

    def viewer(self, browser_only=False, key_callback=None, running=None):
        """Return the browser-only loop or native MuJoCo viewer context."""
        if not self.is_open:
            raise RuntimeError("RuntimeScene.open() must be called first")
        return viewer_context(
            browser_only,
            self.model,
            self.data,
            key_callback=key_callback,
            runtime=self.runtime,
            running=running,
        )

    def close(self):
        """Release browser/render threads and temporary composed assets."""
        runtime, temp_dir = self.runtime, self._temp_dir
        self.runtime = None
        self._temp_dir = None
        try:
            if runtime is not None:
                runtime.close()
        finally:
            if self.adapter is not None:
                self.adapter.unbind()
            self.model = None
            self.data = None
            self.combined_xml = None
            if temp_dir is not None:
                temp_dir.cleanup()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
