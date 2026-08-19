"""Robot-facing contract used by policy players built on runtime_control."""

from abc import ABC, abstractmethod

import mujoco
import numpy as np

from .map_manager import validate_model_dimensions


class RobotAdapter(ABC):
    """Translate one robot/policy pair to the reusable MuJoCo runtime.

    Implementations belong to the robot project, not to this package.  They
    own joint ordering, observations, action decoding, actuator control and
    reset semantics.  ``runtime_control`` continues to own only scene, UI and
    simulation infrastructure.
    """

    root_body_name = None
    expected_dimensions = None

    def __init__(self):
        self._bound_model = None
        self._actuator_ids = None
        self.last_control = None

    @property
    def actuator_ids(self):
        """Actuator IDs in the exact order returned by ``compute_control``."""
        if self._actuator_ids is None:
            raise RuntimeError("adapter is not bound; call bind_and_validate(model)")
        return self._actuator_ids

    def bind_and_validate(self, model):
        """Validate the generic contract, bind robot addresses, and return self."""
        if not self.root_body_name:
            raise ValueError("RobotAdapter.root_body_name must be set")
        if self.expected_dimensions is not None:
            validate_model_dimensions(
                model,
                self.expected_dimensions,
                "%s adapter" % type(self).__name__,
            )
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name
        )
        if body_id < 0:
            raise ValueError(
                "MuJoCo model has no adapter root body %r" % self.root_body_name
            )
        self.bind(model)
        ids = np.asarray(self._actuator_ids, dtype=np.int32)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("adapter must bind at least one actuator ID")
        if np.any(ids < 0) or np.any(ids >= model.nu):
            raise ValueError("adapter bound an actuator ID outside model.nu")
        if np.unique(ids).size != ids.size:
            raise ValueError("adapter actuator IDs must be unique")
        self._actuator_ids = ids
        self._bound_model = model
        return self

    def _require_bound(self, model):
        if self._bound_model is not model:
            raise RuntimeError(
                "adapter is not bound to this model; call bind_and_validate(model)"
            )

    def unbind(self):
        """Forget model-owned IDs when the surrounding RuntimeScene closes."""
        self._bound_model = None
        self._actuator_ids = None
        self.last_control = None

    @abstractmethod
    def bind(self, model):
        """Resolve and store robot-specific joint and actuator addresses."""

    @abstractmethod
    def reset(self, model, data, runtime_config):
        """Reset MuJoCo state and all adapter-owned policy/control history."""

    @abstractmethod
    def build_observation(self, model, data, command, **context):
        """Return one policy input batch for the current robot state."""

    @abstractmethod
    def compute_control(self, model, data, action, runtime_state):
        """Return actuator commands in ``actuator_ids`` order."""

    def apply_control(self, model, data, action, runtime_state):
        """Validate and write robot-specific control output to ``data.ctrl``."""
        self._require_bound(model)
        control = np.asarray(
            self.compute_control(model, data, action, runtime_state),
            dtype=np.float64,
        )
        expected = (self.actuator_ids.size,)
        if control.shape != expected:
            raise ValueError(
                "compute_control returned shape %s, expected %s"
                % (control.shape, expected)
            )
        if not np.all(np.isfinite(control)):
            raise ValueError("compute_control returned NaN or infinite values")
        data.ctrl[self.actuator_ids] = control
        self.last_control = control.copy()
        return control
