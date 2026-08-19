"""Reusable named-joint position-PD adapter for legged robots."""

import mujoco
import numpy as np

from .adapter import RobotAdapter
from .control import MotorCommandDelay, compute_pd_torques, scale_torque_limits


class PositionPDAdapter(RobotAdapter):
    """Configurable adapter for policies controlling every joint by position.

    Observation layouts remain policy-specific and are injected through
    ``observation_builder(adapter, model, data, command, **context)``.
    """

    def __init__(
        self,
        *,
        root_body_name,
        expected_dimensions,
        joint_names,
        default_positions,
        kp,
        kd,
        action_scale,
        observation_builder,
        actuator_names=None,
        torque_limits=None,
        kp_reference=None,
        kd_reference=None,
        torque_limit_reference=None,
    ):
        super().__init__()
        self.root_body_name = str(root_body_name)
        self.expected_dimensions = tuple(expected_dimensions)
        self.joint_names = tuple(joint_names)
        self.actuator_names = (
            tuple(actuator_names) if actuator_names is not None
            else tuple(name.replace("_joint", "") for name in self.joint_names)
        )
        self.default_positions = np.asarray(default_positions, dtype=np.float64)
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        self.observation_builder = observation_builder
        self.configured_torque_limits = (
            None if torque_limits is None
            else np.asarray(torque_limits, dtype=np.float64)
        )
        self.kp_reference = (
            float(np.max(np.abs(self.kp)))
            if kp_reference is None else float(kp_reference)
        )
        self.kd_reference = (
            float(np.max(np.abs(self.kd)))
            if kd_reference is None else float(kd_reference)
        )
        self.configured_torque_limit_reference = torque_limit_reference
        self.torque_limit_reference = torque_limit_reference
        self.action_size = len(self.joint_names)
        self._validate_configuration()

        self.qpos_adr = None
        self.qvel_adr = None
        self.root_qpos_adr = None
        self.nominal_limits = None
        self.effective_limits = None
        self.motor_delay = None
        self.last_action = np.zeros(self.action_size, dtype=np.float32)

    def _validate_configuration(self):
        expected = (self.action_size,)
        values = {
            "actuator_names": np.asarray(self.actuator_names),
            "default_positions": self.default_positions,
            "kp": self.kp,
            "kd": self.kd,
        }
        for name, value in values.items():
            if value.shape != expected:
                raise ValueError("%s has shape %s, expected %s" % (
                    name, value.shape, expected
                ))
        if self.action_scale.ndim > 1 or self.action_scale.size not in (1, self.action_size):
            raise ValueError("action_scale must be scalar or one value per joint")
        if (
            self.configured_torque_limits is not None
            and self.configured_torque_limits.shape != expected
        ):
            raise ValueError("torque_limits must contain one value per joint")
        if not callable(self.observation_builder):
            raise TypeError("observation_builder must be callable")
        if self.kp_reference <= 0.0 or self.kd_reference <= 0.0:
            raise ValueError("kp_reference and kd_reference must be positive")

    def bind(self, model):
        qpos, qvel, actuators = [], [], []
        for joint_name, actuator_name in zip(
            self.joint_names, self.actuator_names
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(
                    "Missing joint/actuator: %s / %s"
                    % (joint_name, actuator_name)
                )
            if model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError("controlled joint must have one degree of freedom")
            qpos.append(model.jnt_qposadr[joint_id])
            qvel.append(model.jnt_dofadr[joint_id])
            actuators.append(actuator_id)
        self.qpos_adr = np.asarray(qpos, dtype=np.int32)
        self.qvel_adr = np.asarray(qvel, dtype=np.int32)
        self._actuator_ids = np.asarray(actuators, dtype=np.int32)

        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name
        )
        joint_start = int(model.body_jntadr[body_id])
        joint_count = int(model.body_jntnum[body_id])
        root_joint = next((
            joint_id for joint_id in range(joint_start, joint_start + joint_count)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ), None)
        if root_joint is None:
            raise ValueError("adapter root body must own a free joint")
        self.root_qpos_adr = int(model.jnt_qposadr[root_joint])

        self.nominal_limits = (
            np.max(np.abs(model.actuator_ctrlrange[self._actuator_ids]), axis=1)
            if self.configured_torque_limits is None
            else self.configured_torque_limits.copy()
        )
        self.torque_limit_reference = (
            float(np.max(self.nominal_limits))
            if self.configured_torque_limit_reference is None
            else float(self.configured_torque_limit_reference)
        )
        self.motor_delay = MotorCommandDelay(model.opt.timestep)
        self.reset_policy_state()

    def unbind(self):
        super().unbind()
        self.qpos_adr = None
        self.qvel_adr = None
        self.root_qpos_adr = None
        self.nominal_limits = None
        self.effective_limits = None
        self.motor_delay = None

    def reset_policy_state(self):
        self.last_action.fill(0.0)
        self.last_control = None
        if self.motor_delay is not None:
            self.motor_delay.reset()

    def reset(self, model, data, runtime_config):
        self._require_bound(model)
        simulation = runtime_config["simulation"]
        root = self.root_qpos_adr
        mujoco.mj_resetData(model, data)
        data.qpos[root:root + 3] = simulation["initial_position"]
        data.qpos[root + 3:root + 7] = simulation["initial_quaternion"]
        data.qpos[self.qpos_adr] = self.default_positions
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.reset_policy_state()
        mujoco.mj_forward(model, data)

    def build_observation(self, model, data, command, **context):
        self._require_bound(model)
        observation = np.asarray(
            self.observation_builder(
                self, model, data, command, **context
            ),
            dtype=np.float32,
        )
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation_builder returned NaN or infinite values")
        return observation

    def compute_control(self, model, data, action, runtime_state):
        self._require_bound(model)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError("action has shape %s, expected (%d,)" % (
                action.shape, self.action_size
            ))
        self.last_action[:] = action
        delayed = self.motor_delay.apply(
            action, runtime_state["motor_delay_ms"]
        )
        target = self.default_positions + self.action_scale * delayed
        self.effective_limits = scale_torque_limits(
            self.nominal_limits,
            runtime_state["torque_limit"],
            reference_limit=self.torque_limit_reference,
        )
        return compute_pd_torques(
            target,
            data.qpos[self.qpos_adr],
            data.qvel[self.qvel_adr],
            self.kp * (runtime_state["kp"] / self.kp_reference),
            self.kd * (runtime_state["kd"] / self.kd_reference),
            motor_strength=runtime_state["motor_strength"],
            torque_limit=self.effective_limits,
        )
