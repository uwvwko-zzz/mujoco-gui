"""Reusable delayed motor command and PD torque helpers."""

from collections import deque

import numpy as np


class MotorCommandDelay:
    """A bounded target history implementing a configurable actuator delay."""

    def __init__(self, timestep, history_seconds=2.0):
        self.timestep = float(timestep)
        self.history = deque(
            maxlen=max(2, int(float(history_seconds) / self.timestep))
        )

    def reset(self):
        self.history.clear()

    def apply(self, targets, delay_ms):
        self.history.append(np.asarray(targets, dtype=np.float64).copy())
        delay_steps = max(
            0, int(round(float(delay_ms) * 1e-3 / self.timestep))
        )
        history_index = min(delay_steps, len(self.history) - 1)
        return self.history[-1 - history_index]


def compute_pd_torques(
    targets,
    positions,
    velocities,
    kp,
    kd,
    motor_strength=1.0,
    torque_limit=np.inf,
):
    """Compute strength-scaled PD torques with scalar or per-joint values."""
    torques = np.asarray(motor_strength, dtype=np.float64) * (
        np.asarray(kp, dtype=np.float64)
        * (np.asarray(targets) - np.asarray(positions))
        - np.asarray(kd, dtype=np.float64) * np.asarray(velocities)
    )
    limits = np.asarray(torque_limit, dtype=np.float64)
    return np.clip(torques, -limits, limits)


def scale_torque_limits(nominal_limits, runtime_limit, reference_limit=None):
    """Scale per-joint limits while preserving their nominal ratios.

    The runtime UI exposes one scalar torque-limit slider.  Robots often have
    different limits for hip, thigh and calf motors.  ``runtime_limit`` is
    therefore treated as the new value of ``reference_limit`` (the largest
    nominal limit by default), and every joint limit is scaled by the same
    ratio.
    """
    nominal = np.asarray(nominal_limits, dtype=np.float64)
    reference = (
        float(np.max(np.abs(nominal)))
        if reference_limit is None else float(reference_limit)
    )
    if reference <= 0.0:
        raise ValueError("reference_limit must be positive")
    return nominal * (float(runtime_limit) / reference)
