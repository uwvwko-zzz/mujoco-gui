import unittest

import numpy as np

from runtime_control import (
    MotorCommandDelay,
    compute_pd_torques,
    scale_torque_limits,
)


class ControlTests(unittest.TestCase):
    def test_pd_and_limit_ratios(self):
        torque = compute_pd_torques(
            [1.0, -1.0], [0.0, 0.0], [0.5, -0.5],
            [10.0, 20.0], [2.0, 2.0], torque_limit=[5.0, 8.0],
        )
        np.testing.assert_allclose(torque, [5.0, -8.0])
        np.testing.assert_allclose(
            scale_torque_limits([120.0, 175.38, 28.68], 87.69, 175.38),
            [60.0, 87.69, 14.34],
        )

    def test_motor_delay_reset(self):
        delay = MotorCommandDelay(0.005)
        for value in range(5):
            result = delay.apply([value], 20.0)
        np.testing.assert_allclose(result, [0.0])
        delay.reset()
        np.testing.assert_allclose(delay.apply([9.0], 20.0), [9.0])


if __name__ == "__main__":
    unittest.main()
