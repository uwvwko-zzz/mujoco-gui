from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from runtime_control import PositionPDAdapter, RobotAdapter


MODEL_XML = """<mujoco><worldbody><body name="base"><freejoint/>
<geom type="sphere" size="0.1" mass="1"/>
<body name="link" pos="0 0 0.2"><joint name="motor_joint" type="hinge"/>
<geom type="sphere" size="0.05" mass="0.1"/></body></body></worldbody>
<actuator><motor name="motor" joint="motor_joint" ctrllimited="true"
ctrlrange="-5 5"/>
</actuator></mujoco>"""


class DummyAdapter(RobotAdapter):
    root_body_name = "base"
    expected_dimensions = (8, 7, 1)

    def bind(self, model):
        self._actuator_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor"
        )]

    def reset(self, model, data, runtime_config):
        self._require_bound(model)
        mujoco.mj_resetData(model, data)

    def build_observation(self, model, data, command, **context):
        self._require_bound(model)
        return np.asarray(command, dtype=np.float32)[None, :]

    def compute_control(self, model, data, action, runtime_state):
        return np.asarray(action) * runtime_state["motor_strength"]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "model.xml"
        path.write_text(MODEL_XML)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

    def tearDown(self):
        self.directory.cleanup()

    def test_bind_observe_and_apply(self):
        adapter = DummyAdapter().bind_and_validate(self.model)
        np.testing.assert_allclose(
            adapter.build_observation(self.model, self.data, [1, 2, 3]),
            [[1, 2, 3]],
        )
        control = adapter.apply_control(
            self.model, self.data, [2.0], {"motor_strength": 0.5}
        )
        np.testing.assert_allclose(control, [1.0])
        np.testing.assert_allclose(self.data.ctrl, [1.0])

    def test_rejects_unbound_and_bad_shape(self):
        adapter = DummyAdapter()
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            adapter.apply_control(
                self.model, self.data, [1.0], {"motor_strength": 1.0}
            )
        adapter.bind_and_validate(self.model)
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.apply_control(
                self.model, self.data, [1.0, 2.0], {"motor_strength": 1.0}
            )

    def test_position_pd_adapter_is_configurable(self):
        adapter = PositionPDAdapter(
            root_body_name="base",
            expected_dimensions=(8, 7, 1),
            joint_names=("motor_joint",),
            actuator_names=("motor",),
            default_positions=(0.0,),
            kp=(10.0,),
            kd=(1.0,),
            action_scale=0.5,
            torque_limits=(5.0,),
            observation_builder=lambda adapter, model, data, command, **context: (
                np.asarray(command, dtype=np.float32)[None, :]
            ),
        ).bind_and_validate(self.model)
        adapter.reset(self.model, self.data, {
            "simulation": {
                "initial_position": [0, 0, 0.2],
                "initial_quaternion": [1, 0, 0, 0],
            }
        })
        control = adapter.apply_control(self.model, self.data, [1.0], {
            "kp": 10.0,
            "kd": 1.0,
            "torque_limit": 5.0,
            "motor_strength": 1.0,
            "motor_delay_ms": 0.0,
        })
        np.testing.assert_allclose(control, [5.0])


if __name__ == "__main__":
    unittest.main()
