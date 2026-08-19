from pathlib import Path
import tempfile
import unittest

import mujoco

from runtime_control import MapSpec, RobotAdapter, RuntimeScene, make_runtime_config


ROBOT_XML = """<mujoco><worldbody><body name="base"><freejoint/>
<geom name="robot" type="sphere" size="0.1" mass="1"/>
<body name="link" pos="0 0 0.2"><joint name="motor_joint" type="hinge"/>
<geom type="sphere" size="0.05" mass="0.1"/></body>
</body></worldbody><actuator><motor name="motor" joint="motor_joint"/>
</actuator></mujoco>"""
MAP_XML = """<mujoco><worldbody><geom name="floor" type="plane"
size="0 0 0.1"/></worldbody></mujoco>"""


class SceneAdapter(RobotAdapter):
    root_body_name = "base"
    expected_dimensions = (8, 7, 1)

    def bind(self, model):
        self._actuator_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor"
        )]

    def reset(self, model, data, runtime_config):
        pass

    def build_observation(self, model, data, command, **context):
        return command

    def compute_control(self, model, data, action, runtime_state):
        return action


class RuntimeSceneTests(unittest.TestCase):
    def test_scene_owns_composition_model_runtime_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = root / "robot.xml"
            terrain = root / "terrain.xml"
            robot.write_text(ROBOT_XML)
            terrain.write_text(MAP_XML)
            config = make_runtime_config(
                gui=False,
                title="test",
                maps={"flat": "Flat"},
                map_spawns={
                    "flat": {
                        "position": [0, 0, 0.2],
                        "quaternion": [1, 0, 0, 0],
                    }
                },
                kp=10,
                kd=1,
                torque_limit=20,
                initial_position=[0, 0, 0.2],
            )
            adapter = SceneAdapter()
            scene = RuntimeScene.for_adapter(
                adapter,
                robot,
                {"flat": MapSpec(terrain)},
                config,
            )

            with scene:
                self.assertTrue(scene.is_open)
                self.assertIsInstance(scene.model, mujoco.MjModel)
                self.assertIsInstance(scene.data, mujoco.MjData)
                self.assertTrue(scene.combined_xml.is_file())
                self.assertEqual(adapter.actuator_ids.tolist(), [0])
                scene.runtime.runtime_control(scene.model, scene.data)
                combined_xml = scene.combined_xml

            self.assertFalse(scene.is_open)
            self.assertIsNone(scene.model)
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                _ = adapter.actuator_ids
            self.assertFalse(combined_xml.exists())
            scene.close()  # idempotent


if __name__ == "__main__":
    unittest.main()
