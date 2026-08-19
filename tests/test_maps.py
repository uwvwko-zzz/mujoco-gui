from pathlib import Path
import tempfile
import unittest

import mujoco

from runtime_control import MapSpec, compose_scene, validate_model_dimensions


ROBOT_XML = """<mujoco><worldbody><body name="base"><freejoint/>
<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>"""
MAP_XML = """<mujoco><worldbody><geom name="floor" type="plane"
size="0 0 0.1"/></worldbody></mujoco>"""
BAD_MAP_XML = """<mujoco><worldbody><body name="moving"><joint/>
<geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>"""


class MapTests(unittest.TestCase):
    def test_compose_preserves_dimensions_and_applies_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot, terrain, output = root / "robot.xml", root / "map.xml", root / "out.xml"
            robot.write_text(ROBOT_XML)
            terrain.write_text(MAP_XML)
            original = mujoco.MjModel.from_xml_path(str(robot))
            compose_scene(
                robot, {"flat": MapSpec(
                    terrain, contact_params={"friction": "0.7 0.01 0.001"}
                )}, output, robot_body_name="base",
            )
            combined = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(
                validate_model_dimensions(combined, original),
                (original.nq, original.nv, original.nu),
            )
            geom_id = mujoco.mj_name2id(
                combined, mujoco.mjtObj.mjOBJ_GEOM, "map_flat_0"
            )
            self.assertAlmostEqual(combined.geom_friction[geom_id, 0], 0.7)

    def test_strict_map_rejects_joint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot, terrain, output = root / "robot.xml", root / "map.xml", root / "out.xml"
            robot.write_text(ROBOT_XML)
            terrain.write_text(BAD_MAP_XML)
            with self.assertRaisesRegex(ValueError, "terrain-only"):
                compose_scene(
                    robot, {"bad": MapSpec(terrain)}, output,
                    robot_body_name="base",
                )


if __name__ == "__main__":
    unittest.main()
