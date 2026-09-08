from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from runtime_control import (
    ObstacleSpec,
    SceneSpec,
    TerrainSpec,
    export_scene_map,
    generate_heightfield,
    load_scene_spec,
    scene_map_spec,
)


class SceneBuilderTests(unittest.TestCase):
    def test_heightfield_is_reproducible_and_bounded(self):
        spec = TerrainSpec(kind="noise", rows=24, cols=32, seed=7)
        first = generate_heightfield(spec)
        second = generate_heightfield(spec)
        self.assertEqual(first.shape, (24, 32))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_round_trip_normalizes_quaternion(self):
        scene = SceneSpec(
            name="course_1",
            terrain=TerrainSpec(kind="flat"),
            obstacles=[ObstacleSpec("wall", x=2, params={"height": 0.7})],
            spawn_quaternion=(2, 0, 0, 0),
            metadata={"purpose": "regression"},
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_scene_map(scene, directory)
            loaded = load_scene_spec(paths["scene"])
            self.assertEqual(loaded.to_dict(), scene.to_dict())
            self.assertEqual(loaded.spawn_quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_export_is_terrain_only_and_contains_all_obstacles(self):
        scene = SceneSpec(
            name="playground",
            terrain=TerrainSpec(kind="noise", rows=16, cols=16, seed=3),
            obstacles=[
                ObstacleSpec("platform"), ObstacleSpec("wall", x=2),
                ObstacleSpec("stairs", x=3), ObstacleSpec("gap", x=5),
                ObstacleSpec("stepping_stones", y=2), ObstacleSpec("slalom", y=-2),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_scene_map(scene, directory)
            self.assertEqual(set(paths), {"xml", "scene", "heightfield"})
            self.assertTrue(paths["heightfield"].is_file())
            root = ET.parse(paths["xml"]).getroot()
            self.assertIsNone(root.find(".//joint"))
            self.assertIsNone(root.find(".//freejoint"))
            self.assertGreater(len(root.findall(".//geom")), 20)
            hfield = root.find("./asset/hfield")
            self.assertEqual(hfield.get("file"), "terrain.png")

    def test_scene_map_spec_connects_to_existing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = scene_map_spec(SceneSpec(name="flat"), directory)
            self.assertEqual(spec.path, Path(directory).resolve() / "terrain.xml")
            self.assertTrue(spec.strict)

    def test_exported_map_compiles_when_mujoco_is_available(self):
        import mujoco
        if not hasattr(mujoco, "MjModel"):
            self.skipTest("official mujoco package is not installed in this interpreter")
        with tempfile.TemporaryDirectory() as directory:
            paths = export_scene_map(SceneSpec(
                name="compiled_course",
                terrain=TerrainSpec(kind="stairs", rows=16, cols=16),
                obstacles=[ObstacleSpec("gap", x=2), ObstacleSpec("slalom", x=4)],
            ), directory)
            model = mujoco.MjModel.from_xml_path(str(paths["xml"]))
            self.assertGreater(model.ngeom, 2)
            self.assertEqual(model.njnt, 0)

    def test_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            TerrainSpec(kind="lava")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "positive"):
                # Kind-specific parameters are validated when they are rendered.
                export_scene_map(
                    SceneSpec(obstacles=[ObstacleSpec("wall", params={"height": -1})]),
                    directory,
                )

    def test_extended_obstacle_library_exports_static_and_dynamic_geometry(self):
        kinds = (
            "ramp", "side_slope", "speed_bumps", "hurdles",
            "narrow_bridge", "wave_ground", "uneven_stairs",
            "random_blocks", "seesaw", "rotating_bar",
        )
        scene = SceneSpec(
            name="extended_library",
            obstacles=[
                ObstacleSpec(kind, x=(index % 5) * 3, y=(index // 5) * 3)
                for index, kind in enumerate(kinds)
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_scene_map(scene, directory)
            root = ET.parse(paths["xml"]).getroot()
            names = [item.get("name", "") for item in root.findall(".//geom")]
            self.assertTrue(any("ramp" in name for name in names))
            self.assertTrue(any("random_blocks" in name for name in names))
            dynamic = root.findall("./worldbody/body[@mocap='true']")
            self.assertEqual(len(dynamic), 2)
            self.assertIn("dynamic_seesaw", dynamic[0].get("name"))
            self.assertIn("dynamic_rotating_bar", dynamic[1].get("name"))

    def test_random_block_density_controls_generated_count(self):
        scene = SceneSpec(obstacles=[ObstacleSpec(
            "random_blocks",
            params={
                "length": 3, "width": 2, "density": 2,
                "min_size": 0.1, "max_size": 0.2,
                "max_height": 0.1, "seed": 4,
            },
        )])
        with tempfile.TemporaryDirectory() as directory:
            paths = export_scene_map(scene, directory)
            root = ET.parse(paths["xml"]).getroot()
            blocks = [
                geom for geom in root.findall(".//geom")
                if "random_blocks" in geom.get("name", "")
            ]
            self.assertEqual(len(blocks), 12)


if __name__ == "__main__":
    unittest.main()
