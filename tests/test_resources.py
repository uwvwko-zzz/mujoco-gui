import unittest

from runtime_control import (
    MAPS_DIR,
    available_bundled_maps,
    bundled_map_path,
    bundled_map_spec,
    bundled_map_specs,
)


class ResourceTests(unittest.TestCase):
    def test_every_registered_map_is_packaged(self):
        names = available_bundled_maps()
        self.assertIn("rc26_track", names)
        self.assertIn("dynamic_obstacles", names)
        self.assertEqual(set(bundled_map_specs()), set(names))
        for name in names:
            self.assertTrue(bundled_map_path(name).is_file())

        self.assertTrue((MAPS_DIR / "imgs" / "perlin_rough.png").is_file())
        self.assertTrue(
            (MAPS_DIR / "barkour_assets" / "cone1__1default.stl").is_file()
        )
        self.assertEqual(bundled_map_spec("rc26_track").exclude_bodies, ("trunk",))

    def test_unknown_map_has_actionable_error(self):
        with self.assertRaisesRegex(KeyError, "choose from"):
            bundled_map_path("not-a-map")


if __name__ == "__main__":
    unittest.main()
