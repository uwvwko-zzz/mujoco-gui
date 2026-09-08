import unittest

import numpy as np

from runtime_control.runtime import apply_browser_camera_moves


class DummyCamera:
    azimuth = 135.0
    elevation = -20.0
    distance = 2.0

    def __init__(self):
        self.lookat = np.zeros(3)


class BrowserCameraTests(unittest.TestCase):
    def test_rotate_zoom_and_pan(self):
        camera = DummyCamera()
        apply_browser_camera_moves(camera, [
            ("rotate", 0.25, 0.1),
            ("zoom", 0.0, -0.2),
            ("pan", 0.1, -0.2),
        ])
        self.assertAlmostEqual(camera.azimuth, 90.0)
        self.assertAlmostEqual(camera.elevation, -32.0)
        self.assertLess(camera.distance, 2.0)
        self.assertGreater(float(np.linalg.norm(camera.lookat)), 0.0)

    def test_limits_prevent_invalid_camera(self):
        camera = DummyCamera()
        apply_browser_camera_moves(camera, [
            ("rotate", 0.0, 10.0), ("zoom", 0.0, -10.0),
        ])
        self.assertEqual(camera.elevation, -89.0)
        self.assertEqual(camera.distance, 0.1)


if __name__ == "__main__":
    unittest.main()
