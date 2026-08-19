import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from runtime_control import ActionSpec, ParameterSpec, RuntimeControlPanel
from runtime_control.panel import merge_parameter_schema, DEFAULT_SCHEMA


class PanelTests(unittest.TestCase):
    def make_panel(self, port=0, snapshot=None):
        parameter = ParameterSpec(
            "wheel_velocity_kp", "Wheel Kp", 0.0, 5.0, 0.05, 1.5
        )
        schema = merge_parameter_schema(DEFAULT_SCHEMA, [parameter])
        return RuntimeControlPanel(
            {
                "linear_x": 1.0, "linear_y": 1.0, "yaw": 1.0,
                "height": 0.45, "kp": 80.0, "kd": 2.5,
                "torque_limit": 175.38, "motor_strength": 1.0,
                "motor_delay_ms": 0.0, "mass_scale": 1.0,
                "payload_mass": 0.0, "friction_scale": 1.0,
                "gravity_z": -9.81, "wheel_velocity_kp": 1.5,
            },
            ui_config={"host": "127.0.0.1", "port": port},
            schema=schema,
            actions=[ActionSpec("fall_side", "Side fall", "n")],
            snapshot_path=snapshot,
            random_seed=7,
        )

    def post(self, panel, path, payload):
        request = Request(
            panel.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urlopen(request, timeout=2).read())

    def test_validation_custom_action_snapshot_and_port_release(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "preset.json"
            panel = self.make_panel(snapshot=snapshot)
            port = panel.server.server_address[1]
            self.post(panel, "/api/state", {"wheel_velocity_kp": 99})
            self.assertEqual(panel.snapshot()["wheel_velocity_kp"], 5.0)
            self.post(panel, "/api/action", {"action": "fall_side"})
            self.assertIn("fall_side", panel.consume_actions())
            panel.save_snapshot()
            self.assertTrue(snapshot.is_file())
            with self.assertRaises(HTTPError) as error:
                urlopen(panel.url + "/missing", timeout=2)
            self.assertEqual(error.exception.code, 404)
            panel.close()

            reopened = self.make_panel(port=port, snapshot=snapshot)
            reopened.close()

    def test_close_endpoint_stops_panel(self):
        panel = self.make_panel()
        self.post(panel, "/api/action", {"action": "close_panel"})
        self.assertTrue(panel.stop_event.wait(2.0))
        self.assertFalse(panel.is_running())


if __name__ == "__main__":
    unittest.main()
