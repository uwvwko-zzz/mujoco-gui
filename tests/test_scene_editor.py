import json
import tempfile
import unittest
from urllib.request import Request, urlopen

from runtime_control import SceneSpec, export_scene_map
from runtime_control.scene_editor import SceneEditorServer, build_scene_editor_html


class SceneEditorTests(unittest.TestCase):
    def test_html_contains_editor_controls(self):
        page = build_scene_editor_html()
        self.assertIn("MuJoCo 可视化地图编辑器", page)
        self.assertIn("stepping_stones", page)
        self.assertIn("单阶长度", page)
        self.assertIn("密集程度", page)
        self.assertIn("/api/export", page)
        self.assertIn("用演示机器人在 MuJoCo 展示", page)
        self.assertIn("退出编辑器", page)
        self.assertIn("/api/shutdown", page)
        self.assertNotIn("用 W1W + ONNX", page)

    def test_server_exports_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            editor = SceneEditorServer(directory).start()
            try:
                payload = {
                    "name": "web_course",
                    "terrain": {"kind": "flat"},
                    "obstacles": [{"kind": "wall", "x": 2, "params": {}}],
                }
                request = Request(
                    editor.url + "api/export",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(request, timeout=2).read())
                self.assertIn("terrain.xml", result["xml"])
                self.assertIn("scene.json", result["scene"])
                scenes = json.loads(urlopen(editor.url + "api/scenes", timeout=2).read())
                self.assertEqual(scenes["scenes"][0]["name"], "web_course")
                loaded = editor.load_scene(result["id"])
                self.assertEqual(loaded["obstacles"][0]["kind"], "wall")
            finally:
                editor.close()

    def test_delete_is_scoped_to_selected_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            editor = SceneEditorServer(directory)
            scene = SceneSpec(name="delete_me")
            target, scene_id = editor.export_target(scene)
            export_scene_map(scene, target)
            editor.delete_scene(scene_id)
            self.assertFalse(target.exists())
            self.assertTrue(editor.output_dir.exists())
            editor.close()


if __name__ == "__main__":
    unittest.main()
