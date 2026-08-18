"""Reusable browser control panel and JPEG frame transport for MuJoCo tools."""

from io import BytesIO
import atexit
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser

import numpy as np
from PIL import Image


DEFAULT_SCHEMA = {
    "commands": [
        ["linear_x", "W/S 前后速度档位", 0, 2, 0.05],
        ["linear_y", "A/D 横向速度档位", 0, 1.5, 0.05],
        ["yaw", "Q/E 偏航速度档位", 0, 3.14, 0.05],
        ["height", "Command Height", 0.2, 0.8, 0.01],
    ],
    "control": [
        ["kp", "Kp", 0, 250, 1],
        ["kd", "Kd", 0, 15, 0.1],
        ["torque_limit", "力矩上限", 0, 200, 1],
        ["motor_strength", "电机强度", 0.2, 1.5, 0.01],
        ["motor_delay_ms", "电机延迟 (ms)", 0, 100, 1],
    ],
    "model": [
        ["mass_scale", "整机质量倍率", 0.5, 2, 0.01],
        ["payload_mass", "机身负载 (kg)", 0, 10, 0.1],
        ["friction_scale", "摩擦倍率", 0.2, 2, 0.01],
        ["gravity_z", "重力 Z (m/s²)", -15, -3, 0.05],
    ],
}


def encode_rgb_jpeg(rgb, quality=75):
    output = BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        output, format="JPEG", quality=int(quality), optimize=False
    )
    return output.getvalue()


def build_panel_html(title, schema, fps, maps=None, cameras=None):
    schema_json = json.dumps(schema, ensure_ascii=False)
    maps_json = json.dumps(maps or {}, ensure_ascii=False)
    cameras_json = json.dumps(cameras or {}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
:root{{--bg:#07101d;--panel:#101c2d;--panel2:#142338;--line:#263750;--text:#eef6ff;--muted:#8da2bd;--blue:#4da3ff;--cyan:#42dfd2;--purple:#9b87ff;--danger:#ff5263}}
*{{box-sizing:border-box}}body{{font:14px Inter,"Noto Sans SC",system-ui,sans-serif;background:radial-gradient(circle at 15% -10%,#16375b 0,transparent 35%),radial-gradient(circle at 95% 10%,#25204e 0,transparent 30%),var(--bg);color:var(--text);margin:0;min-height:100vh}}main{{max-width:1580px;margin:auto;padding:22px}}
.topbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}}.brand small{{display:block;color:var(--cyan);font-size:11px;font-weight:800;letter-spacing:.18em;text-transform:uppercase}}h1{{font-size:24px;margin:5px 0 0;letter-spacing:-.03em}}h2{{font-size:14px;margin:0 0 16px;letter-spacing:.04em}}.live{{display:flex;align-items:center;gap:8px;color:#b9c9dc;background:#101c2dcc;border:1px solid var(--line);padding:8px 12px;border-radius:999px}}.dot{{display:block;width:8px;height:8px;border-radius:50%;background:#36e49b;box-shadow:0 0 12px #36e49b;animation:pulse 1.8s infinite}}@keyframes pulse{{50%{{opacity:.45}}}}
.layout{{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,1fr);gap:18px;align-items:start}}.stack{{display:grid;gap:18px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}section{{background:linear-gradient(145deg,#142338e8,#0d1828ee);border:1px solid #263750cc;padding:18px;border-radius:16px;box-shadow:0 14px 40px #0005;backdrop-filter:blur(12px)}}.viewer{{padding:10px;overflow:hidden}}.viewer-head{{display:flex;justify-content:space-between;align-items:center;padding:8px 8px 14px}}.fps{{font:12px ui-monospace;color:var(--cyan);background:#0b2b30;padding:5px 8px;border-radius:7px}}
.view{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#02060c;border:1px solid #2a3d58;border-radius:11px}}.toolbar{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:12px 4px 2px}}.toolbar label{{grid-template-columns:54px 1fr;margin:0}}label{{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:8px;margin:15px 0;color:#c9d7e8}}output{{min-width:64px;text-align:center;color:#fff;font:600 12px ui-monospace;background:#213753;border:1px solid #345070;padding:4px 7px;border-radius:7px}}input[type=range]{{grid-column:1/-1;width:100%;height:5px;appearance:none;background:#263a55;border-radius:9px;outline:none}}input[type=range]::-webkit-slider-thumb{{appearance:none;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,var(--cyan),var(--blue));border:2px solid #e7fbff;box-shadow:0 0 0 4px #42dfd220;cursor:pointer}}select{{width:100%;color:var(--text);background:#172941;border:1px solid #334b69;border-radius:9px;padding:9px 34px 9px 10px;outline:none}}
.actions{{display:flex;flex-wrap:wrap;gap:9px;margin-top:18px;padding:16px;background:#0c1726aa;border:1px solid var(--line);border-radius:15px}}button{{border:1px solid #42648b;background:linear-gradient(180deg,#284b70,#1c3857);color:#f7fbff;padding:10px 14px;border-radius:9px;font-weight:650;cursor:pointer;transition:.16s transform,.16s filter}}button:hover{{transform:translateY(-1px);filter:brightness(1.17)}}button.secondary{{background:#182a40;border-color:#344b67}}button.warn{{background:linear-gradient(180deg,#705229,#543b1f);border-color:#98703a}}button.danger{{margin-left:auto;background:linear-gradient(180deg,#b53747,#7d2633);border-color:#db5868}}button.close-panel{{background:#202a38;border-color:#536174}}.keys{{color:var(--muted);font-size:12px;margin:12px 2px 0}}kbd{{font:11px ui-monospace;color:#dceaff;background:#1b2b40;border:1px solid #3a506c;border-bottom-width:2px;border-radius:5px;padding:2px 6px}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}.grid{{grid-template-columns:1fr}}}}@media(max-width:520px){{main{{padding:14px}}.toolbar{{grid-template-columns:1fr}}button.danger{{margin-left:0}}}}
</style><main><header class="topbar"><div class="brand"><small>Robotics Control Center</small><h1>{title}</h1></div><div class="live"><i class="dot"></i>SIM LIVE</div></header>
<div class="layout"><div class="stack"><section class="viewer"><div class="viewer-head"><h2>MUJOCO 实时画面</h2><span class="fps">TARGET {fps:g} FPS</span></div><img class="view" id="simview" alt="正在等待 MuJoCo 画面…"><div class="toolbar"><label id="maprow" hidden><span>地图</span><select id="mapselect"></select></label><label id="camerarow" hidden><span>视角</span><select id="cameraselect"></select></label></div></section><div class="grid"><section><h2>◇ 运动命令</h2><div id="commands"></div></section><section><h2>◇ 电机与控制</h2><div id="control"></div></section></div></div><aside><section><h2>◇ 模型与地面</h2><div id="model"></div></section></aside></div>
<div class="actions"><button onclick="action('randomize')">⚄ 一键随机化</button><button class="secondary" onclick="action('nominal')">↺ 恢复标称</button><button class="warn" onclick="action('push')">↯ 随机推力</button><button class="secondary" onclick="action('reset')">⟲ 重置机器人</button><button class="danger" onclick="action('stop')">■ 急停</button><button class="close-panel" onclick="closePanel()">✕ 关闭页面</button></div><p class="keys">键盘控制：<kbd>W</kbd>/<kbd>S</kbd> 前后　<kbd>A</kbd>/<kbd>D</kbd> 横向　<kbd>Q</kbd>/<kbd>E</kbd> 转向</p>
<script>const specs={schema_json},maps={maps_json},cameras={cameras_json};
for(const [group,items] of Object.entries(specs))for(const [key,name,min,max,step] of items){{let l=document.createElement('label');l.innerHTML=`<span>${{name}}</span><output id="o_${{key}}"></output><input id="${{key}}" type="range" min="${{min}}" max="${{max}}" step="${{step}}">`;document.getElementById(group).append(l);let i=l.lastChild;i.onpointerdown=()=>i.dragging=true;i.onpointerup=i.onpointercancel=()=>i.dragging=false;i.oninput=()=>{{document.getElementById('o_'+key).value=i.value;send({{[key]:+i.value}})}}}}
let timer;function send(v){{clearTimeout(timer);timer=setTimeout(()=>fetch('/api/state',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(v)}}),35)}}function action(name){{fetch('/api/action',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:name}})}})}}
function refresh(){{fetch('/api/state').then(r=>r.json()).then(s=>{{for(const[k,v]of Object.entries(s)){{let i=document.getElementById(k);if(i&&!i.dragging){{i.value=v;document.getElementById('o_'+k).value=v}}}}}})}}refresh();setInterval(refresh,100);
const mapselect=document.getElementById('mapselect');if(Object.keys(maps).length){{document.getElementById('maprow').hidden=false;for(const[k,v]of Object.entries(maps)){{let o=document.createElement('option');o.value=k;o.textContent=v;mapselect.append(o)}}mapselect.onpointerdown=releaseMotionKeys;mapselect.onchange=()=>{{releaseMotionKeys();fetch('/api/map',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{map:mapselect.value}})}})}};setInterval(()=>fetch('/api/map').then(r=>r.json()).then(s=>mapselect.value=s.map),250)}}
const cameraselect=document.getElementById('cameraselect');if(Object.keys(cameras).length){{document.getElementById('camerarow').hidden=false;for(const[k,v]of Object.entries(cameras)){{let o=document.createElement('option');o.value=k;o.textContent=v;cameraselect.append(o)}}cameraselect.onpointerdown=releaseMotionKeys;cameraselect.onchange=()=>{{releaseMotionKeys();fetch('/api/camera',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{camera:cameraselect.value}})}})}};setInterval(()=>fetch('/api/camera').then(r=>r.json()).then(s=>cameraselect.value=s.camera),250)}}
const view=document.getElementById('simview');function connectFrameStream(){{view.src='/api/stream.mjpg?t='+Date.now()}}view.onerror=()=>setTimeout(connectFrameStream,250);connectFrameStream();
const motionKeys=new Set(['w','s','a','d','q','e']),keyAliases={{arrowup:'w',arrowdown:'s',arrowleft:'a',arrowright:'d'}};function normalizedMotionKey(e){{let k=e.key.toLowerCase();return keyAliases[k]||k}}function sendKey(key,pressed){{fetch('/api/key',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{key,pressed}})}})}}function releaseMotionKeys(){{for(const k of motionKeys)sendKey(k,false)}}function captureMotion(e,pressed){{let k=normalizedMotionKey(e);if(!motionKeys.has(k))return;e.preventDefault();e.stopPropagation();if(document.activeElement instanceof HTMLSelectElement)document.activeElement.blur();if(!e.repeat||!pressed)sendKey(k,pressed)}}addEventListener('keydown',e=>captureMotion(e,true),true);addEventListener('keyup',e=>captureMotion(e,false),true);addEventListener('blur',releaseMotionKeys);document.addEventListener('visibilitychange',()=>{{if(document.hidden)releaseMotionKeys()}});
async function closePanel(){{for(const k of motionKeys)sendKey(k,false);try{{await fetch('/api/action',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'close_panel'}})}})}}catch(e){{}}window.close()}}
</script></main></html>"""


class RuntimeControlPanel:
    """Thread-safe state, HTTP endpoints and browser key handling."""

    def __init__(self, initial_state, ui_config=None, randomization=None, schema=None):
        ui_config = ui_config or {}
        self.state = {name: float(value) for name, value in initial_state.items()}
        self.nominal_state = dict(self.state)
        self.randomization = randomization or {}
        self.actions, self.pressed_keys = set(), set()
        self.maps = dict(ui_config.get("maps", {}))
        self.selected_map = ui_config.get(
            "default_map", next(iter(self.maps), None)
        )
        self.pending_map = None
        self.cameras = dict(ui_config.get("cameras", {}))
        self.selected_camera = ui_config.get(
            "default_camera", next(iter(self.cameras), None)
        )
        self.pending_camera = None
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.frame_bytes, self.frame_sequence = None, 0
        self.last_frame_request = 0.0
        self._browser_lock = threading.Lock()
        self._browser_process = None
        self._browser_profile = None
        title = ui_config.get("title", "MuJoCo 实时调参")
        fps = float(ui_config.get("fps", 120))
        self.html = build_panel_html(
            title, schema or DEFAULT_SCHEMA, fps, self.maps, self.cameras
        )
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                if self.path == "/api/state":
                    with panel.lock: payload = dict(panel.state)
                    self.reply(payload); return
                if self.path == "/api/map":
                    with panel.lock: selected = panel.selected_map
                    self.reply({"map": selected}); return
                if self.path == "/api/camera":
                    with panel.lock: selected = panel.selected_camera
                    self.reply({"camera": selected}); return
                if self.path.startswith("/api/stream.mjpg"):
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    sequence = -1
                    try:
                        while True:
                            with panel.frame_condition:
                                panel.last_frame_request = time.monotonic()
                                panel.frame_condition.wait_for(
                                    lambda: panel.frame_sequence != sequence,
                                    timeout=1.0,
                                )
                                if panel.frame_sequence == sequence:
                                    continue
                                sequence = panel.frame_sequence
                                frame = panel.frame_bytes
                            if frame is None:
                                continue
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                                + frame
                                + b"\r\n"
                            )
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path.startswith("/api/frame.jpg"):
                    with panel.lock: panel.last_frame_request=time.monotonic(); frame=panel.frame_bytes
                    if frame is None: self.send_response(204); self.end_headers(); return
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(frame))); self.end_headers(); self.wfile.write(frame); return
                body = panel.html.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                close_requested = False
                with panel.lock:
                    if self.path == "/api/action":
                        action = payload.get("action")
                        if action == "close_panel": close_requested = True
                        else: panel.actions.add(action)
                    elif self.path == "/api/map":
                        selected = str(payload.get("map", ""))
                        panel.pressed_keys.clear()
                        if selected in panel.maps and selected != panel.selected_map:
                            panel.selected_map = selected
                            panel.pending_map = selected
                    elif self.path == "/api/camera":
                        selected = str(payload.get("camera", ""))
                        panel.pressed_keys.clear()
                        if selected in panel.cameras and selected != panel.selected_camera:
                            panel.selected_camera = selected
                            panel.pending_camera = selected
                    elif self.path == "/api/key":
                        key=str(payload.get("key", "")).lower()
                        if key in {"w","s","a","d","q","e"}:
                            panel.pressed_keys.add(key) if payload.get("pressed") else panel.pressed_keys.discard(key)
                    else:
                        for key,value in payload.items():
                            if key in panel.state: panel.state[key]=float(value)
                self.reply({"ok": True})
                if close_requested:
                    threading.Thread(
                        target=panel._close_browser_after_reply, daemon=True
                    ).start()
            def log_message(self, *_): pass

        host, port = ui_config.get("host", "127.0.0.1"), int(ui_config.get("port", 8765))
        self.url = f"http://{host}:{port}"
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[UI] 实时调参面板: {self.url}")

    def open_browser(self):
        """Open an isolated app window that can be closed without touching other tabs."""
        chrome = next(
            (
                path for name in (
                    "google-chrome", "google-chrome-stable",
                    "chromium", "chromium-browser",
                )
                if (path := shutil.which(name)) is not None
            ),
            None,
        )
        if chrome is None:
            webbrowser.open(self.url, new=2)
            return
        profile = tempfile.mkdtemp(prefix="himloco_runtime_ui_")
        command = [
            chrome,
            f"--user-data-dir={profile}",
            f"--app={self.url}",
            "--new-window",
            "--window-size=1560,980",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
        ]
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError:
            shutil.rmtree(profile, ignore_errors=True)
            webbrowser.open(self.url, new=2)
            return
        with self._browser_lock:
            self._browser_process = process
            self._browser_profile = profile
        atexit.register(self.close_browser)

    def _close_browser_after_reply(self):
        time.sleep(0.15)
        self.close_browser()

    def close_browser(self):
        """Close only the browser app window opened for this panel."""
        with self._browser_lock:
            process = self._browser_process
            profile = self._browser_profile
            self._browser_process = None
            self._browser_profile = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if profile:
            shutil.rmtree(profile, ignore_errors=True)

    def snapshot(self):
        with self.lock: return dict(self.state)
    def consume_actions(self):
        with self.lock: actions=set(self.actions); self.actions.clear(); return actions
    def keyboard_snapshot(self):
        with self.lock: return set(self.pressed_keys)
    def consume_map_change(self):
        with self.lock:
            selected = self.pending_map
            self.pending_map = None
            return selected
    def consume_camera_change(self):
        with self.lock:
            selected = self.pending_camera
            self.pending_camera = None
            return selected
    def randomize(self):
        with self.lock:
            for name,limits in self.randomization.items():
                if name in self.state and len(limits)==2: self.state[name]=float(np.random.uniform(*limits))
    def restore_nominal(self):
        with self.lock: self.state.update(self.nominal_state)
    def set_frame(self, frame):
        with self.frame_condition:
            self.frame_bytes = frame
            self.frame_sequence += 1
            self.frame_condition.notify_all()
    def wants_frames(self):
        with self.lock: return time.monotonic()-self.last_frame_request < 2.0
