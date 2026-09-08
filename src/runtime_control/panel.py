"""Reusable browser control panel and JPEG frame transport for MuJoCo tools."""

from dataclasses import dataclass
from io import BytesIO
import atexit
import html
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ParameterSpec:
    """Description of one bounded numeric runtime parameter."""

    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    default: float
    group: str = "control"

    def __post_init__(self):
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError("parameter key must contain only letters, digits and '_'")
        if self.group not in {"commands", "control", "model"}:
            raise ValueError("parameter group must be commands, control or model")
        values = (self.minimum, self.maximum, self.step, self.default)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("parameter values must be finite")
        if self.maximum <= self.minimum or self.step <= 0:
            raise ValueError("parameter maximum/step must be positive")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError("parameter default must be inside its range")

    def schema_row(self):
        return [
            self.key, self.label, float(self.minimum), float(self.maximum),
            float(self.step),
        ]


@dataclass(frozen=True)
class ActionSpec:
    """Description of a custom panel button and optional keyboard shortcut."""

    key: str
    label: str
    shortcut: str = ""
    style: str = "secondary"

    def __post_init__(self):
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError("action key must contain only letters, digits and '_'")
        shortcut = self.shortcut.lower().strip()
        if len(shortcut) > 1:
            raise ValueError("action shortcut must be empty or one character")
        if self.style not in {"", "secondary", "warn", "danger"}:
            raise ValueError("unsupported action button style")
        object.__setattr__(self, "shortcut", shortcut)

    def as_dict(self):
        return {
            "label": self.label,
            "shortcut": self.shortcut,
            "style": self.style,
        }


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

GROUP_TITLES = {
    "commands": "◇ 运动命令",
    "control": "◇ 电机与控制",
    "model": "◇ 模型与地面",
}
MOTION_KEYS = {"w", "s", "a", "d", "q", "e"}
BUILTIN_ACTIONS = {"randomize", "nominal", "push", "reset", "stop"}


def encode_rgb_jpeg(rgb, quality=75):
    output = BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        output, format="JPEG", quality=int(quality), optimize=False
    )
    return output.getvalue()


def merge_parameter_schema(schema, parameters):
    """Return a schema with custom ParameterSpec entries added or replaced."""
    merged = {group: [list(row) for row in rows] for group, rows in schema.items()}
    for spec in parameters or ():
        if not isinstance(spec, ParameterSpec):
            spec = ParameterSpec(**dict(spec))
        for rows in merged.values():
            rows[:] = [row for row in rows if row[0] != spec.key]
        merged.setdefault(spec.group, []).append(spec.schema_row())
    return merged


def normalize_actions(actions):
    normalized = {}
    if isinstance(actions, dict):
        iterator = []
        for key, value in actions.items():
            if isinstance(value, ActionSpec):
                iterator.append(value)
            else:
                item = dict(value)
                item.setdefault("key", key)
                iterator.append(ActionSpec(**item))
    else:
        iterator = actions or ()
    for spec in iterator:
        if not isinstance(spec, ActionSpec):
            spec = ActionSpec(**dict(spec))
        if spec.key in BUILTIN_ACTIONS or spec.key == "close_panel":
            raise ValueError("custom action conflicts with a built-in action")
        normalized[spec.key] = spec.as_dict()
    shortcuts = [item["shortcut"] for item in normalized.values() if item["shortcut"]]
    if len(shortcuts) != len(set(shortcuts)):
        raise ValueError("custom action shortcuts must be unique")
    return normalized


def build_panel_html(
    title, schema, fps, maps=None, cameras=None, actions=None,
    snapshot_enabled=False, render_width=640, render_height=480,
):
    schema_json = json.dumps(schema, ensure_ascii=False)
    maps_json = json.dumps(maps or {}, ensure_ascii=False)
    cameras_json = json.dumps(cameras or {}, ensure_ascii=False)
    actions_json = json.dumps(actions or {}, ensure_ascii=False)
    safe_title = html.escape(str(title))
    render_width = max(1, int(render_width))
    render_height = max(1, int(render_height))
    custom_buttons = "".join(
        '<button class="{style}" onclick="action(\'{key}\')">{label}{shortcut}</button>'.format(
            style=html.escape(item.get("style", "secondary")),
            key=html.escape(key),
            label=html.escape(item.get("label", key)),
            shortcut=(
                " [" + html.escape(item["shortcut"].upper()) + "]"
                if item.get("shortcut") else ""
            ),
        )
        for key, item in (actions or {}).items()
    )
    snapshot_buttons = (
        '<button class="secondary" onclick="action(\'save_snapshot\')">保存参数</button>'
        '<button class="secondary" onclick="action(\'load_snapshot\')">读取参数</button>'
        if snapshot_enabled else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{safe_title}</title><style>
:root{{--bg:#07101d;--panel:#101c2d;--line:#263750;--text:#eef6ff;--muted:#8da2bd;--blue:#4da3ff;--cyan:#42dfd2}}
*{{box-sizing:border-box}}body{{font:14px Inter,"Noto Sans SC",system-ui,sans-serif;background:radial-gradient(circle at 15% -10%,#16375b 0,transparent 35%),radial-gradient(circle at 95% 10%,#25204e 0,transparent 30%),var(--bg);color:var(--text);margin:0;min-height:100vh}}main{{max-width:1580px;margin:auto;padding:22px}}
.topbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}}.brand small{{display:block;color:var(--cyan);font-size:11px;font-weight:800;letter-spacing:.18em}}h1{{font-size:24px;margin:5px 0 0}}h2{{font-size:14px;margin:0 0 16px}}.live{{display:flex;align-items:center;gap:8px;background:#101c2dcc;border:1px solid var(--line);padding:8px 12px;border-radius:999px}}.dot{{width:8px;height:8px;border-radius:50%;background:#36e49b;box-shadow:0 0 12px #36e49b}}
.layout{{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,1fr);gap:18px;align-items:start}}.stack,.parameter-grid{{display:grid;gap:18px}}.parameter-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}section{{background:linear-gradient(145deg,#142338e8,#0d1828ee);border:1px solid #263750cc;padding:18px;border-radius:16px;box-shadow:0 14px 40px #0005}}.viewer{{padding:10px}}.viewer-head{{display:flex;justify-content:space-between;align-items:center;padding:8px 8px 14px}}.fps{{font:12px ui-monospace;color:var(--cyan);background:#0b2b30;padding:5px 8px;border-radius:7px}}
.view{{display:block;width:100%;aspect-ratio:{render_width}/{render_height};object-fit:contain;background:#02060c;border:1px solid #2a3d58;border-radius:11px;cursor:grab;user-select:none;-webkit-user-drag:none;touch-action:none}}.view.dragging{{cursor:grabbing}}.camera-help{{padding:7px 9px 0;color:var(--muted);font-size:12px}}.toolbar{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:12px 4px 2px}}label{{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:8px;margin:15px 0;color:#c9d7e8}}output{{min-width:64px;text-align:center;font:600 12px ui-monospace;background:#213753;border:1px solid #345070;padding:4px 7px;border-radius:7px}}input[type=range]{{grid-column:1/-1;width:100%;height:5px;appearance:none;background:#263a55;border-radius:9px}}input[type=range]::-webkit-slider-thumb{{appearance:none;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,var(--cyan),var(--blue));border:2px solid #e7fbff}}select{{width:100%;color:var(--text);background:#172941;border:1px solid #334b69;border-radius:9px;padding:9px}}
.actions{{display:flex;flex-wrap:wrap;gap:9px;margin-top:18px;padding:16px;background:#0c1726aa;border:1px solid var(--line);border-radius:15px}}button{{border:1px solid #42648b;background:#284b70;color:#f7fbff;padding:10px 14px;border-radius:9px;font-weight:650;cursor:pointer}}button.secondary{{background:#182a40}}button.warn{{background:#705229}}button.danger{{background:#9d3040}}.keys{{color:var(--muted);font-size:12px}}kbd{{font:11px ui-monospace;background:#1b2b40;border:1px solid #3a506c;border-radius:5px;padding:2px 6px}}
@media(max-width:900px){{.layout,.parameter-grid{{grid-template-columns:1fr}}}}
</style><main><header class="topbar"><div class="brand"><small>ROBOTICS CONTROL CENTER</small><h1>{safe_title}</h1></div><div class="live"><i class="dot"></i>SIM LIVE</div></header>
<div class="layout"><div class="stack"><section class="viewer"><div class="viewer-head"><h2>MUJOCO 实时画面</h2><span class="fps">TARGET {fps:g} FPS</span></div><img class="view" id="simview" draggable="false"><div class="camera-help">左键拖动：旋转　右键拖动：平移　滚轮：缩放</div><div class="toolbar"><label id="maprow" hidden><span>地图</span><select id="mapselect"></select></label><label id="camerarow" hidden><span>视角</span><select id="cameraselect"></select></label></div></section><div class="parameter-grid"><section><h2>{GROUP_TITLES['commands']}</h2><div id="commands"></div></section><section><h2>{GROUP_TITLES['control']}</h2><div id="control"></div></section></div></div><aside><section><h2>{GROUP_TITLES['model']}</h2><div id="model"></div></section></aside></div>
<div class="actions"><button onclick="action('randomize')">⚔ 一键随机化</button><button class="secondary" onclick="action('nominal')">恢复标称</button><button class="warn" onclick="action('push')">随机推力</button><button class="secondary" onclick="action('reset')">重置机器人</button>{snapshot_buttons}{custom_buttons}<button class="danger" onclick="action('stop')">急停</button><button class="secondary" onclick="closePanel()">关闭页面</button></div><p class="keys">键盘：<kbd>W/S</kbd> 前后　<kbd>A/D</kbd> 横向　<kbd>Q/E</kbd> 转向</p>
<script>const specs={schema_json},maps={maps_json},cameras={cameras_json},customActions={actions_json};
for(const [group,items] of Object.entries(specs))for(const [key,name,min,max,step] of items){{const root=document.getElementById(group);if(!root)continue;let l=document.createElement('label');l.innerHTML=`<span>${{name}}</span><output id="o_${{key}}"></output><input id="${{key}}" type="range" min="${{min}}" max="${{max}}" step="${{step}}">`;root.append(l);let i=l.lastChild;i.onpointerdown=()=>i.dragging=true;i.onpointerup=i.onpointercancel=()=>i.dragging=false;i.oninput=()=>{{document.getElementById('o_'+key).value=i.value;send({{[key]:+i.value}})}}}}
let timer;function send(v){{clearTimeout(timer);timer=setTimeout(()=>fetch('/api/state',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(v)}}),35)}}function action(name){{fetch('/api/action',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:name}})}})}}
function refresh(){{fetch('/api/state').then(r=>r.json()).then(s=>{{for(const[k,v]of Object.entries(s)){{let i=document.getElementById(k);if(i&&!i.dragging){{i.value=v;document.getElementById('o_'+k).value=v}}}}}})}}refresh();setInterval(refresh,150);
const mapselect=document.getElementById('mapselect');if(Object.keys(maps).length){{document.getElementById('maprow').hidden=false;for(const[k,v]of Object.entries(maps)){{let o=document.createElement('option');o.value=k;o.textContent=v;mapselect.append(o)}}mapselect.onchange=()=>{{releaseMotionKeys();fetch('/api/map',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{map:mapselect.value}})}})}};setInterval(()=>fetch('/api/map').then(r=>r.json()).then(s=>mapselect.value=s.map),300)}}
const cameraselect=document.getElementById('cameraselect');if(Object.keys(cameras).length){{document.getElementById('camerarow').hidden=false;for(const[k,v]of Object.entries(cameras)){{let o=document.createElement('option');o.value=k;o.textContent=v;cameraselect.append(o)}}let free=document.createElement('option');free.value='free';free.textContent='鼠标自由视角';cameraselect.append(free);cameraselect.onchange=()=>{{releaseMotionKeys();fetch('/api/camera',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{camera:cameraselect.value}})}})}};setInterval(()=>fetch('/api/camera').then(r=>r.json()).then(s=>cameraselect.value=s.camera),300)}}
const view=document.getElementById('simview');function connectFrameStream(){{view.src='/api/stream.mjpg?t='+Date.now()}}view.onerror=()=>setTimeout(connectFrameStream,250);connectFrameStream();
let cameraDrag=null,pendingCamera=null,cameraFrame=0;function sendCameraMove(move){{pendingCamera=move;if(cameraFrame)return;cameraFrame=requestAnimationFrame(()=>{{cameraFrame=0;let current=pendingCamera;pendingCamera=null;fetch('/api/camera_move',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(current)}})}})}}view.oncontextmenu=e=>e.preventDefault();view.onpointerdown=e=>{{if(e.button!==0&&e.button!==2)return;e.preventDefault();view.setPointerCapture(e.pointerId);cameraDrag={{x:e.clientX,y:e.clientY,button:e.button}};view.classList.add('dragging')}};view.onpointermove=e=>{{if(!cameraDrag)return;let dx=(e.clientX-cameraDrag.x)/Math.max(1,view.clientWidth),dy=(e.clientY-cameraDrag.y)/Math.max(1,view.clientHeight);cameraDrag.x=e.clientX;cameraDrag.y=e.clientY;if(dx||dy)sendCameraMove({{action:cameraDrag.button===0?'rotate':'pan',dx,dy}})}};view.onpointerup=view.onpointercancel=()=>{{cameraDrag=null;view.classList.remove('dragging')}};view.addEventListener('wheel',e=>{{e.preventDefault();sendCameraMove({{action:'zoom',dx:0,dy:Math.sign(e.deltaY)*0.12}})}},{{passive:false}});
const motionKeys=new Set(['w','s','a','d','q','e']),keyAliases={{arrowup:'w',arrowdown:'s',arrowleft:'a',arrowright:'d'}},actionKeys=new Map(Object.entries(customActions).filter(([k,v])=>v.shortcut).map(([k,v])=>[v.shortcut,k]));function normalizedKey(e){{let k=e.key.toLowerCase();return keyAliases[k]||k}}function sendKey(key,pressed){{fetch('/api/key',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{key,pressed}})}})}}function releaseMotionKeys(){{for(const k of motionKeys)sendKey(k,false)}}function captureKey(e,pressed){{const k=normalizedKey(e);if(motionKeys.has(k)){{e.preventDefault();if(!e.repeat||!pressed)sendKey(k,pressed);return}}if(pressed&&!e.repeat&&actionKeys.has(k)){{e.preventDefault();action(actionKeys.get(k))}}}}addEventListener('keydown',e=>captureKey(e,true),true);addEventListener('keyup',e=>captureKey(e,false),true);addEventListener('blur',releaseMotionKeys);document.addEventListener('visibilitychange',()=>{{if(document.hidden)releaseMotionKeys()}});
async function closePanel(){{releaseMotionKeys();try{{await fetch('/api/action',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'close_panel'}})}})}}catch(e){{}}window.close()}}
</script></main></html>"""


class RuntimeControlPanel:
    """Thread-safe state, validated HTTP endpoints and browser lifecycle."""

    def __init__(
        self, initial_state, ui_config=None, randomization=None, schema=None,
        actions=None, random_seed=None, snapshot_path=None,
    ):
        ui_config = ui_config or {}
        self.schema = schema or DEFAULT_SCHEMA
        self.bounds = {
            row[0]: (float(row[2]), float(row[3]))
            for rows in self.schema.values() for row in rows
        }
        self.state = {
            name: self._bounded_value(name, value)
            for name, value in initial_state.items()
        }
        self.nominal_state = dict(self.state)
        self.randomization = dict(randomization or {})
        self.rng = np.random.default_rng(random_seed)
        self.custom_actions = normalize_actions(actions)
        self.allowed_actions = BUILTIN_ACTIONS | set(self.custom_actions)
        if snapshot_path:
            self.allowed_actions |= {"save_snapshot", "load_snapshot"}
        self.snapshot_path = Path(snapshot_path).expanduser().resolve() if snapshot_path else None
        self.actions, self.pressed_keys = set(), set()
        self.maps = dict(ui_config.get("maps", {}))
        self.selected_map = ui_config.get("default_map", next(iter(self.maps), None))
        self.pending_map = None
        self.cameras = dict(ui_config.get("cameras", {}))
        self.selected_camera = ui_config.get("default_camera", next(iter(self.cameras), None))
        self.pending_camera = None
        self.camera_moves = []
        self.stop_on_panel_close = bool(ui_config.get("stop_on_panel_close", True))
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.frame_condition = threading.Condition(self.lock)
        self.frame_bytes, self.frame_sequence = None, 0
        self.last_frame_request = 0.0
        self._browser_lock = threading.Lock()
        self._browser_process = None
        self._browser_profile = None
        self._server_thread = None
        self._closed = False
        title = ui_config.get("title", "MuJoCo 实时调参")
        fps = float(ui_config.get("fps", 120))
        self.html = build_panel_html(
            title, self.schema, fps, self.maps, self.cameras,
            self.custom_actions, self.snapshot_path is not None,
            ui_config.get("width", 640), ui_config.get("height", 480),
        )
        self._start_server(ui_config)
        atexit.register(self.close)

    def _bounded_value(self, name, value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("runtime parameter %s must be finite" % name)
        limits = self.bounds.get(name)
        if limits is not None:
            number = min(limits[1], max(limits[0], number))
        return number

    def _start_server(self, ui_config):
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def reply_json(self, payload, status=200):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def reply_error(self, status, message):
                self.reply_json({"ok": False, "error": message}, status)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/api/state":
                    with panel.lock:
                        payload = dict(panel.state)
                    self.reply_json(payload)
                    return
                if path == "/api/map":
                    with panel.lock:
                        selected = panel.selected_map
                    self.reply_json({"map": selected})
                    return
                if path == "/api/camera":
                    with panel.lock:
                        selected = panel.selected_camera
                    self.reply_json({"camera": selected})
                    return
                if path == "/api/stream.mjpg":
                    self._stream_frames()
                    return
                if path == "/api/frame.jpg":
                    with panel.lock:
                        panel.last_frame_request = time.monotonic()
                        frame = panel.frame_bytes
                    if frame is None:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    return
                if path not in {"/", "/index.html"}:
                    self.reply_error(404, "unknown endpoint")
                    return
                body = panel.html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream_frames(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                sequence = -1
                try:
                    while not panel.stop_event.is_set():
                        with panel.frame_condition:
                            panel.last_frame_request = time.monotonic()
                            panel.frame_condition.wait_for(
                                lambda: panel.frame_sequence != sequence or panel.stop_event.is_set(),
                                timeout=1.0,
                            )
                            if panel.stop_event.is_set():
                                break
                            if panel.frame_sequence == sequence:
                                continue
                            sequence = panel.frame_sequence
                            frame = panel.frame_bytes
                        if frame is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + ("Content-Length: %d\r\n\r\n" % len(frame)).encode()
                            + frame + b"\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 65536:
                        self.reply_error(413, "request body too large")
                        return
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("JSON body must be an object")
                except (ValueError, json.JSONDecodeError) as exc:
                    self.reply_error(400, str(exc))
                    return
                try:
                    should_close = panel._handle_post(path, payload)
                except ValueError as exc:
                    self.reply_error(400, str(exc))
                    return
                if should_close is None:
                    self.reply_error(404, "unknown endpoint")
                    return
                self.reply_json({"ok": True})
                if should_close:
                    threading.Thread(target=panel._close_after_reply, daemon=True).start()

            def log_message(self, *_):
                pass

        host = ui_config.get("host", "127.0.0.1")
        port = int(ui_config.get("port", 8765))
        try:
            self.server = ThreadingHTTPServer((host, port), Handler)
        except OSError as exc:
            raise OSError(
                "cannot start MuJoCo runtime panel on %s:%d: %s" % (host, port, exc)
            ) from exc
        self.server.daemon_threads = True
        self.server.allow_reuse_address = True
        actual_port = self.server.server_address[1]
        self.url = "http://%s:%d" % (host, actual_port)
        self._server_thread = threading.Thread(
            target=self.server.serve_forever, name="mujoco-runtime-http", daemon=True
        )
        self._server_thread.start()
        print("[UI] 实时调参面板: %s" % self.url)

    def _handle_post(self, path, payload):
        close_requested = False
        with self.lock:
            if path == "/api/action":
                action = str(payload.get("action", ""))
                if action == "close_panel":
                    close_requested = True
                elif action not in self.allowed_actions:
                    raise ValueError("unknown action: %s" % action)
                else:
                    self.actions.add(action)
            elif path == "/api/map":
                selected = str(payload.get("map", ""))
                if selected not in self.maps:
                    raise ValueError("unknown map: %s" % selected)
                self.pressed_keys.clear()
                if selected != self.selected_map:
                    self.selected_map = selected
                    self.pending_map = selected
            elif path == "/api/camera":
                selected = str(payload.get("camera", ""))
                if selected not in self.cameras and selected != "free":
                    raise ValueError("unknown camera: %s" % selected)
                self.pressed_keys.clear()
                if selected != self.selected_camera:
                    self.selected_camera = selected
                    self.pending_camera = selected
            elif path == "/api/camera_move":
                action = str(payload.get("action", ""))
                if action not in {"rotate", "pan", "zoom"}:
                    raise ValueError("unknown camera movement: %s" % action)
                dx = float(payload.get("dx", 0.0))
                dy = float(payload.get("dy", 0.0))
                if not math.isfinite(dx) or not math.isfinite(dy):
                    raise ValueError("camera movement must be finite")
                dx = min(1.0, max(-1.0, dx))
                dy = min(1.0, max(-1.0, dy))
                self.selected_camera = "free"
                self.camera_moves.append((action, dx, dy))
            elif path == "/api/key":
                key = str(payload.get("key", "")).lower()
                if key not in MOTION_KEYS:
                    raise ValueError("unsupported motion key")
                if payload.get("pressed"):
                    self.pressed_keys.add(key)
                else:
                    self.pressed_keys.discard(key)
            elif path == "/api/state":
                for key, value in payload.items():
                    if key not in self.state:
                        raise ValueError("unknown runtime parameter: %s" % key)
                    self.state[key] = self._bounded_value(key, value)
            else:
                return None
        return close_requested

    def open_browser(self):
        """Open an isolated app window that can be closed independently."""
        chrome = next((shutil.which(name) for name in (
            "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"
        ) if shutil.which(name)), None)
        if chrome is None:
            webbrowser.open(self.url, new=2)
            return
        profile = tempfile.mkdtemp(prefix="himloco_runtime_ui_")
        command = [
            chrome, "--user-data-dir=" + profile, "--app=" + self.url,
            "--new-window", "--window-size=1560,980", "--no-first-run",
            "--no-default-browser-check", "--disable-session-crashed-bubble",
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

    def _close_after_reply(self):
        time.sleep(0.1)
        if self.stop_on_panel_close:
            self.close()
        else:
            self.close_browser()

    def close_browser(self):
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

    def close(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self.stop_event.set()
            self.pressed_keys.clear()
            self.frame_condition.notify_all()
        self.close_browser()
        self.server.shutdown()
        self.server.server_close()
        if self._server_thread is not None and threading.current_thread() is not self._server_thread:
            self._server_thread.join(timeout=2.0)

    def is_running(self):
        return not self.stop_event.is_set()

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def consume_actions(self):
        with self.lock:
            actions = set(self.actions)
            self.actions.clear()
            return actions

    def keyboard_snapshot(self):
        with self.lock:
            return set(self.pressed_keys)

    def consume_map_change(self):
        with self.lock:
            selected, self.pending_map = self.pending_map, None
            return selected

    def consume_camera_change(self):
        with self.lock:
            selected, self.pending_camera = self.pending_camera, None
            return selected

    def consume_camera_moves(self):
        """Return and clear normalized browser mouse camera movements."""
        with self.lock:
            moves = list(self.camera_moves)
            self.camera_moves.clear()
            return moves

    def randomize(self):
        with self.lock:
            for name, limits in self.randomization.items():
                if name in self.state and len(limits) == 2:
                    value = self.rng.uniform(float(limits[0]), float(limits[1]))
                    self.state[name] = self._bounded_value(name, value)

    def restore_nominal(self):
        with self.lock:
            self.state.update(self.nominal_state)

    def save_snapshot(self, path=None):
        candidate = path if path is not None else self.snapshot_path
        if candidate is None:
            raise ValueError("no runtime snapshot path configured")
        target = Path(candidate).expanduser().resolve()
        with self.lock:
            payload = {
                "version": 1,
                "state": dict(self.state),
                "map": self.selected_map,
                "camera": self.selected_camera,
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def load_snapshot(self, path=None):
        candidate = path if path is not None else self.snapshot_path
        if candidate is None:
            raise ValueError("no runtime snapshot path configured")
        target = Path(candidate).expanduser().resolve()
        payload = json.loads(target.read_text(encoding="utf-8"))
        state = payload.get("state", {})
        with self.lock:
            for name, value in state.items():
                if name in self.state:
                    self.state[name] = self._bounded_value(name, value)
            selected_map = payload.get("map")
            if selected_map in self.maps and selected_map != self.selected_map:
                self.selected_map = selected_map
                self.pending_map = selected_map
            selected_camera = payload.get("camera")
            if selected_camera in self.cameras and selected_camera != self.selected_camera:
                self.selected_camera = selected_camera
                self.pending_camera = selected_camera
        return target

    def set_frame(self, frame):
        with self.frame_condition:
            self.frame_bytes = frame
            self.frame_sequence += 1
            self.frame_condition.notify_all()

    def wants_frames(self):
        with self.lock:
            return time.monotonic() - self.last_frame_request < 2.0
