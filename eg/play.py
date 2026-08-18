"""
Dog Sim2Sim — IsaacGym → MuJoCo

关节顺序:
  MuJoCo XML 顺序 (即 URDF 顺序): FL(0-2), FR(3-5), RR(6-8), RL(9-11)
  IsaacGym dof 顺序 (内部重排):    FL(0-2), FR(3-5), RL(6-8), RR(9-11)
  → 后两条腿反了! 需要 RL/RR 映射!


   sudo chmod 666 /dev/input/event*
"""
import time
from pathlib import Path
import sys
import tempfile
import mujoco
import numpy as np
import onnxruntime as ort
import yaml


# 所有仓库内资源都从本文件定位，运行时不依赖当前工作目录或本机绝对路径。
DEMO_DIR = Path(__file__).resolve().parent
RUNTIME_CONTROL_DIR = DEMO_DIR.parent
MUJOCO_DIR = RUNTIME_CONTROL_DIR.parent
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from runtime_control import (
    MapSpec,
    MotorCommandDelay,
    RuntimeControl,
    compose_scene,
    compute_pd_torques,
    make_runtime_config,
    make_standard_robot_cameras,
    scale_torque_limits,
    setup_tracking_camera,
    standard_camera_options,
    viewer_context,
)


MAP_DIR = RUNTIME_CONTROL_DIR / "maps"
DEFAULT_CONFIG = DEMO_DIR / "dog.yaml"
DEFAULT_ONNX = DEMO_DIR / "model_3400.onnx"
DEFAULT_ROBOT_XML = DEMO_DIR / "dog" / "xml" / "dog_terrain.xml"
MAP_SPECS = {
    "rc26_track": MapSpec(MAP_DIR / "26rc_track.xml", exclude_bodies=("trunk",)),
    "race_track": MapSpec(MAP_DIR / "race_track.xml"),
    "stairs": MapSpec(MAP_DIR / "stairs.xml"),
    "cross_stairs": MapSpec(MAP_DIR / "cross_stairs.xml"),
    "cross_slope": MapSpec(MAP_DIR / "cross_slope.xml"),
    "google_barkour": MapSpec(MAP_DIR / "google_barkour.xml"),
    "gap_jump": MapSpec(MAP_DIR / "gap_jump.xml"),
    "hurdles": MapSpec(MAP_DIR / "hurdles.xml"),
    "suspended_steps": MapSpec(MAP_DIR / "suspended_steps.xml"),
    "perlin_rough": MapSpec(MAP_DIR / "perlin_rough.xml"),
    "dynamic_obstacles": MapSpec(MAP_DIR / "dynamic_obstacles.xml"),
}

# 由 compose_scene 注入 trunk，位置和朝向随机器人运动。
ROBOT_CAMERAS = make_standard_robot_cameras(prefix="dog")
CAMERA_OPTIONS = standard_camera_options(prefix="dog")


# ============================================================
#  RL/RR 映射 — IsaacGym 内部重排关节顺序
# ============================================================
# MuJoCo qpos[7:19] 顺序: FL(0-2), FR(3-5), RR(6-8), RL(9-11)
# IsaacGym dof 顺序:      FL(0-2), FR(3-5), RL(6-8), RR(9-11)
#
# MuJoCo[i] -> Isaac[j]:
#   MuJoCo 0-5  (FL,FR) -> Isaac 0-5  (FL,FR)  不变
#   MuJoCo 6-8  (RR)    -> Isaac 9-11 (RR)
#   MuJoCo 9-11 (RL)    -> Isaac 6-8  (RL)

MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]

# 默认关节角度 — IsaacGym dof 顺序: FL, FR, RL, RR
# 来自 dog_config.py default_joint_angles
# FL: hip=-0.1, thigh=-0.8, calf=-1.5
# FR: hip=0.1,  thigh=0.8,  calf=1.5
# RL: hip=0.1,  thigh=-1.0, calf=-1.5
# RR: hip=-0.1, thigh=1.0,  calf=1.5
DEFAULT_ANGLES_ISAAC = np.array([
    -0.1, -0.8, -1.5,    # FL (Isaac dof 0-2)
     0.1,  0.8,  1.5,    # FR (Isaac dof 3-5)
     0.1, -1.0, -1.5,    # RL (Isaac dof 6-8)
    -0.1,  1.0,  1.5,    # RR (Isaac dof 9-11)
], dtype=np.float64)

# MuJoCo 顺序: FL, FR, RR, RL — 通过映射从 Isaac 顺序得到
DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]

# 力矩限制 (来自 URDF actuatorfrcrange / dog.xml)
TAU_LIMIT_HIP_THIGH = 23.7   # hip, thigh 关节力矩限制 [Nm]
TAU_LIMIT_CALF = 35.55        # calf 关节力矩限制 [Nm]
OUTPUT_PRINT_SCALE = 0.25


def quat_rotate_inverse(q, v):
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class ObsHistoryBuffer:
    def __init__(self, history_len, single_obs_dim):
        self.history_len = history_len
        self.single_obs_dim = single_obs_dim
        self.total_dim = history_len * single_obs_dim
        self.buffer = np.zeros(self.total_dim, dtype=np.float32)

    def push(self, new_obs):
        self.buffer[self.single_obs_dim:] = self.buffer[:-self.single_obs_dim].copy()
        self.buffer[:self.single_obs_dim] = new_obs

    def get(self):
        return self.buffer.reshape(1, -1).copy()

    def reset(self):
        self.buffer[:] = 0.0


# ============================================================
#  键盘控制
#  移动键 (W/A/S/D/Q/E): evdev 监听按下/释放, 即按即动, 松手即停
#  功能键 (R/F/Z/T/Y/X): MuJoCo key_callback 单次触发
# ============================================================

# --- evdev: 可选的 Linux 全局键盘输入；浏览器控制不依赖它。 ---
try:
    import evdev
except ImportError:
    evdev = None
import threading

def _find_keyboards():
    """自动检测键盘设备 (支持 EV_KEY 且有字母键)"""
    keyboards = []
    if evdev is None:
        return keyboards
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            if evdev.ecodes.EV_KEY in caps:
                keys = caps[evdev.ecodes.EV_KEY]
                if evdev.ecodes.KEY_W in keys:
                    keyboards.append(dev)
                    continue
        except Exception:
            pass
    return keyboards

_pressed_keys = set()   # 当前正被按住的键 (小写字母)

# 方向键映射: evdev scancode → 方向字母
_DIR_SCANCODES = {} if evdev is None else {
    evdev.ecodes.KEY_W: 'w', evdev.ecodes.KEY_S: 's',
    evdev.ecodes.KEY_A: 'a', evdev.ecodes.KEY_D: 'd',
    evdev.ecodes.KEY_Q: 'q', evdev.ecodes.KEY_E: 'e',
    evdev.ecodes.KEY_UP: 'w', evdev.ecodes.KEY_DOWN: 's',
    evdev.ecodes.KEY_LEFT: 'a', evdev.ecodes.KEY_RIGHT: 'd',
}

def _evdev_keyboard_thread(dev):
    """后台线程: 读取 evdev 键盘事件"""
    try:
        for event in dev.read_loop():
            if event.type == evdev.ecodes.EV_KEY:
                scancode = event.code
                value = event.value  # 1=按下, 0=释放, 2=长按重复
                if scancode in _DIR_SCANCODES:
                    d = _DIR_SCANCODES[scancode]
                    if value == 1:      # 按下
                        _pressed_keys.add(d)
                    elif value == 0:    # 释放
                        _pressed_keys.discard(d)
    except Exception as e:
        print(f"[KEYBOARD] evdev 设备读取异常: {e}")

# 启动 evdev 键盘监听
_kb_devs = _find_keyboards()

if not _kb_devs and evdev is not None:
    # 权限不足时, 尝试用 sudo 打开已知键盘设备
    import glob, os
    _keyboard_event = "/dev/input/event3"  # AT Translated Set 2 keyboard
    if os.path.exists(_keyboard_event):
        try:
            _dev = evdev.InputDevice(_keyboard_event)
            _kb_devs = [_dev]
        except PermissionError:
            print(f"\n{'='*60}")
            print("[KEYBOARD] ⚠ 无法访问键盘设备 (Permission denied)")
            print(f"{'='*60}")
            print("  原因: Wayland 下 pynput/evdev 需要 /dev/input 权限")
            print("  修复方法 (任选一种):")
            print("")
            print("  方法1: 添加当前用户到 input 组 (推荐, 重启后永久生效)")
            print("    sudo usermod -aG input $USER")
            print("    然后注销并重新登录")
            print("")
            print("  方法2: 临时修改设备权限 (每次重启后需重新执行)")
            print("    sudo chmod 666 /dev/input/event*")
            print("")
            print("  方法3: 用 sudo 运行本脚本")
            print(f"    sudo python3 {os.path.abspath(__file__)} ...")
            print(f"{'='*60}\n")
        except Exception:
            pass

for _dev in _kb_devs:
    _t = threading.Thread(target=_evdev_keyboard_thread, args=(_dev,), daemon=True)
    _t.start()

if _kb_devs:
    print(f"[KEYBOARD] evdev: 监听 {len(_kb_devs)} 个键盘设备")
    for _dev in _kb_devs:
        print(f"  - {_dev.path}: {_dev.name}")
elif evdev is None:
    print("[KEYBOARD] 未安装 evdev；可使用浏览器键盘控制")

# --- MuJoCo key_callback: 功能键 ---
GLFW_KEY_R = 82;  GLFW_KEY_F = 70
GLFW_KEY_Z = 90;  GLFW_KEY_T = 84
GLFW_KEY_Y = 89;  GLFW_KEY_X = 88
GLFW_KEY_SPACE = 32

# 全局状态
height_cmd = 0.25
reset_flag = False
print_action_flag = False


def key_callback(keycode):
    """MuJoCo 键盘回调 — 仅处理功能键"""
    global height_cmd, reset_flag, print_action_flag
    if keycode == GLFW_KEY_R:
        height_cmd = max(0.20, height_cmd - 0.02)
    elif keycode == GLFW_KEY_F:
        height_cmd = min(0.35, height_cmd + 0.02)
    elif keycode == GLFW_KEY_Z:
        height_cmd = 0.25
    elif keycode == GLFW_KEY_T:
        reset_flag = True
    elif keycode == GLFW_KEY_Y:
        print_action_flag = not print_action_flag
    elif keycode == GLFW_KEY_X:
        _pressed_keys.clear()
    elif keycode == GLFW_KEY_SPACE and "runtime" in globals():
        runtime.request_push()

    # 原生 viewer 快捷键和浏览器面板保持同步，否则下一仿真步会被面板值覆盖。
    if "runtime" in globals():
        if keycode in (GLFW_KEY_R, GLFW_KEY_F, GLFW_KEY_Z):
            runtime_config["command"]["height"] = height_cmd
            runtime.sync_height_to_panel()
        elif keycode == GLFW_KEY_X:
            runtime.sync_stop_to_panel()


def get_commands():
    """根据当前按键状态返回指令 — 即按即动, 松手即停"""
    vx = ( 1.0 if 'w' in _pressed_keys else
          -1.0 if 's' in _pressed_keys else 0.0)
    vy = ( 1.0 if 'a' in _pressed_keys else
          -1.0 if 'd' in _pressed_keys else 0.0)
    wz = ( 1.0 if 'q' in _pressed_keys else
          -1.0 if 'e' in _pressed_keys else 0.0)
    return np.array([vx, vy, wz], dtype=np.float32)


def build_single_obs(quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                     last_action_isaac, default_angles_isaac,
                     cmd, cmd_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
                     clip_obs, height_cmd=0.25):
    """
    构建46维单步观测 — 所有关节量使用 IsaacGym dof 顺序: FL, FR, RL, RR
    第46维: 高度指令 (height_cmd - 0.25) / 0.1
    """
    obs = np.zeros(46, dtype=np.float32)
    obs[0:3] = cmd * cmd_scale[:3]

    omega_body = omega  # MuJoCo free-joint angular qvel is already in the local body frame
    obs[3:6] = omega_body.astype(np.float32) * ang_vel_scale

    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[6:9] = proj_gravity.astype(np.float32)

    obs[9:21] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)
    obs[21:33] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)
    obs[33:45] = last_action_isaac
    obs[45] = np.float32((height_cmd - 0.25) / 0.1)
    obs = np.clip(obs, -clip_obs, clip_obs)
    return obs


def reset_robot(model, data, default_angles_mujoco, position=None, quaternion=None):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = position if position is not None else [0.0, 0.0, 0.42]
    data.qpos[3:7] = quaternion if quaternion is not None else [1, 0, 0, 0]
    data.qpos[7:19] = default_angles_mujoco
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def build_runtime_config(args, kps, kds):
    """把 Dog 的扁平策略配置适配成 runtime_control 使用的结构。"""
    map_spawns = {
        name: {
            "position": [0.0, 0.0, 0.42],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        }
        for name in MAP_SPECS
    }
    map_spawns["rc26_track"] = {
        "position": [3.7, -9.0, 0.45],
        "quaternion": [-0.7071068, 0.0, 0.0, 0.7071068],
    }
    map_spawns["google_barkour"] = {
        "position": [1.0, -1.5, 0.42],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    map_labels = {
        "rc26_track": "26RC赛道",
        "race_track": "竞速赛道",
        "stairs": "楼梯赛道",
        "cross_stairs": "交叉楼梯赛道",
        "cross_slope": "交叉斜坡赛道",
        "google_barkour": "Google Barkour赛道",
        "gap_jump": "沟槽跳跃赛道",
        "hurdles": "跨栏赛道",
        "suspended_steps": "悬空踏板赛道",
        "perlin_rough": "Perlin崎岖地形",
        "dynamic_obstacles": "动态随机障碍赛道",
    }
    return make_runtime_config(
        gui=args.gui,
        title="Dog MuJoCo 实时调参",
        maps=map_labels,
        map_spawns=map_spawns,
        kp=kps[0],
        kd=kds[0],
        torque_limit=TAU_LIMIT_CALF,
        initial_position=map_spawns["rc26_track"]["position"],
        initial_quaternion=map_spawns["rc26_track"]["quaternion"],
        command=(1.0, 1.0, 1.0, 0.25),
        height_range=(0.2, 0.35),
        cameras=CAMERA_OPTIONS,
        port=args.gui_port,
        tracking_camera={
            "camera_distance": 2.0,
            "camera_azimuth": 135.0,
            "camera_elevation": -25.0,
        },
        randomization={
            "kp": [32.0, 48.0],
            "kd": [0.8, 1.2],
            "torque_limit": [28.0, 42.0],
            "motor_strength": [0.9, 1.1],
            "motor_delay_ms": [0.0, 15.0],
            "mass_scale": [0.9, 1.1],
            "payload_mass": [0.0, 2.0],
            "friction_scale": [0.5, 1.5],
            "gravity_z": [-10.3, -9.3],
        },
        push={
            "force_range": [40.0, 100.0],
            "duration_range": [0.10, 0.25],
        },
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
        help="ONNX 路径（默认使用 eg/model_3400.onnx）",
    )
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开浏览器或原生 viewer，用于自动验证",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="覆盖 YAML 中的仿真时长（秒）",
    )
    parser.add_argument(
        "--verbose-inference",
        action="store_true",
        help="打印前 10 次完整观测，默认关闭",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="只启用浏览器实时调参面板（不打开原生 MuJoCo viewer）",
    )
    parser.add_argument("--gui-port", type=int, default=8765, help="浏览器面板端口")
    args = parser.parse_args()
    if args.gui and args.headless:
        parser.error("--gui 和 --headless 不能同时使用")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须大于 0")

    config_path = DEFAULT_CONFIG
    policy_path = (
        None if args.no_policy else str(args.onnx.expanduser().resolve())
    )
    if policy_path is not None and not Path(policy_path).is_file():
        raise FileNotFoundError(f"找不到 ONNX 模型: {policy_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simulation_duration = (
        args.duration
        if args.duration is not None else config["simulation_duration"]
    )
    simulation_dt       = config["simulation_dt"]
    control_decimation  = config["control_decimation"]
    kps            = np.array(config["kps"], dtype=np.float64)
    kds            = np.array(config["kds"], dtype=np.float64)
    action_scale   = config["action_scale"]
    ang_vel_scale  = config["ang_vel_scale"]
    dof_pos_scale  = config["dof_pos_scale"]
    dof_vel_scale  = config["dof_vel_scale"]
    cmd_scale      = np.array(config["cmd_scale"], dtype=np.float32)
    clip_obs       = config.get("clip_obs", 100.0)
    runtime_config = build_runtime_config(args, kps, kds)

    # 力矩限制: hip/thigh=23.7, calf=35.55 (MuJoCo 顺序)
    tau_limits_mujoco = np.array([
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
    ])

    NUM_ONE_STEP_OBS = 46
    HISTORY_LEN      = 6
    NUM_ACTIONS      = 12

    print(f"\n{'='*60}")
    print(f"[CONFIG] Dog — 带 RL/RR 映射")
    print(f"{'='*60}")
    print(f"  IsaacGym dof: FL, FR, RL, RR")
    print(f"  MuJoCo dof:   FL, FR, RR, RL")
    print(f"  映射: MUJOCO_TO_ISAAC = {MUJOCO_TO_ISAAC}")
    print(f"  default_isaac:  {DEFAULT_ANGLES_ISAAC}")
    print(f"  default_mujoco: {DEFAULT_ANGLES_MUJOCO}")
    print(f"  kps={kps[0]}, kds={kds[0]}, action_scale={action_scale}")
    print(f"  tau_limits: hip/thigh={TAU_LIMIT_HIP_THIGH}, calf={TAU_LIMIT_CALF}")
    print(f"{'='*60}")

    # 把 Dog 机器人和 runtime_control 中的所有地图编译进同一个临时 MJCF，
    # 后续切图只切换 geom，无需重新加载模型。
    scene_temp_dir = tempfile.TemporaryDirectory(prefix="dog_mujoco_runtime_")
    combined_xml = Path(scene_temp_dir.name) / "dog_all_maps.xml"
    compose_scene(
        robot_xml=DEFAULT_ROBOT_XML,
        map_specs=MAP_SPECS,
        output_path=combined_xml,
        robot_body_name="trunk",
        robot_cameras=ROBOT_CAMERAS,
    )

    mj_model = mujoco.MjModel.from_xml_path(str(combined_xml))
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)
    runtime = RuntimeControl(
        runtime_config,
        map_names=MAP_SPECS,
        base_body_name="trunk",
        dynamic_obstacle_map="dynamic_obstacles",
    )
    motor_delay = MotorCommandDelay(simulation_dt)

    print(f"\n[验证] MuJoCo joint 顺序:")
    for i in range(mj_model.njnt):
        jn = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jn: print(f"  [{i}] {jn}")

    reset_robot(
        mj_model,
        mj_data,
        DEFAULT_ANGLES_MUJOCO,
        runtime_config["simulation"]["initial_position"],
        runtime_config["simulation"]["initial_quaternion"],
    )

    # 验证: MuJoCo qpos 转换到 Isaac 顺序应该等于 DEFAULT_ANGLES_ISAAC
    q_isaac = mj_data.qpos[7:19][MUJOCO_TO_ISAAC]
    print(f"\n[验证] qpos(MuJoCo→Isaac): {q_isaac}")
    print(f"[验证] 期望(Isaac):         {DEFAULT_ANGLES_ISAAC}")
    print(f"[验证] {'✓ 一致' if np.allclose(q_isaac, DEFAULT_ANGLES_ISAAC) else '✗ 不一致!'}")

    if not args.no_policy:
        policy = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
        input_name  = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        print(f"[ONNX] {policy.get_inputs()[0].shape} → {policy.get_outputs()[0].shape}")

    obs_history      = ObsHistoryBuffer(HISTORY_LEN, NUM_ONE_STEP_OBS)
    target_q_mujoco  = DEFAULT_ANGLES_MUJOCO.copy()
    action_isaac     = np.zeros(NUM_ACTIONS, dtype=np.float64)
    last_action_isaac = np.zeros(NUM_ACTIONS, dtype=np.float32)
    labels = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
              "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
    count = 0
    inference_count = 0

    # 预热
    for _ in range(HISTORY_LEN):
        quat_wxyz = mj_data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        omega = mj_data.qvel[3:6].astype(np.float64)
        joint_q_isaac = mj_data.qpos[7:19].astype(np.float64)[MUJOCO_TO_ISAAC]
        joint_dq_isaac = mj_data.qvel[6:18].astype(np.float64)[MUJOCO_TO_ISAAC]
        obs = build_single_obs(
            quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
            last_action_isaac, DEFAULT_ANGLES_ISAAC,
            np.zeros(3, dtype=np.float32), cmd_scale,
            ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs,
            height_cmd=0.25
        )
        obs_history.push(obs)

    print(f"[INFO] 预热完成, norm={np.linalg.norm(obs_history.buffer):.4f}")
    print(f"\n  W/S:前后 A/D:左右 Q/E:转 X:紧急停止 T:重置 Y:打印模型输出")
    print(f"  R:蹲下↓ F:站起↑ Z:重置高度(默认 0.25m)")
    print(f"  [按住移动，松手即停] evdev 全局监听, Wayland/X11 通用\n")

    display = viewer_context(
        args.gui or args.headless,
        mj_model,
        mj_data,
        key_callback=key_callback,
    )
    with display as viewer:
        # 浏览器和原生 viewer 两种显示模式互斥，避免重复渲染拖慢仿真。
        if not args.gui and not args.headless:
            setup_tracking_camera(
                viewer,
                mj_model,
                "trunk",
                distance=2.0,
                azimuth=135.0,
                elevation=-25.0,
            )

        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # 浏览器和物理键盘共同更新运动指令。未启用 --gui 时保留原有键盘逻辑。
            if runtime.update_command(_pressed_keys):
                cmd = np.array([
                    runtime_config["command"]["linear_x"],
                    runtime_config["command"]["linear_y"],
                    runtime_config["command"]["yaw"],
                ], dtype=np.float32)
                height_cmd = runtime_config["command"]["height"]
            else:
                cmd = get_commands()

            runtime_state = runtime.runtime_control(mj_model, mj_data)
            if runtime.consume_reset():
                reset_flag = True

            if reset_flag:
                reset_robot(
                    mj_model,
                    mj_data,
                    DEFAULT_ANGLES_MUJOCO,
                    runtime_config["simulation"]["initial_position"],
                    runtime_config["simulation"]["initial_quaternion"],
                )
                obs_history.reset()
                last_action_isaac[:] = 0; action_isaac[:] = 0; count = 0; inference_count = 0
                print_action_flag = False
                height_cmd = runtime_config["command"]["height"]
                motor_delay.reset()
                runtime.reset_simulation_state()
                reset_flag = False

            # 读取 MuJoCo 状态
            joint_q_mujoco  = mj_data.qpos[7:19].astype(np.float64)
            joint_dq_mujoco = mj_data.qvel[6:18].astype(np.float64)

            # ★ 映射到 IsaacGym 顺序 (送给策略)
            joint_q_isaac  = joint_q_mujoco[MUJOCO_TO_ISAAC]
            joint_dq_isaac = joint_dq_mujoco[MUJOCO_TO_ISAAC]

            quat_wxyz = mj_data.qpos[3:7]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            omega = mj_data.qvel[3:6].astype(np.float64)

            if count % control_decimation == 0:
                if not args.no_policy:
                    single_obs = build_single_obs(
                        quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                        last_action_isaac, DEFAULT_ANGLES_ISAAC,
                        cmd, cmd_scale,
                        ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs,
                        height_cmd=height_cmd
                    )
                    obs_history.push(single_obs)
                    obs_input = obs_history.get()

                    if args.verbose_inference and inference_count < 10:
                        print(f"\n{'='*60}")
                        print(f"[推理 #{inference_count}] 完整 276 维 history obs (6帧×46维):")
                        buf = obs_history.buffer
                        for slot in range(HISTORY_LEN):
                            base = slot * 46
                            frame = buf[base:base+46]
                            print(f"\n  --- 帧{slot} (offset {base}) ---")
                            print(f"    cmd:          {frame[0:3]}")
                            print(f"    gyro_body:    {frame[3:6]}")
                            print(f"    proj_gravity: {frame[6:9]}")
                            print(f"    dof_pos_dev:  {np.array2string(frame[9:21], precision=4, separator=', ')}")
                            print(f"    dof_vel:      {np.array2string(frame[21:33], precision=4, separator=', ')}")
                            print(f"    last_action:  {np.array2string(frame[33:45], precision=4, separator=', ')}")
                            print(f"    height_cmd:   {frame[45]:.4f}")
                            print(f"    frame norm:   {np.linalg.norm(frame):.4f}")
                        if inference_count == 9:
                            print(f"\n[INFO] 前10帧 obs 已输出完毕，按 Y 开始打印模型输出。")
                        print(f"{'='*60}")

                    action_raw = policy.run([output_name], {input_name: obs_input})[0][0]
                    action_isaac[:] = np.clip(action_raw, -10.0, 10.0)
                    last_action_isaac = action_isaac.astype(np.float32)

                    if inference_count >= 10 and print_action_flag:
                        print(f"\n[推理 #{inference_count}] 模型输出 action x {OUTPUT_PRINT_SCALE} (Isaac顺序 FL,FR,RL,RR):")
                        for j in range(12):
                            print(f"    {labels[j]:12s}: action={action_isaac[j] * OUTPUT_PRINT_SCALE:+.4f}")

                    inference_count += 1

                    # ★ 目标角: IsaacGym顺序 → MuJoCo顺序
                    target_q_isaac = action_isaac * action_scale + DEFAULT_ANGLES_ISAAC
                    target_q_mujoco = target_q_isaac[ISAAC_TO_MUJOCO]
                else:
                    target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()

            # PD 控制 (MuJoCo 顺序)。实时 Kp/Kd、电机强度、延迟和力矩上限
            # 都在真正写入 data.ctrl 前生效。
            applied_target_q = motor_delay.apply(
                target_q_mujoco, runtime_state["motor_delay_ms"]
            )
            effective_tau_limits = scale_torque_limits(
                tau_limits_mujoco,
                runtime_state["torque_limit"],
                reference_limit=TAU_LIMIT_CALF,
            )
            tau = compute_pd_torques(
                applied_target_q,
                joint_q_mujoco,
                joint_dq_mujoco,
                runtime_state["kp"],
                runtime_state["kd"],
                motor_strength=runtime_state["motor_strength"],
                torque_limit=effective_tau_limits,
            )
            mj_data.ctrl[:NUM_ACTIONS] = tau

            runtime.apply_external_forces(mj_model, mj_data)
            mujoco.mj_step(mj_model, mj_data)
            count += 1

            if count % (control_decimation * 50) == 0:
                grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                h = mj_data.qpos[2]
                print(f"[{time.time()-start:.1f}s] Step {count} H={h:.3f}(cmd={height_cmd:.2f}) "
                      f"vx={mj_data.qvel[0]:.2f} vy={mj_data.qvel[1]:.2f} wz={mj_data.qvel[5]:.2f} "
                      f"grav_z={grav[2]:.3f} "
                      f"act=[{action_isaac.min():.2f},{action_isaac.max():.2f}]")

            if not args.gui and not args.headless:
                viewer.sync()
            elapsed = time.time() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)

    runtime.close()
    scene_temp_dir.cleanup()
    print("\n[INFO] 仿真结束")
