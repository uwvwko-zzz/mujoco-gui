# MuJoCo Runtime Control：快速移植与 AI 适配指南

`runtime_control` 是一套与策略、机器人型号和关节顺序解耦的 MuJoCo 运行期组件。当前完整适配参考：`mujoco/dog/play_onnx_46.py`。

它提供浏览器画面、键盘命令、地图切换、机载相机、实时 PD 参数、电机延迟、质量/摩擦/重力调整、随机推力、随机化、重置和急停。

## 1. 文件与公共接口

```text
runtime_control/
├── __init__.py                 # 稳定的公共导入入口
├── integration.py              # 移植胶水：相机、viewer、配置工厂
├── runtime.py                  # 运行时状态、渲染、推力和模型参数
├── panel.py                    # 浏览器 UI、HTTP/MJPEG 和防卡键
├── map_manager.py              # MJCF 合并、地图切换和出生点
├── control.py                  # 电机延迟、PD 和分关节力矩限制
├── example_runtime_config.yaml # YAML 配置参考
└── maps/                       # terrain-only 地图与资源
```

新适配优先从包入口导入，不要引用内部文件：

```python
from runtime_control import (
    MapSpec, MotorCommandDelay, RuntimeControl, compose_scene,
    compute_pd_torques, make_runtime_config,
    make_standard_robot_cameras, scale_torque_limits,
    setup_tracking_camera, standard_camera_options, viewer_context,
)
```

## 2. 移植前必须查清的事实

不要根据机器人名字猜以下内容，应从目标脚本、MJCF、训练配置和 ONNX 输入输出实际确认：

1. 机器人 MJCF 路径和自由根机身名称；
2. MuJoCo 的 `qpos/qvel/ctrl` 地址和关节顺序；
3. 策略使用的关节顺序、默认角、观测布局和动作缩放；
4. 仿真时间步、控制降采样、Kp/Kd 和每个电机的力矩上限；
5. 根坐标重置高度、四元数顺序和每张地图的安全出生点；
6. 目标脚本是“键盘控制类”还是全局按键集合；
7. 实际运行环境是否有 `mujoco`、`Pillow`、`yaml` 和策略后端。

MuJoCo 根四元数为 `wxyz`。策略可能使用 `xyzw`，二者不能混用。

## 3. 最短移植流程

### 3.1 让入口找到公共包

入口位于 `mujoco/<robot>/play.py` 并以文件方式执行时：

```python
from pathlib import Path
import sys

MUJOCO_DIR = Path(__file__).resolve().parents[1]
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))
```

随后再 `from runtime_control import ...`。不要复制一份组件到机器人目录，否则公共修复无法同步。

### 3.2 声明地图

```python
MAP_DIR = MUJOCO_DIR / "runtime_control/maps"
MAP_SPECS = {
    "flat": MapSpec(MAP_DIR / "race_track.xml"),
    "stairs": MapSpec(MAP_DIR / "stairs.xml"),
}
```

地图 XML 自带另一个机器人时必须排除其根 body：

```python
"course": MapSpec(
    MAP_DIR / "26rc_track.xml",
    exclude_bodies=("old_robot_root",),
)
```

地图最好是 terrain-only MJCF，只包含静态地形。相对 mesh、texture 和 hfield 路径会在合并时转成绝对路径。

### 3.3 生成真正的机载相机

世界坐标 `track/trackcom` 相机通常和第三人称差别不大。公共工厂会生成三台真正挂在机器人根 body 下的 fixed 相机：

```python
ROBOT_CAMERAS = make_standard_robot_cameras(prefix="my_robot")
CAMERA_OPTIONS = standard_camera_options(prefix="my_robot")
```

默认假设本体坐标为 `+X` 前、`+Y` 左、`+Z` 上，尺寸接近中型四足。尺寸不同时调整局部位置：

```python
ROBOT_CAMERAS = make_standard_robot_cameras(
    prefix="my_robot",
    front_position=(0.20, 0.0, 0.12),
    rear_position=(-0.80, 0.0, 0.50),
    top_position=(0.0, 0.0, 1.00),
)
```

UI 键名必须和 MJCF camera 的 `name` 完全一致。两个函数使用相同 `prefix` 可避免错配。

### 3.4 合并机器人和地图

```python
import tempfile

scene_temp = tempfile.TemporaryDirectory(prefix="my_robot_runtime_")
combined_xml = Path(scene_temp.name) / "scene.xml"
compose_scene(
    robot_xml=ROBOT_XML,
    map_specs=MAP_SPECS,
    output_path=combined_xml,
    robot_body_name="base_link",  # 必须是真实根 body 名称
    robot_cameras=ROBOT_CAMERAS,
)
model = mujoco.MjModel.from_xml_path(str(combined_xml))
data = mujoco.MjData(model)
```

所有地图编译进一个模型，运行时通过 geom 分组切换，不会重新创建 `MjModel`。未启用地图会关闭碰撞并隐藏。

### 3.5 构造运行时配置

策略 YAML 可以保持原结构。单独构造运行期配置，避免为了 UI 改坏训练/推理参数：

```python
map_labels = {"flat": "平地", "stairs": "楼梯"}
map_spawns = {
    "flat": {"position": [0, 0, 0.45], "quaternion": [1, 0, 0, 0]},
    "stairs": {"position": [0, 0, 0.45], "quaternion": [1, 0, 0, 0]},
}

runtime_config = make_runtime_config(
    gui=args.gui,
    title="My Robot MuJoCo 实时调参",
    maps=map_labels,
    map_spawns=map_spawns,
    kp=40.0,
    kd=1.0,
    torque_limit=35.0,
    initial_position=map_spawns["flat"]["position"],
    command=(1.0, 1.0, 1.0, 0.30),  # vx, vy, yaw, height
    height_range=(0.20, 0.40),
    cameras=CAMERA_OPTIONS,
    port=args.gui_port,
    randomization={
        "kp": [32.0, 48.0], "kd": [0.8, 1.2],
        "torque_limit": [28.0, 42.0],
        "motor_strength": [0.9, 1.1],
        "motor_delay_ms": [0.0, 15.0],
        "mass_scale": [0.9, 1.1], "payload_mass": [0.0, 2.0],
        "friction_scale": [0.5, 1.5], "gravity_z": [-10.3, -9.3],
    },
)
runtime = RuntimeControl(
    runtime_config, map_names=MAP_SPECS,
    base_body_name="base_link", dynamic_obstacle_map=None,
)
delay = MotorCommandDelay(model.opt.timestep)
```

`make_runtime_config` 会检查每个 UI 地图是否都有出生点，提前暴露键名遗漏。

### 3.6 浏览器和原生 viewer 必须互斥

不要在 `--gui` 时仍无条件调用 `mujoco.viewer.launch_passive()`，否则同一仿真会渲染两次，帧率明显下降。

```python
display = viewer_context(args.gui, model, data, key_callback=key_callback)
with display as viewer:
    if not args.gui:
        setup_tracking_camera(
            viewer, model, "base_link",
            distance=1.7, azimuth=135, elevation=-20,
        )

    while viewer.is_running():
        # simulation loop
        if not args.gui:
            viewer.sync()
```

`--gui` 时仅提供无窗口循环上下文，浏览器渲染由 `RuntimeControl` 管理；不加 `--gui` 时才创建原生 viewer。

### 3.7 每步接入运行时钩子

全局物理按键集合版本：

```python
if runtime.update_command(physical_pressed_keys):
    cmd = np.array([
        runtime_config["command"]["linear_x"],
        runtime_config["command"]["linear_y"],
        runtime_config["command"]["yaw"],
    ], dtype=np.float32)
else:
    cmd = original_get_commands()

state = runtime.runtime_control(model, data)
if runtime.consume_reset():
    reset_robot_from_runtime_config()
    delay.reset()
    runtime.reset_simulation_state()

runtime.apply_external_forces(model, data)
```

已有键盘类时可用 `RuntimeKeyboardMixin`。不要为了 Mixin 强行把简单的全局按键脚本重构成类。

### 3.8 在写入 `data.ctrl` 前应用电机参数

```python
applied_targets = delay.apply(targets, state["motor_delay_ms"])
data.ctrl[:] = compute_pd_torques(
    applied_targets, joint_positions, joint_velocities,
    state["kp"], state["kd"],
    motor_strength=state["motor_strength"],
    torque_limit=state["torque_limit"],
)
```

Kp、Kd、强度和限制可以是标量或逐关节数组。

若 hip/thigh/calf 的标称限制不同，而 UI 只有一个滑块，应保持原比例：

```python
effective_limits = scale_torque_limits(
    nominal_limits, state["torque_limit"], reference_limit=35.55,
)
```

再把 `effective_limits` 传给 `compute_pd_torques`。不要用一个标量覆盖全部关节限制。

## 4. 重置的完整语义

地图切换会更新：

```python
runtime_config["simulation"]["initial_position"]
runtime_config["simulation"]["initial_quaternion"]
```

reset 必须读取它们，不能继续硬编码位置。完整重置还应：

- `mj_resetData()` 后恢复根位置、根四元数和默认关节角；
- 清空策略历史观测和 last action；
- `delay.reset()`，防止执行重置前的旧目标；
- `runtime.reset_simulation_state()`，清除残余外力；
- `mj_forward()`。

## 5. 键盘与视角

浏览器面板已经处理窗口失焦、页面隐藏、地图/视角下拉框、急停和关闭页面造成的卡键。前端发送释放事件，后端在地图/相机切换时也清空按键兜底。适配脚本不应另外缓存浏览器按键。

物理功能键修改高度或急停时同步面板：

```python
runtime_config["command"]["height"] = new_height
runtime.sync_height_to_panel()
runtime.sync_stop_to_panel()
```

## 6. 性能原则

- 浏览器 `fps` 是画面目标，不是物理仿真频率；
- `width/height` 增大会提高 OpenGL、回读和 JPEG 编码成本；
- 优先用 CSS 放大，仅在明显模糊时提高渲染分辨率；
- `--gui` 时不要再启动原生 viewer；
- 多地图模型包含更多 geom 元数据，这是即时切图的代价；
- Perlin 地形优先使用 hfield，不要用数千个小 box；
- `runtime_control()` 每个物理步调用，策略推理仍按原 decimation 执行。

## 7. 常见失败

### `ModuleNotFoundError: runtime_control`

按 3.1 把 `mujoco/` 加入 `sys.path`。不要把仓库的 `mujoco/` 命名空间误认为 PyPI 的 `mujoco` 包；验证应使用项目真实 conda 环境。

### 找不到根 body

`compose_scene(robot_body_name=...)` 和 `RuntimeControl(base_body_name=...)` 必须使用同一个真实名称。

### 切图后机器人消失或摔落

检查该地图的出生高度和 `wxyz` 四元数，不要只调整全局初始位置。

### 视角少或“固定相机”无效果

世界坐标 `trackcom` 不等于机载相机。使用 `make_standard_robot_cameras()` 注入根 body，并把 `standard_camera_options()` 传给配置。

### 电机延迟无效果

确认 `delay.apply()` 位于每个物理步，不是只位于策略推理步；重置时调用 `delay.reset()`。

### 调力矩后动作被破坏

不同电机限制使用 `scale_torque_limits()` 保持比例，并核对策略/MuJoCo 关节映射方向。

### 端口占用

暴露 `--gui-port` 并传给 `make_runtime_config(port=args.gui_port)`。

## 8. 验收清单

- [ ] `model.nq/nv/nu` 与原机器人一致；
- [ ] 根 body 存在，标准相机 `cam_bodyid` 等于根 body id；
- [ ] 每张地图至少有一个 `map_<name>_...` geom；
- [ ] 每张地图能切换并在安全出生点重置；
- [ ] 不加 `--gui` 只开原生 viewer；
- [ ] 加 `--gui` 只开浏览器；
- [ ] 按住移动键切视角后不会卡键；
- [ ] 策略输入、动作和关节映射未改变；
- [ ] 标称设置下新旧 PD 输出一致；
- [ ] Kp/Kd、强度、延迟和力矩确实影响 `data.ctrl`；
- [ ] reset 清空观测历史、延迟队列和外力；
- [ ] 退出调用 `runtime.close()` 和临时目录 `cleanup()`。

先执行静态检查，再用实际环境编译组合 MJCF：

```bash
python -m py_compile mujoco/runtime_control/*.py mujoco/<robot>/play.py
```

## 9. 可直接交给 AI 的任务模板

```text
请把 mujoco/runtime_control 接入 <目标入口脚本>。

要求：
1. 先读 runtime_control/README_zh.md、目标入口、策略配置和机器人 MJCF；
2. 实际确认根 body、qpos/qvel/ctrl 地址、两侧关节顺序和力矩限制，不要猜；
3. 保留现有观测、动作缩放、默认角和关节映射；
4. 使用 MapSpec + compose_scene，不复制 runtime_control；
5. 使用 make_standard_robot_cameras + standard_camera_options；
6. 使用 make_runtime_config 构造独立运行时配置；
7. 使用 viewer_context，保证 --gui 只开浏览器，非 --gui 只开原生 viewer；
8. 每个物理步接入 RuntimeControl、MotorCommandDelay、外力和真实 PD 参数；
9. 分关节力矩不同时用 scale_torque_limits 保持比例；
10. reset 读取当前出生点并清空观测历史、延迟队列和外力；
11. 不覆盖目标脚本中与任务无关的已有修改；
12. 最后执行 py_compile、组合 MJCF 编译、根 body/相机归属、地图分组和
    新旧 PD 数值回归验证，并报告启动命令。
```

## 10. 最小复制范围

跨仓库使用时复制完整 `mujoco/runtime_control/`。不需要全部地图时，删除不用的 XML/资源，并同步删除适配脚本中对应的 `MAP_SPECS`、UI 标签和出生点。
