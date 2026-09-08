# MuJoCo Runtime Control：快速移植与 AI 适配指南

`runtime_control` 是一套与策略、机器人型号和关节顺序解耦的 MuJoCo 运行期组件。轮足混合控制适配可参考 `mujoco/w1w/play_gui.py`。

它提供浏览器画面、键盘命令、地图切换、机载相机、实时 PD 参数、电机延迟、质量/摩擦/重力调整、随机推力、随机化、重置和急停。

## 1. 文件与公共接口

```text
runtime_control/
├── pyproject.toml / setup.cfg  # 可安装包元数据
├── MANIFEST.in                 # wheel/sdist 资源声明
├── src/runtime_control/
│   ├── __init__.py             # 稳定的公共导入入口
│   ├── session.py              # 组合场景的生命周期封装
│   ├── adapter.py              # 机器人/策略接入契约
│   ├── position_pd_adapter.py  # 纯位置 PD 机器人通用实现
│   ├── resources.py            # 包内地形资源 API
│   ├── integration.py          # 相机、viewer、配置工厂
│   ├── runtime.py / panel.py   # 运行时与浏览器 UI
│   ├── map_manager.py          # MJCF 合并、切图和出生点
│   ├── scene_builder.py        # 可复现地形、参数化障碍和场景 JSON
│   ├── control.py              # 电机延迟、PD 与力矩限制
│   └── maps/                   # 随 wheel 发布的 terrain-only 资源
└── eg/ / tests/                 # 可运行示例与回归测试
```

推荐在目标 Python 环境中以可编辑方式安装：

```bash
python -m pip install -e mujoco/mujoco-gui
```

新适配优先从包入口导入，不要引用内部文件：

```python
from runtime_control import (
    ActionSpec, MapSpec, MotorCommandDelay, ParameterSpec,
    RuntimeControl, RuntimeScene, RobotAdapter, PositionPDAdapter,
    bundled_map_specs,
    compute_pd_torques, make_runtime_config,
    make_standard_robot_cameras, scale_torque_limits,
    setup_tracking_camera, standard_camera_options, viewer_context,
)```

目录本身可以叫 `mujoco-gui`，安装后稳定的 Python 包名仍是
`runtime_control`。

依赖方向是单向的：用户项目导入 `runtime_control`，包内不反向导入
W1W、Dog、ONNX 或任何训练框架。观测构造、关节映射、策略推理和执行器
始终留在用户项目中。

### 1.1 生成可回放场景

`scene_builder` 将场景源数据和运行时地图分开：`scene.json` 用于编辑、
版本管理和失败回放，`terrain.xml` 仍是可由 `RuntimeScene` 安全合并的
terrain-only MJCF。

```python
from runtime_control import ObstacleSpec, SceneSpec, TerrainSpec, scene_map_spec

scene = SceneSpec(
    name="rough_course",
    terrain=TerrainSpec(kind="noise", seed=42, length=12, width=8),
    obstacles=[
        ObstacleSpec("stairs", x=2.0, params={"count": 6, "rise": 0.12}),
        ObstacleSpec("gap", x=5.0, params={"gap": 0.35}),
        ObstacleSpec("stepping_stones", x=7.0, params={"count": 8}),
    ],
)
generated = scene_map_spec(scene, "generated/rough_course")
scene_runtime = RuntimeScene.for_adapter(
    adapter, ROBOT_XML, {"rough": generated}, runtime_config
)
```

基础地形支持 `flat/slope/stairs/noise`；障碍支持 `platform/wall/stairs/gap/`
`stepping_stones/slalom/ramp/side_slope/speed_bumps/hurdles/narrow_bridge/`
`wave_ground/uneven_stairs/random_blocks/seesaw/rotating_bar`。随机地形由
`seed` 复现，导出的非平地使用单个
MuJoCo `hfield` geom。

`seesaw` 和 `rotating_bar` 使用 MuJoCo mocap body 生成动态碰撞几何；它们由运行时
根据仿真时间驱动，不会增加机器人的 `nq/nv/nu`。

### 1.2 可视化地图编辑器

安装本包后启动本地浏览器编辑器：

```bash
mujoco-scene-editor --output generated/visual_course
```

本仓库中可直接使用一键脚本，它会设置源码路径并使用已安装 MuJoCo/ONNX
的 `gym` Python 环境：

```bash
./mujoco/mujoco-gui/start_editor.sh
```

可在后面追加参数覆盖默认值，例如 `--port 9000 --preview-port 9001`。需要
使用其他 Python 时，设置 `HIMLOCO_EDITOR_PYTHON=/path/to/python`。

页面左侧选择障碍类型，在中间俯视场地单击添加，拖动已有对象可修改位置；
右侧可修改地形、坐标、旋转和障碍参数。点击“生成 MuJoCo 场景”会向
`--output` 目录写入 `scene.json` 和 `terrain.xml`，非平地额外写入
`terrain.png`。如果不想自动打开浏览器，使用 `--no-browser`。

编辑器左侧“已有地形”可载入之前保存的 `scene.json`，载入后可继续拖动、
修改和覆盖保存。“删除当前已保存地形”只删除 `--output` 范围内当前选中
场景，并会先请求确认。“用演示机器人在 MuJoCo 展示”会先保存当前场景，
再以仓库内自包含的 `eg/dog` XML 和 ONNX 策略在新标签页启动仿真。
可通过 `--preview-player`、`--robot-xml`、`--policy`、`--python` 和
`--preview-port` 接入其他机器人。W1W 项目专用集成请使用：

```bash
./mujoco/w1w/start_editor.sh
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

### 2.1 12 自由度纯位置 PD

如果 action 全部是关节目标位置，直接配置 `PositionPDAdapter`：

```python
def build_obs(adapter, model, data, command, **context):
    joint_pos = data.qpos[adapter.qpos_adr]
    joint_vel = data.qvel[adapter.qvel_adr]
    # 按训练时的布局、缩放和历史顺序构造，必须与策略一致。
    return make_policy_observation(joint_pos, joint_vel, command)

adapter = PositionPDAdapter(
    root_body_name="trunk",
    expected_dimensions=(19, 18, 12),
    joint_names=JOINT_NAMES,
    actuator_names=ACTUATOR_NAMES,
    default_positions=DEFAULT_POS,
    kp=KP,
    kd=KD,
    action_scale=0.25,
    torque_limits=TORQUE_LIMITS,
    observation_builder=build_obs,
)
```

### 2.2 16 自由度混合控制

如果同时包含位置和速度 action，继承 `RobotAdapter` 实现
`bind/reset/build_observation/compute_control`。W1W 的 12 条腿位置 PD +
4 轮速度控制完整实现在 `mujoco/w1w/w1w_adapter.py`。

两种适配器都会验证根 body、`nq/nv/nu`、actuator ID、输出维度以及
NaN/Inf，不再让关节顺序错误静默进入仿真。

## 3. 最短移植流程

### 3.1 安装并导入公共包

开发时使用可编辑安装：

```bash
python -m pip install -e /path/to/mujoco-gui
```

部署到其他项目可直接安装构建出的 wheel，不需要复制源码目录。
W1W 中的源码路径回退只用于本仓库上层调试，不是对外 API。

### 3.2 声明地图

```python
from runtime_control import bundled_map_specs

MAP_SPECS = bundled_map_specs(("race_track", "stairs"))
```

地图 XML 自带另一个机器人时必须排除其根 body：

```python
from runtime_control import bundled_map_spec

MAP_SPECS["course"] = bundled_map_spec(
    "rc26_track", exclude_bodies=("old_robot_root",)
)
```

地图最好是 terrain-only MJCF，只包含静态地形。相对 mesh、texture 和 hfield 路径会在合并时转成绝对路径。
每张地图可通过 `minimum_box_half_thickness` 单独设置薄板最小半厚度，
并用 `contact_params={"friction": "0.6 0.005 0.0001", "solref": "0.03 2"}`
设置接触参数。`strict=True` 为默认值，导入 body 如果含有
joint/freejoint 会立即报错，防止地图悄悄改变机器人自由度。

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

先按 3.5 构造 `runtime_config`，然后由会话对象统一管理临时目录、
MJCF 合并、`MjModel/MjData`、运行时和关闭清理：

```python
scene = RuntimeScene.for_adapter(
    adapter,
    robot_xml=ROBOT_XML,
    map_specs=MAP_SPECS,
    runtime_config=runtime_config,
    robot_cameras=ROBOT_CAMERAS,
    dynamic_obstacle_map="dynamic_obstacles",
)
scene.open()
model, data, runtime = scene.model, scene.data, scene.runtime
```

所有地图编译进一个模型，运行时通过 geom 分组切换，不会重新创建 `MjModel`。
`for_adapter()` 会自动使用适配器的根 body/模型维度，打开时 bind，关闭时
unbind。未启用地图会关闭碰撞并隐藏。

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
    parameters=[
        ParameterSpec(
            "wheel_velocity_kp", "轮子速度 Kp", 0, 5, 0.05, 1.5
        ),
    ],
    actions=[
        ActionSpec("fall_side", "侧翻测试", shortcut="n", style="warn"),
    ],
    random_seed=1,
    snapshot_path="runtime_preset.json",
)
delay = MotorCommandDelay(model.opt.timestep)
```

`make_runtime_config` 会检查每个 UI 地图是否都有出生点，提前暴露键名遗漏。

### 3.6 浏览器和原生 viewer 必须互斥

不要在 `--gui` 时仍无条件调用 `mujoco.viewer.launch_passive()`，否则同一仿真会渲染两次，帧率明显下降。

```python
display = scene.viewer(args.gui, key_callback=key_callback)
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
关闭浏览器面板会通知 `viewer.is_running()` 退出，并回收 HTTP
端口与渲染线程。

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
- 插件会按 `width/height` 自动扩大 MuJoCo offscreen framebuffer；
- `width/height` 增大会提高 OpenGL、回读和 JPEG 编码成本；
- 优先用 CSS 放大，仅在明显模糊时提高渲染分辨率；
- `--gui` 时不要再启动原生 viewer；
- 多地图模型包含更多 geom 元数据，这是即时切图的代价；
- Perlin 地形优先使用 hfield，不要用数千个小 box；
- `runtime_control()` 每个物理步调用，策略推理仍按原 decimation 执行。

## 7. 常见失败

### `ModuleNotFoundError: runtime_control`

按 3.1 在当前 Python/conda 环境安装本包，并用
`python -m pip show mujoco-runtime-control` 确认环境一致。

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
- [ ] 退出调用 `scene.close()`，或使用 `with RuntimeScene(...)`。
- [ ] 关闭后同一进程可以在原端口再次启动面板。

先执行静态检查，再用实际环境编译组合 MJCF：

```bash
python -m py_compile mujoco/mujoco-gui/src/runtime_control/*.py mujoco/<robot>/play.py
python -m unittest discover -s mujoco/mujoco-gui/tests -v
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

## 10. 跨项目发布

不再复制包内 Python/XML/纹理文件。在包目录执行
`python -m pip wheel . --no-build-isolation --no-deps -w dist`，
然后在用户项目安装 `dist/mujoco_runtime_control-*.whl`。地形 XML、纹理和
mesh 会随 wheel 安装，并由 `bundled_map_specs()` 定位。用户项目只保留
机器人 MJCF、策略适配代码、出生点和它自己的地形。
