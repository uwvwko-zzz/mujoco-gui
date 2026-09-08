# Runtime Control Demo

这个目录是自包含的 Dog ONNX 示例，仓库内资源均通过 `play.py` 所在位置定位，运行不依赖当前工作目录，也不包含开发机绝对路径。

包含：

- `model_3400.onnx`：示例策略；
- `dog.yaml`：策略和仿真参数；
- `dog/`：MuJoCo XML、mesh 和 URDF；
- `play.py`：Runtime Control 完整接入示例。

## 安装依赖

```bash
pip install mujoco numpy onnxruntime pyyaml pillow
```

Linux 原生全局键盘监听可选：

```bash
pip install evdev
```

没有 `evdev` 时仍可在浏览器面板中使用 `W/S/A/D/Q/E`。

## 从仓库根目录运行

浏览器控制面板：

```bash
python eg/play.py --gui
```

原生 MuJoCo viewer：

```bash
python eg/play.py
```

无窗口快速验证：

```bash
python eg/play.py --headless --duration 1
```

使用其他策略：

```bash
python eg/play.py --gui --onnx path/to/model.onnx
```

其他参数可通过以下命令查看：

```bash
python eg/play.py --help
```
