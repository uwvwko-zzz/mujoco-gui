#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${HIMLOCO_EDITOR_PYTHON:-python}"
EDITOR_HOST="127.0.0.1"
EDITOR_PORT="8765"
PREVIEW_PORT="8766"

arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    argument="${arguments[index]}"
    case "${argument}" in
        --host=*) EDITOR_HOST="${argument#*=}" ;;
        --port=*) EDITOR_PORT="${argument#*=}" ;;
        --preview-port=*) PREVIEW_PORT="${argument#*=}" ;;
        --host|--port|--preview-port)
            if ((index + 1 >= ${#arguments[@]})); then
                echo "错误：${argument} 后面缺少参数。" >&2
                exit 2
            fi
            value="${arguments[index+1]}"
            case "${argument}" in
                --host) EDITOR_HOST="${value}" ;;
                --port) EDITOR_PORT="${value}" ;;
                --preview-port) PREVIEW_PORT="${value}" ;;
            esac
            ((index+=1))
            ;;
    esac
done

PYTHON_PATH="$(command -v -- "${PYTHON_BIN}" 2>/dev/null || true)"
if [[ -z "${PYTHON_PATH}" || ! -x "${PYTHON_PATH}" ]]; then
    echo "错误：找不到可执行的 Python：${PYTHON_BIN}" >&2
    echo "可以通过 HIMLOCO_EDITOR_PYTHON=/path/to/python 指定环境。" >&2
    exit 1
fi
PYTHON_BIN="${PYTHON_PATH}"

if ! "${PYTHON_BIN}" -c "import mujoco, numpy, onnxruntime, PIL" 2>/dev/null; then
    echo "错误：${PYTHON_BIN} 缺少 mujoco、numpy、onnxruntime 或 Pillow。" >&2
    exit 1
fi

if [[ ! "${EDITOR_PORT}" =~ ^[0-9]+$ ]] || ((EDITOR_PORT < 1 || EDITOR_PORT > 65535)); then
    echo "错误：编辑器端口必须是 1 到 65535：${EDITOR_PORT}" >&2
    exit 2
fi
if [[ ! "${PREVIEW_PORT}" =~ ^[0-9]+$ ]] || ((PREVIEW_PORT < 1 || PREVIEW_PORT > 65535)); then
    echo "错误：预览端口必须是 1 到 65535：${PREVIEW_PORT}" >&2
    exit 2
fi
if [[ "${EDITOR_PORT}" == "${PREVIEW_PORT}" ]]; then
    echo "错误：编辑器端口和 MuJoCo 预览端口不能相同。" >&2
    exit 2
fi

CONNECT_HOST="${EDITOR_HOST}"
if [[ "${CONNECT_HOST}" == "0.0.0.0" || "${CONNECT_HOST}" == "::" ]]; then
    CONNECT_HOST="127.0.0.1"
fi

if "${PYTHON_BIN}" -c 'import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(0 if s.connect_ex((sys.argv[1],int(sys.argv[2]))) == 0 else 1)' "${CONNECT_HOST}" "${EDITOR_PORT}"; then
    editor_url="http://${CONNECT_HOST}:${EDITOR_PORT}/"
    if command -v curl >/dev/null 2>&1 && curl --max-time 1 --silent "${editor_url}" | grep -q "MuJoCo 可视化地图编辑器"; then
        echo "地图编辑器已经在运行：${editor_url}"
        echo "无需重复启动；请直接在浏览器中打开上面的地址。"
        exit 0
    fi
    echo "错误：${EDITOR_HOST}:${EDITOR_PORT} 已被其他程序占用。" >&2
    echo "请关闭占用程序，或改用：$0 --port 9000 --preview-port 9001" >&2
    exit 1
fi

if "${PYTHON_BIN}" -c 'import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(0 if s.connect_ex((sys.argv[1],int(sys.argv[2]))) == 0 else 1)' "${CONNECT_HOST}" "${PREVIEW_PORT}"; then
    echo "错误：MuJoCo 预览端口 ${PREVIEW_PORT} 已被占用。" >&2
    echo "请关闭已有预览，或改用：$0 --preview-port 9001" >&2
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${SCRIPT_DIR}"

exec "${PYTHON_BIN}" -m runtime_control.scene_editor \
    --output "${SCRIPT_DIR}/generated/visual_course" \
    --port 8765 \
    --preview-port 8766 \
    --python "${PYTHON_BIN}" \
    "$@"
