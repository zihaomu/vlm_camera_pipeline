# VLM Camera Pipeline

AMD Ryzen AI MAX+ 395 / Radeon 8060S（`gfx1151`）本地低延迟摄像头 Demo。

当前已实现 M0-M2 的 PyTorch ROCm 主路径：

```text
V4L2 camera -> depth-1 latest frame -> YOLO26x ROCm -> latest result -> OpenCV UI
```

本机 M2 长稳 Gate 已通过：持续 1800.130 秒，capture/inference/display 为
29.820/29.820/29.812 FPS，capture-to-display P95 为 14.951 ms，摄像头读取失败 0、GPU
reset/fault/hang 0。完整证据见 `output/realtime/metrics-camera-30m.json` 和
`output/realtime/m2-gate.json`。

Python 环境只由 `uv` 管理，`.venv`、依赖 cache 和应用配置均位于本仓库内。MIGraphX、
VA-API 录制和 Qwen3-VL 属于后续独立里程碑，目前不会静默启用或回退 CPU。

## 初始化

前提：本机已有受支持的 ROCm 7.2.1、Python 3.12、uv、Git、curl 和可访问的摄像头。

```bash
bash scripts/setup_native_gfx1151.sh
```

该脚本下载并校验锁定的 AMD wheel、上游源码和 `yolo26x.pt`，再运行环境检查与测试。它不
安装系统包，也不向系统 Python 或用户 site-packages 写入内容。

## 运行实时 Demo

```bash
uv run --frozen python scripts/run_camera.py \
  --device /dev/video0 \
  --fourcc NV12 \
  --width 1280 --height 720 --camera-fps 30 \
  --model models/yolo26x.pt \
  --backend pytorch \
  --display --record off --vlm off
```

按 `q`、Esc 或 `Ctrl+C` 可退出。无窗口 smoke 示例：

```bash
uv run --frozen python scripts/run_camera.py \
  --no-display --duration-seconds 30 \
  --metrics-json output/realtime/metrics-smoke.json
```

## 验证

```bash
uv run --frozen ruff check src scripts tests
uv run --frozen python -m pytest -q
uv run --frozen python scripts/check_gfx1151_environment.py \
  --require-torch --output native-lock/environment.json
```

完整设计、实时状态和实测数据见 [doc/amd_395_pipeline.md](doc/amd_395_pipeline.md)。
