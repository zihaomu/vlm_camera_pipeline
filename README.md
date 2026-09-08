# VLM Camera Pipeline

AMD Ryzen AI MAX+ 395 / Radeon 8060S（`gfx1151`）本地低延迟摄像头 Demo。

当前已实现 M0-M2 的 PyTorch ROCm 主路径和 M5 的 Qwen3-VL GPU 异步字幕：

```text
V4L2 camera -> depth-1 latest frame -> YOLO26x ROCm -> latest result -> OpenCV UI
                                  \-> depth-1 VLM snapshot -> Qwen3-VL ROCm -> caption
```

本机 M2 长稳 Gate 已通过：持续 1800.130 秒，capture/inference/display 为
29.820/29.820/29.812 FPS，capture-to-display P95 为 14.951 ms，摄像头读取失败 0、GPU
reset/fault/hang 0。完整证据见 `output/realtime/metrics-camera-30m.json` 和
`output/realtime/m2-gate.json`。

M5 的 30 秒真实窗口短测为 capture/inference/display 29.859/25.893/29.793 FPS，显示 P95
28.721 ms，Qwen3-VL 请求 6/6 成功、均值 2.119 秒。主模型 37/37 层和视觉 mmproj 均在
`ROCm0`；程序拒绝部分 offload 或 CPU fallback。

Python 环境只由 `uv` 管理，`.venv`、依赖 cache 和应用配置均位于本仓库内。MIGraphX 和
VA-API 录制属于后续独立里程碑。

## 初始化

前提：本机已有受支持的 ROCm 7.2.1、Python 3.12、uv、Git、curl 和可访问的摄像头。

```bash
bash scripts/setup_native_gfx1151.sh
```

该脚本下载并校验锁定的 AMD wheel、上游源码和 `yolo26x.pt`，再运行环境检查与测试。它不
安装系统包，也不向系统 Python 或用户 site-packages 写入内容。

启用 VLM 还需执行：

```bash
bash scripts/setup_vlm_gfx1151.sh
uv run --frozen --extra vlm python scripts/check_vlm_gpu.py
```

第一条命令在仓库内构建锁定的 llama.cpp 并下载两份 GGUF；第二条发送一张真实图片并验证
唯一 `gfx1151` code object、37/37 GPU 层、mmproj=`ROCm0`、`/dev/kfd` 与模型 SHA。

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

运行 YOLO + Qwen3-VL 实时字幕：

```bash
uv run --frozen python scripts/run_camera.py \
  --device /dev/video0 --fourcc NV12 \
  --width 1280 --height 720 --camera-fps 30 \
  --model models/yolo26x.pt --backend pytorch \
  --display --record off \
  --vlm llamacpp --vlm-interval 6
```

VLM 服务只监听随机的 localhost 端口，由程序健康检查并在退出时清理。字幕 worker 只读取
最新快照，深度固定为 1；单次 VLM 超时不会阻塞摄像头、YOLO 或 UI。

## 验证

```bash
uv run --frozen ruff check src scripts tests
uv run --frozen python -m pytest -q
uv run --frozen python scripts/check_gfx1151_environment.py \
  --require-torch --output native-lock/environment.json
uv run --frozen python scripts/evaluate_m5_metrics.py \
  --metrics output/realtime/metrics-yolo-vlm-display-30s.json \
  --gpu-kernel-error-count 0
```

完整设计、实时状态和实测数据见 [doc/amd_395_pipeline.md](doc/amd_395_pipeline.md)。
