# VLM Camera Pipeline

AMD Ryzen AI MAX+ 395 / Radeon 8060S（`gfx1151`）本地低延迟摄像头 Demo。

当前已实现 M0-M2 的 PyTorch ROCm 主路径、M3 的 ONNX Runtime MIGraphX host-copy 路径、
M5 的 Qwen3-VL GPU 异步字幕，以及不加载 YOLO 的纯 VLM 中文字幕 Demo：

```text
V4L2 camera -> depth-1 latest frame -> YOLO26x ROCm -> latest result -> OpenCV UI
                                  \-> depth-1 VLM snapshot -> Qwen3-VL ROCm -> caption
V4L2/video  -> 25/30 FPS live display -> fixed subtitle panel
                    \-> every 3 s latest frame -> Qwen3-VL ROCm -> Chinese caption
V4L2/video  -> depth-1 latest frame -> YOLO26x ONNX/MIGraphX FP16 -> boxes
                    \-> independent Qwen3-VL worker -> subtitle panel
```

本机 M2 长稳 Gate 已通过：持续 1800.130 秒，capture/inference/display 为
29.820/29.820/29.812 FPS，capture-to-display P95 为 14.951 ms，摄像头读取失败 0、GPU
reset/fault/hang 0。完整证据见 `output/realtime/metrics-camera-30m.json` 和
`output/realtime/m2-gate.json`。

M5 的 30 秒真实窗口短测为 capture/inference/display 29.859/25.893/29.793 FPS，显示 P95
28.721 ms，Qwen3-VL 请求 6/6 成功、均值 2.119 秒。主模型 37/37 层和视觉 mmproj 均在
`ROCm0`；程序拒绝部分 offload 或 CPU fallback。

纯 VLM 摄像头窗口的 15 秒实测为 capture/display `29.878/28.081 FPS`，VLM `6/6`
成功、平均 `1.649 秒`，检测器状态为 `off/loaded=false`。字幕 Unicode 面板只在内容或状态
变化时重绘，视频帧不等待 VLM 推理。

MIGraphX + VLM 的 20 秒并发短测在 25.014 FPS 视频源上处理了 476 个 YOLO frame
（23.766 FPS），VLM 7/7 成功、均值 2.014 秒。YOLO 的 provider 为
`MIGraphXExecutionProvider`、FP16 已启用，严格 graph preflight 禁止 CPU EP fallback；
当前用户分支尚未打开 MIGraphX I/O Binding，因此如实记录为 host-copy，不宣称零拷贝。
15 秒可见窗口复测为 capture/inference/display `24.992/23.397/24.992 FPS`，显示 P95
`31.371 ms`，VLM `6/6` 成功。
真实 NV12 1280×720@30 摄像头的 15 秒复测为 `29.923/25.858/29.657 FPS`，显示 P95
`28.606 ms`、VLM `6/6` 成功且摄像头读取失败为 0。

Python 环境只由 `uv` 管理，`.venv`、依赖 cache 和应用配置均位于本仓库内。VA-API 录制
仍属于后续独立里程碑。

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

启用 ONNX Runtime MIGraphX 路径：

```bash
bash scripts/setup_migraphx_gfx1151.sh
```

该脚本使用用户提供的
`zihaomu/ultralytics:add-onnx-migraphx-backend`，下载 AMD ROCm 7.2.1 官方 cp312
`onnxruntime-migraphx` wheel 与 `yolo26x.onnx`，全部放入仓库并由 uv lock 管理。首次生成
gfx1151 `.mxr` cache 实测约需 109 秒，之后热启动约 2 秒。

## 运行实时 Demo

只运行 Qwen3-VL，并在视频下方显示中文字幕：

```bash
uv run --frozen python scripts/run_vlm_demo.py \
  --device /dev/video0 --fourcc NV12 \
  --width 1280 --height 720 --camera-fps 30 \
  --vlm-interval 3 --window-scale 0.75
```

也可用仓库内锁定视频循环演示；它按视频原始 25 FPS 播放，不会高速读取：

```bash
uv run --frozen python scripts/run_vlm_demo.py \
  --video-file third_party/notebook/ultralytics_yolo26/data/sidewalk.mp4 \
  --vlm-interval 3
```

两种纯 VLM 模式都不会构造 YOLO/PyTorch detector。视频持续播放，VLM 独立读取深度为 1 的
最新快照；上一条字幕保留到新字幕就绪，并显示“正在理解”状态。按 `q`、Esc 或 `Ctrl+C`
退出。窗口默认按原始画布的 75% 创建，可直接拖拽边框缩放；窗口聚焦时按 `+`/`-` 可逐级
放大/缩小，按 `0` 回到 `--window-scale` 指定的初始比例。如果仍受屏幕空间限制，可改用
`--window-scale 0.6`。

本机 `amd_isp_capture` 在重复开关摄像头后存在偶发不发布首帧的已知问题；循环视频入口不受
影响。摄像头 15 秒短测已通过，但在驱动恢复完成长期验证前不宣称摄像头长稳通过，细节见
实施文档的 M5 小节。

运行仅 YOLO 的摄像头 Demo：

```bash
uv run --frozen python scripts/run_camera.py \
  --device /dev/video0 \
  --fourcc NV12 \
  --width 1280 --height 720 --camera-fps 30 \
  --model models/yolo26x.pt \
  --backend pytorch \
  --display --record off --vlm off
```

无窗口 YOLO smoke 示例：

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

运行独立的 ONNX Runtime MIGraphX YOLO + Qwen3-VL Demo：

```bash
uv run --frozen --extra migraphx --extra vlm \
  python scripts/run_yolo_vlm_demo.py \
  --device /dev/video0 --fourcc NV12 \
  --width 1280 --height 720 --camera-fps 30 \
  --vlm-interval 3 --window-scale 0.75
```

如果摄像头暂时不发布首帧，可用锁定视频验证完整窗口：

```bash
uv run --frozen --extra migraphx --extra vlm \
  python scripts/run_yolo_vlm_demo.py \
  --video-file third_party/notebook/ultralytics_yolo26/data/sidewalk.mp4
```

该入口与 `run_vlm_demo.py` 完全分开。视频区域显示 YOLO 框，下面显示中文 VLM 字幕；窗口
同样支持拖拽、`+`/`-` 缩放、`0` 复位以及 `q`/Esc 退出。字幕内容只显示在窗口中，metrics
JSON 默认写入脱敏占位符，不持久化真实摄像头描述。

## 验证

```bash
uv run --frozen ruff check src scripts tests
uv run --frozen python -m pytest -q
uv run --frozen python scripts/check_gfx1151_environment.py \
  --require-torch --output native-lock/environment.json
uv run --frozen python scripts/evaluate_m5_metrics.py \
  --metrics output/realtime/metrics-yolo-vlm-display-30s.json \
  --gpu-kernel-error-count 0
uv run --frozen --extra migraphx --extra vlm \
  python scripts/check_migraphx_backend.py --iterations 20
```

完整设计、实时状态和实测数据见 [doc/amd_395_pipeline.md](doc/amd_395_pipeline.md)。
