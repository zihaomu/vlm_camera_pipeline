# Prompt-only Web 前端与 GPU 视频流实施记录

> 状态：**GPU YUYV 烧框版已落地，真实 Camera + YOLO26 + Qwen3-VL 端到端短测通过；长稳尚未执行**
> 日期：2026-09-10
> 平台：AMD Ryzen AI MAX+ 395 / Radeon `gfx1151`
> 基线：[Hybrid YOLO+VLM](./hybrid_yolo_vlm_design.md) 与
> [严格零主机图像拷贝](./zerocopy_yolo_vlm_design.md)

## 1. 产品边界

网页把大部分空间留给实时摄像头，检测框和 class name 直接叠加在视频上，VLM 字幕固定在
视频下方，不会再被屏幕边缘遮挡。用户界面只有一个可编辑项：**VLM prompt**。以下内容全部
只读展示，不提供开关、下拉框或滑块：

- Camera 实际 FPS；
- YOLO completion FPS 与单次 inference ms；
- VLM 单次端到端秒数、generation tok/s 和 `RUNNING/IDLE`；
- 固定 `3.0 s` 调用节拍；
- Camera、OpenCV HIP、MIGraphX、llama.cpp HIP IPC 和 VAAPI 编码路径。

原 EGL 窗口入口继续保留；网页是独立启动方式，不改变 clean/Hybrid 回归入口。

## 2. 启动方式

首次同步网页可选依赖仍使用仓库内 uv 环境：

```bash
uv sync --frozen --extra vlm --extra migraphx --extra web
```

启动完整网页 Demo：

```bash
./scripts/run_yolo_vlm_web_zerocopy.sh
```

默认自动打开 `http://127.0.0.1:8765/`。该入口固定使用 Hybrid、VLM `3.0 s` cadence、
top-8 hints 与 `zero-copy=require`。服务只允许绑定 `127.0.0.1` 或 `localhost`；当前没有认证，
因此不允许通过参数直接暴露到局域网。

该网页入口还默认传入 `--camera-horizontal-flip`，把 `/dev/video0` 的前置摄像头镜像在 GPU
NV12 转换阶段还原成物理视角。接入本来就不镜像的后置/外置摄像头时，可在命令末尾传
`--no-camera-horizontal-flip` 覆盖默认值。

## 3. 数据路径

```text
Camera NV12 (HIP/HSA DMA-BUF)
  ├─ HIP NV12 -> RGB8 + optional horizontal unmirror ──> YOLO26 MIGraphX ──> GPU [300,6]
  │                              ├─> latest result -> HIP box/class compositor ─────┐
  │                              └─> exact-frame Hybrid GPU overlay -> Qwen3-VL HIP IPC
  └─ HIP NV12 -> packed YUYV + optional horizontal unmirror ─> linear DMA-BUF <────┘
                                  (in-place colored boxes + COCO class names)
                                  -> VAAPI VPP YUYV -> native tiled NV12
                                  -> VAAPI H.264 -> fragmented MP4 -> WebSocket -> browser MSE
```

本机 `vah264enc` 只接受带 AMD 原生 modifier 的 NV12 surface，HIP/HSA 导出的线性 NV12
不能直接伪装成该 surface。实施中使用 VAAPI 明确支持的线性 `YUYV` DMA-BUF 输入：一个 HIP
kernel 在 GPU 上完成 NV12→YUYV 打包。最新 YOLO `[300,6]` tensor 随后直接在同一 YUYV
allocation 上烧入 20 色 class 框和 COCO class name；每个 HIP thread 完整拥有一个 YUYV pixel
pair，以免共享 U/V 产生相邻像素写竞争。完成后仅同步该资源自己的 HIP event，再由 VAAPI VPP
在 GPU 上生成编码器需要的原生 NV12。禁止协商失败后回退到 `videoconvert`、CPU map 或软件
x264。

CPU/浏览器必然会接触已经压缩的 H.264/fMP4 网络字节；这不属于摄像头像素回读。CPU 还会
接触 prompt、字幕与性能 JSON。浏览器不再接收逐帧检测框坐标，原 500 ms 状态轮询只更新
只读文字状态，不参与视频画框。每 3 秒 Hybrid VLM 请求仍按原设计下载固定 260 bytes top-8
提示元数据来构造文字 prompt；它不是 Web overlay，也不会限制视频框刷新率。生产计数器继续
分别保证：

```text
camera/yolo/vlm/web-encoder image_h2d_bytes = 0
camera/yolo/vlm/web-encoder image_d2h_bytes = 0
full_detection_tensor_d2h_bytes = 0
web_detection_control_metadata_d2h_bytes = 0
```

COCO80 字形 alpha atlas（`80×224×36` 加 80 个宽度值，共 `645440 bytes`）只在启动时上传
一次，属于静态 UI 控制资源；逐帧 overlay 没有 H2D/D2H。框已经成为 H.264 视频像素，因此视频
缩放、MSE 播放和丢帧时始终与画面保持同一时间轴，不再出现 Canvas 以约 2 FPS 跳动的问题。
前置摄像头去镜像同样融合在已有的 NV12→RGB/YUYV kernel 中，不增加额外 kernel pass；检测和
VLM 在翻转后的 RGB 上运行，YUYV 编码画面使用完全相同的坐标系，class name 则在翻转后烧入，
因此文字保持正向。

## 4. Prompt 热更新语义

Prompt 由线程安全的 `PromptStore` 管理，最大 512 字符，空字符串和非法控制字符会返回 HTTP
400。每次修改生成单调递增的 version。调度器在某个 3 秒请求被 armed 时原子读取
`(text, version)`，该请求从合成到结果始终携带同一版本；网页分别显示：

- 当前已保存版本；
- 已进入 VLM 的 scheduled version；
- 最新字幕实际使用的 applied version。

用户 prompt 是最高优先级指令；同帧 YOLO hints 仍被标记为“可能错误或不完整”，禁止模型在
字幕中提到框、编号或分数。中文 prompt 使用中文辅助约束，避免英文 hint 模板改变输出语言。
自定义语言不使用原来的 English-only GBNF，但保留多语言首句 stop 和 32-token 硬上限，兼顾
自由度与 3 秒体验。

## 5. 视频流与生命周期

- Camera 侧是 8 个固定 YUYV DMA-BUF slot，不随帧申请无限内存；
- `appsrc`、queue 和 encoder 都是有界 latest-oriented 策略，不能反压 Camera/YOLO/VLM；
- H.264 使用 CBR、关闭 B frame，keyframe 间隔 6 帧，MP4 fragment 为 50 ms；
- `vah264enc` 会改写输出 PTS 并清除 frame offset，因此在 encoder sink 记录实际进入编码器的
  input PTS 队列，在无 B frame 的 encoder src 按序配对；完成 lease 仍由 Camera 创建线程统一
  归还，不跨线程调用 V4L2 生命周期；
- 无浏览器连接时不提交编码帧，Camera、YOLO 和 VLM 继续运行；
- 有浏览器连接时，最新 YOLO GPU tensor 在每个待编码 frame 上执行一次 in-place HIP burn-in；
- WebSocket 新客户端先收到缓存的 `ftyp+moov`，再从下一个 `moof` fragment 开始；
- 服务端解析 `tfhd/trun` sample flags，新客户端只从 sync sample 开始，并把每个
  `moof+mdat` 合并成一次原子 WebSocket/MSE media-segment append；
- encoder 输入 PTS/duration 使用 Camera 单调采集时间间隔；服务端另外把已完成 fragment 的
  真实 capture timestamp 暴露给只读诊断，不再以 mux 时间轴推测画面年龄；
- 浏览器先积累 0.36 秒媒体、在约 0.16 秒 live latency 处启动；低于 0.12 秒时短暂使用
  `0.985×` 播放率回补，回到 0.17 秒后恢复 1×，真正断粮持续 180 ms 才进入
  rebuffer；高于 0.40 秒立即回到 live target；
- 浏览器只保留约 10 秒已播放 MSE buffer，长时间运行不会无限累积视频内存；
- 退出时先停止编码器并回收全部 web lease，再关闭 Camera，避免 DMA-BUF 悬挂。

## 6. 已完成验证

### 6.1 组件与静态测试

- HIP linear YUYV DMA-BUF → VAAPI VPP → VAAPI H.264 单帧真实 Camera probe：通过；
- 离线 NV12→YUYV GPU kernel 与 CPU packed reference 逐字节一致，最大误差为 0；
- 水平去镜像的 RGB8 与 YUYV GPU 输出均与 CPU reference 逐字节一致，最大误差为 0；
- 90 帧 Camera/encoder 压测：submitted/encoded/released=`90/90/90`，reject=`0`，fatal=`0`；
- fMP4 含 `ftyp`、`moov`、`moof`；
- Ruff：通过；
- pytest：`56 passed`，覆盖 prompt 边界/version、MP4 分箱/原子组段/sync sample、Camera PTS、
  UI 唯一输入项、状态 schema 和 GPU YUYV burn-in source/ABI contract。
- Firefox headless `1440x1000` 静态布局检查通过，视频、字幕、完整 prompt 表单和性能侧栏均在
  同一视口内；证据：`output/realtime/web-prompt-ui-1440x1000.png`。
- Web YUYV overlay 离线数值/视觉检查通过：`person` 与 `traffic light` 的 palette、class name、
  box interior 和 confidence filter 全部通过，首次 640×360 kernel+局部 event 边界为 `0.422 ms`；
  证据为 `output/realtime/web-gpu-overlay-check.json` 和
  `output/realtime/web-gpu-overlay-check.png`。

### 6.2 完整端到端短测

证据：`output/realtime/metrics-yolo-vlm-web-e2e.jsonl`。

| 项目 | 结果 |
| --- | ---: |
| 持续时间 / Camera frames | 25.022 s / 744 |
| Camera effective FPS | 29.734 |
| YOLO completion FPS / p50 | 29.694 / 27.216 ms |
| VLM completed / failed | 8 / 0 |
| VLM p50 / p95 | 2.319 / 3.041 s |
| Prompt 热更新 | v2 scheduled、v2 applied |
| v2 实际字幕 | `画面中的人物是穿着白色T恤的男性，正在做手势动作.` |
| Camera web converted/released/drop | 744 / 744 / 0 |
| 退出时 active web leases | 0 |
| Web encoder fatal error | 0 |
| 图像 H2D / D2H | 0 / 0 |

同轮通过 WebSocket 把 fMP4 喂给 `ffprobe`，识别为 `H.264 Constrained Baseline`、
`1280x720`、`yuv420p`。探针在识别出 stream 后主动关闭 pipe，因此该轮只用于证明浏览器负载
可解码，不用于视频长稳或编码 FPS Gate。

首个/冷态 VLM 请求仍可能略超过 3 秒，本轮因此记录 1 次 missed deadline；随后请求保持
latest-only、无 backlog，视频和 YOLO 不等待 VLM。发布前仍需在真实 Chromium 可见页面完成
实时播放 QA，并跑一轮不少于 20 分钟的网页连接长稳；在这两项完成前不宣称 Web Gate
最终通过。

### 6.3 GPU 烧框版完整短测（2026-09-10）

证据：`output/realtime/metrics-yolo-vlm-web-burnin-e2e.jsonl`，解码帧：
`output/realtime/web-gpu-burnin-camera.png`。

| 项目 | 结果 |
| --- | ---: |
| 持续时间 / Camera frames | 35.030 s / 1043 |
| Camera effective FPS | 29.774 |
| YOLO completion FPS / p50 | 29.746 / 25.494 ms |
| Web GPU overlay draws | 418 |
| Web overlay p50 / p95 / max | 0.200 / 0.414 / 0.654 ms |
| VLM completed / failed | 12 / 0 |
| VLM p50 / p95 | 2.238 / 2.770 s |
| VLM missed deadline / schedule drop | 0 / 0 |
| Camera web converted/released/drop | 1043 / 1043 / 0 |
| Encoder submitted/encoded/released/reject | 418 / 418 / 418 / 0 |
| Web detection metadata D2H | 0 bytes |
| 图像 H2D / D2H | 0 / 0 |

WebSocket 捕获的 7.733 秒 fMP4 由 `ffprobe` 识别为 H.264 Constrained Baseline、
`1280×720@30`、`yuv420p`。第五秒解码帧中可直接看到不同颜色的 `person`、`chair`、`tv`
框和 class name，证明标注已经进入编码视频，而不是浏览器 DOM/Canvas 图层。该轮所有 short
runtime checks 通过，YOLO/VLM failure 为 0，退出时 8-slot Web pool 全部归还。

### 6.4 前置摄像头 GPU 去镜像回归（2026-09-10）

证据：`output/realtime/metrics-yolo-vlm-web-unmirrored-e2e.jsonl`，解码帧：
`output/realtime/web-gpu-unmirrored-camera.png`。

| 项目 | 结果 |
| --- | ---: |
| 持续时间 / Camera frames | 38.017 s / 1132 |
| Camera effective FPS | 29.776 |
| YOLO completion FPS / p50 | 29.723 / 26.257 ms |
| Web GPU overlay draws | 239 |
| Web overlay p50 / p95 / max | 0.206 / 0.494 / 0.584 ms |
| VLM completed / failed | 13 / 0 |
| VLM p50 / p95 | 2.350 / 2.782 s |
| Camera web converted/released/drop | 1132 / 1132 / 0 |
| Encoder submitted/encoded/released/reject | 239 / 239 / 239 / 0 |
| 图像 H2D / D2H | 0 / 0 |

Camera runtime manifest 记录 `horizontal_flip=true` 与
`orientation=front-camera physical view (GPU horizontal unmirror)`。WebSocket 捕获的 fMP4 经
`ffprobe` 确认为 H.264 Constrained Baseline、`1280×720@30`、`yuv420p`；解码帧中的框与目标
对齐，class name 正向可读。全部 short runtime checks 通过，退出时 Camera clean/web lease
均为 0。

### 6.5 浏览器视频卡顿定位与修复（2026-09-10）

基线 Camera/encoder 始终接近 30 FPS，录制 fMP4 的 PTS 也是连续的，但 Firefox 的 MSE
buffered-ahead 只剩约 `0.17 s`，15 秒内出现 3 次 `waiting→playing`。因此卡顿位于浏览器播放
边界，不是 Camera、YOLO 或 GPU 烧框吞吐不足。根因包括：启动即播放、`moof/mdat` 分两次
append、新连接未等待同步帧，以及固定 30 FPS PTS 与相机实际约 29.8 FPS 的长期漂移。

修复后以真实 Camera + YOLO26 + Qwen3-VL 运行 `463.601 s`，Camera `29.812 FPS`、YOLO
`29.694 FPS`，encoder submitted/encoded/released=`12547/12547/12547`，reject=`0`，Camera
timestamp discontinuity=`0`，图像 H2D/D2H=`0/0`。Firefox 专项结果：

| 浏览器窗口 | 呈现 | waiting / stalled | dropped | 缓冲 |
| --- | ---: | ---: | ---: | ---: |
| 启动后 20.008 s | 1.139 s 开播，566 frames | 0 / 0 | 0 | 约 0.8 s 目标 |
| 稳态 40.001 s | 1192 frames，29.799 FPS | 0 / 0 | 0 | 0.722→0.729 s |

10 秒 WebSocket 探针收到 1 个 `ftyp+moov` init segment 和 59 个完整 `moof+mdat` media
segment，分片到达间隔 p50/p95/max=`231.355/242.708/244.757 ms`。完整证据：
`output/realtime/web-playback-smooth-check.json`、
`output/realtime/metrics-yolo-vlm-web-playback-keyframe-final.jsonl` 与
`output/realtime/web-playback-smooth-final.png`。

### 6.6 端到端延迟二次定位与修复（2026-09-10）

6.5 解决了 MSE 断粮卡顿，但主动保留的 `0.8 s` target buffer、`1.8 s` 追赶阈值和 250 ms
fragment 本身就会产生明显操作延迟。旧策略 20 秒窗口的 MSE buffer p50/p95 为
`0.637/0.751 s`；再叠加尚未完成的 fragment 与编码/解码，接近 1–2 秒的肉眼体感符合配置，
不是 Camera 或 YOLO 帧率不足。

检查编码生命周期还发现，encoder src PTS 带有 GStreamer VideoEncoder 的固定 1000 小时
偏移，输出 `buffer.offset` 为 `GST_BUFFER_OFFSET_NONE`；旧的“按 encoder src PTS 回收 input
lease”会错误地把尚未消费的 DMA-BUF 判定为完成。最终修复为：

- appsrc/queue 各限制为 2 帧，保留 latest-oriented 丢弃语义；
- encoder sink 记录真正进入编码器的 input PTS，encoder src 按序配对后通知 Camera 主线程
  释放 lease；真实运行 `encoder_input_mapping_misses=0`；
- keyframe 间隔从 15 降到 6 帧，fragment 从 250 ms 降到 100 ms；
- browser startup/target/max buffer 调整为 `0.45/0.25/0.60 s`；
- 每个完成 fragment 关联最后编码帧的 Camera capture timestamp，诊断值直接表示画面年龄。

在其余参数一致时也实际比较了 CBR 与 CQP：CBR capture→display p50/p95 为
`209/245 ms`，CQP 为 `222/273 ms`，两者都无 dropped/rebuffer。CBR 在本机略快且维持固定
码率，因此最终保留 CBR；这也排除了“CBR 是 2 秒延迟主因”的假设。

Firefox 20 秒稳态专项结果如下：

| 指标 | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| Camera capture→display | 83 ms | 209 ms | 245 ms | 259 ms |
| 最后编码帧→MSE live edge | 8 ms | 56 ms | 100 ms | 129 ms |
| MSE buffered ahead | 19 ms | 147 ms | 214 ms | 228 ms |

窗口内呈现 502 帧、dropped delta=`0`、confirmed rebuffer delta=`0`、append overflow=`0`。
真实 Camera+YOLO+VLM CBR 运行 `50.013 s`，Camera `29.772 FPS`；Web encoder
submitted/input/encoded/released=`1082/1081/1081/1082`，最后一帧未进入 encoder 的有界队列在
关闭时安全释放，reject=`0`，input mapping miss=`0`，capture→encoder 最后/最大=
`6.337/101.706 ms`，退出时 active clean/web lease=`0/0`，图像 H2D/D2H=`0/0`。证据为
`output/realtime/web-playback-low-latency-check.json` 与
`output/realtime/metrics-yolo-vlm-web-cbr-content-latency-probe.jsonl`。

### 6.7 保留 MSE 的第三次低延迟收敛（2026-09-10）

保留 WebSocket + fMP4/MSE 传输，不引入 WebRTC 会话和信令层。这一轮把剩余延迟拆成
“已编码帧等待 fragment 完成”与“MSE live edge 到当前播放点”两部分。A/B 结果表明：

- 把 fragment 从 100 ms 降到 50 ms 可把 capture→MSE edge 中位数从约 56 ms 降到
  约 22 ms；`appsink processing-deadline=0` 同时去掉不需要的 sink latency 声明。
- 120 ms target 的激进方案曾做到 p50/p95=`127/201 ms`，但 25 秒窗口有 9 帧
  dropped，因此不采用。
- 250 ms target 加 50 ms fragment 虽然零 dropped，但 p50/p95=`228/276 ms`，说明只缩短
  fragment 不足以降低总延迟。
- 最终使用 startup/target/max=`0.36/0.16/0.40 s`，low/recovered=`0.12/0.17 s`，
  低水位时仅以 `0.985x` 短暂恢复缓冲；不使用倍速追帧，也不在稳态额外 seek。

Firefox headless 先预热 5 秒，随后的 40 秒稳态窗口结果如下：

| 指标 | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| Camera capture→display | 126 ms | 156 ms | 196 ms | 215 ms |
| 最后编码帧→MSE live edge | 5 ms | 22 ms | 38 ms | 41 ms |
| MSE buffered ahead | 99 ms | 151 ms | 185 ms | 213 ms |

窗口内呈现 1192 帧，dropped delta=`0`、confirmed rebuffer delta=`0`、append queue
overflow=`0`、稳态 live seek=`0`；append queue p95=`0`、max=`1`。相比 6.6 的
`209/245 ms`，p50/P95 分别降低约 `25%/20%`。头部 5 秒的 Firefox
`droppedVideoFrames` 包含新 MSE 连接的 seek/解码预滚计数，因此稳态 gate 单独计算，
不把预滚帧冒充为正常播放丢帧。

同时完成真实 Camera + YOLO26 + Qwen3-VL `95.003 s` 运行：Camera `29.788 FPS`、
YOLO completion `29.620 FPS`，encoder submitted/input/encoded/released=
`1374/1372/1372/1374`，reject=`0`、input mapping miss=`0`、output without input=`0`。退出时
Camera clean/web active lease=`0/0`，生产图像 H2D/D2H=`0/0`，说明这次 MSE 调优没有破坏
GPU 零主机图像拷贝契约。证据为
`output/realtime/web-playback-mse-fast-check.json` 与
`output/realtime/metrics-yolo-vlm-web-mse-fast-final.jsonl`。

### 6.8 AMD 加速元素视觉标识（2026-09-10）

在不改变视频占比和交互层级的前提下，前端增加一套统一的 AMD 加速标识：

- 浏览器 favicon 和页面左上角主标识都使用同一 AMD 图标；
- 页头右侧增加 `AMD / GPU ACCELERATED` 紧凑标牌，小屏自动收缩为图标；
- 视频状态栏增加 `AMD GPU` 徽章；
- GPU data path 的 ISP、HIP、MIGraphX、ROCm 和 VAAPI 节点使用 AMD 图标技术标签；
- 浏览器 favicon 和页面左上角的项目图标使用同一 AMD mark；
- 深色背景上的 AMD mark 统一使用白色单色版，边框为中性灰；绿色继续只表示
  live/healthy，不混淆品牌和运行状态语义。

所有标识共用 HTML 中的一个 inline SVG symbol，不增加图片文件、HTTP 请求或 JavaScript
执行。已在 `1440×1000`、`1000×1400` 和 `720×1200` 三种视口完成 Firefox 渲染检查，
页头、视频区域和数据路径均无溢出。截图为
`output/realtime/web-amd-branding-1440x1000.png`、
`output/realtime/web-amd-branding-1000x1400.png` 与
`output/realtime/web-amd-branding-720x1200.png`；左上角主标识更新后的复验截图为
`output/realtime/web-amd-home-icons-1440x1000.png`。实际 aiohttp 路由已确认以
`image/svg+xml` 返回 `web/amd-mark.svg`。

## 7. 文件落点

- `web/index.html`、`web/app.css`、`web/app.js`：响应式页面与 MSE 播放，不含检测 Canvas；
- `src/zerocopy_web.py`：prompt store、aiohttp、GPU YUYV burn-in、MP4 分发与 VAAPI encoder；
- `src/zerocopy_camera.py`、`native/camera_dmabuf/*`：固定 YUYV GPU DMA-BUF lease；
- `native/zerocopy_kernels/zerocopy_kernels.hip`：YUYV pixel-pair 彩色框/class atlas compositor；
- `src/zerocopy_hybrid.py`：Web prompt 校验、版本与 VLM bounded hints；
- `scripts/run_yolo_vlm_web_zerocopy.sh`：独立推荐入口；
- `scripts/probe_web_dmabuf.py`：底层 DMA-BUF→VAAPI 探针。
- `scripts/check_web_gpu_overlay.py`：明确标记 validation-only D2H 的 overlay 数值/视觉检查。
