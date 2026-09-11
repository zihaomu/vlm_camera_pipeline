#!/usr/bin/env python3
"""Probe the HIP DMA-BUF -> VAAPI VPP -> H.264 path with one camera frame."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstAllocators", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import Gst, GstAllocators, GstVideo

from src.zerocopy_camera import StrictGpuCamera

CAMERA_LIBRARY = ".local/zerocopy-gfx1151/lib/libvlm_camera_gpu_capture.so.0.1.0"


def main() -> int:
    Gst.init(None)
    pipeline = Gst.parse_launch(
        "appsrc name=source is-live=true format=time block=true "
        "caps=video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=YUYV,"
        "width=1280,height=720,framerate=30/1 "
        "! queue max-size-buffers=2 "
        "! vapostproc "
        "! video/x-raw(memory:VAMemory),format=NV12,width=1280,height=720 "
        "! vah264enc bitrate=4000 key-int-max=30 "
        "! h264parse "
        "! fakesink sync=false"
    )
    source = pipeline.get_by_name("source")
    allocator = GstAllocators.DmaBufAllocator.new()
    lease = None
    camera = None
    frame = None
    try:
        camera = StrictGpuCamera(
            workspace=WORKSPACE,
            library=CAMERA_LIBRARY,
            web_stream_buffers=1,
        )
        frame = camera.acquire(timeout_ms=3000)
        if frame is None:
            raise RuntimeError("camera returned no frame")
        lease = frame.detach_web_stream()
        if lease is None:
            raise RuntimeError("camera returned no web DMA-BUF lease")
        frame.release()
        frame = None

        fd = lease.duplicate_fd()
        memory = GstAllocators.DmaBufAllocator.alloc(
            allocator,
            fd,
            lease.allocation_bytes,
        )
        buffer = Gst.Buffer.new()
        buffer.append_memory(memory)
        GstVideo.buffer_add_video_meta_full(
            buffer,
            GstVideo.VideoFrameFlags.NONE,
            GstVideo.VideoFormat.YUY2,
            lease.width,
            lease.height,
            1,
            [0, 0, 0, 0],
            [lease.width * 2, 0, 0, 0],
        )
        buffer.pts = 0
        buffer.dts = 0
        buffer.duration = Gst.SECOND // 30

        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline refused PLAYING")
        result = source.emit("push-buffer", buffer)
        if result != Gst.FlowReturn.OK:
            raise RuntimeError(f"appsrc push failed: {result.value_nick}")
        source.emit("end-of-stream")
        message = pipeline.get_bus().timed_pop_filtered(
            10 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        if message is None:
            raise RuntimeError("timed out waiting for encoder")
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            raise RuntimeError(f"GStreamer error: {error}; {debug}")
        print("PASS: HIP linear YUYV DMA-BUF imported by VAAPI and encoded as H.264")
        return 0
    finally:
        pipeline.set_state(Gst.State.NULL)
        if frame is not None:
            frame.release()
        if lease is not None:
            lease.release()
        if camera is not None:
            camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
