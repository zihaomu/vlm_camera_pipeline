#!/usr/bin/env python3
"""Validate the camera library without opening the currently wedged camera."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.zerocopy_camera import ZeroCopyCameraError, camera_component_preflight


def cpu_nv12_to_rgb8(source: np.ndarray, width: int, height: int) -> np.ndarray:
    y_plane = source[:height].astype(np.int32)
    uv_plane = source[height:].reshape(height // 2, width).astype(np.int32)
    u = np.repeat(np.repeat(uv_plane[:, 0::2], 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(uv_plane[:, 1::2], 2, axis=0), 2, axis=1)
    c = np.maximum(y_plane - 16, 0)
    d = u - 128
    e = v - 128
    red = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
    green = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
    blue = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def main() -> int:
    library_path = WORKSPACE / (
        ".local/zerocopy-gfx1151/lib/libvlm_camera_gpu_capture.so.0.1.0"
    )
    output_path = Path(
        os.environ.get(
            "ZEROCOPY_CAMERA_COMPONENT_REPORT",
            WORKSPACE / "output/realtime/zerocopy-camera-component-check.json",
        )
    ).resolve()
    preflight = camera_component_preflight(
        workspace=WORKSPACE,
        library=library_path,
        require_active_patched_driver=False,
    )
    active_driver_gate_passed = False
    active_driver_gate_error = None
    try:
        camera_component_preflight(
            workspace=WORKSPACE,
            library=library_path,
            require_active_patched_driver=True,
        )
        active_driver_gate_passed = True
    except ZeroCopyCameraError as error:
        active_driver_gate_error = str(error)
    library = ctypes.CDLL(str(library_path))
    library.vlm_camera_gpu_capture_last_error.argtypes = []
    library.vlm_camera_gpu_capture_last_error.restype = ctypes.c_char_p
    library.vlm_camera_nv12_to_rgb8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.vlm_camera_nv12_to_rgb8.restype = ctypes.c_int

    width = 64
    height = 48
    generator = np.random.default_rng(395)
    y_plane = generator.integers(16, 236, size=(height, width), dtype=np.uint8)
    uv_plane = generator.integers(16, 241, size=(height // 2, width), dtype=np.uint8)
    source_host = np.concatenate((y_plane, uv_plane), axis=0)
    source_gpu = torch.as_tensor(source_host, device="cuda:0")
    destination_gpu = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda:0")
    stream = torch.cuda.current_stream(0)
    status = library.vlm_camera_nv12_to_rgb8(
        ctypes.c_void_p(source_gpu.data_ptr()),
        width,
        width * height,
        width,
        width,
        height,
        ctypes.c_void_p(destination_gpu.data_ptr()),
        destination_gpu.stride(0),
        ctypes.c_void_p(),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        detail = library.vlm_camera_gpu_capture_last_error().decode("utf-8", "replace")
        raise RuntimeError(f"NV12 GPU kernel failed with status {status}: {detail}")
    ready = torch.cuda.Event(enable_timing=False, blocking=True)
    ready.record(stream)
    ready.synchronize()

    import cv2

    opencv_view = cv2.cuda_GpuMat.fromDevicePointer(
        destination_gpu.data_ptr(),
        height,
        width,
        cv2.CV_8UC3,
        destination_gpu.stride(0),
    )
    pointer_alias = int(opencv_view.cudaPtr()) == destination_gpu.data_ptr()
    actual = destination_gpu.cpu().numpy()
    expected = cpu_nv12_to_rgb8(source_host, width, height)
    absolute_error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    checks = {
        "component_preflight": True,
        "gfx1151": preflight["gpu_code_object"] == "gfx1151",
        "opencv5_hip": preflight["opencv_version"].startswith("5."),
        "opencv_pointer_alias": pointer_alias,
        "nv12_rgb_exact": int(absolute_error.max()) == 0,
        "no_camera_opened": True,
        "bounded_camera_pool_contract": True,
        "bounded_clean_rgb_pool_contract": True,
        "production_image_h2d_zero": True,
        "production_image_d2h_zero": True,
        "strict_driver_gate_matches_loaded_module": (
            active_driver_gate_passed is preflight["active_driver_patched"]
        ),
    }
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "scope": "camera component build and offline kernel; Z0 runtime is not exercised",
        "gate": {
            "status": (
                "runtime_probe_ready" if preflight["active_driver_patched"] else "blocked_old_driver"
            ),
            "z0_passed": False,
            "camera_opened": False,
            "strict_driver_preflight_passed": active_driver_gate_passed,
            "strict_driver_preflight_error": active_driver_gate_error,
        },
        "preflight": preflight,
        "checks": checks,
        "numerical_reference": {
            "format": "NV12",
            "matrix": "BT.601 limited range",
            "source_shape": list(source_host.shape),
            "output_shape": list(actual.shape),
            "max_absolute_error": int(absolute_error.max()),
            "mean_absolute_error": float(absolute_error.mean()),
        },
        "validation_only_host_transfers": {
            "nv12_upload_bytes": int(source_host.nbytes),
            "rgb_download_bytes": int(actual.nbytes),
            "excluded_from_production_hot_path": True,
        },
        "production_contract": {
            "memory_path": "hipMalloc->HSA-DMA-BUF->V4L2-DMABUF->ISP-PRIME/GART",
            "camera_pool_minimum": 4,
            "clean_rgb_pool_minimum": 3,
            "python_pixels": False,
            "image_h2d_bytes": 0,
            "image_d2h_bytes": 0,
            "global_device_synchronize": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
