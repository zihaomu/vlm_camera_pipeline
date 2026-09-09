#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the repo-local OpenCV 5 HIP build")
    parser.add_argument("--expected-prefix", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_prefix = Path(args.expected_prefix).resolve()
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": False,
        "expected_prefix": str(expected_prefix),
        "python": sys.executable,
    }
    try:
        import cv2
        import numpy as np

        module_path = Path(cv2.__file__).resolve()
        device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        has_pointer_factory = hasattr(cv2.cuda_GpuMat, "fromDevicePointer")
        resize_doc = cv2.cuda.resize.__doc__ or ""
        has_align_corners = "align_corners" in resize_doc
        has_nms = hasattr(cv2.cuda, "nms")

        report.update(
            {
                "opencv_version": cv2.__version__,
                "cv2_path": str(module_path),
                "device_count": device_count,
                "has_from_device_pointer": has_pointer_factory,
                "has_resize_align_corners": has_align_corners,
                "has_cuda_nms": has_nms,
            }
        )

        if not str(module_path).startswith(str(expected_prefix) + "/"):
            raise RuntimeError(f"cv2 resolved outside repo-local prefix: {module_path}")
        if not cv2.__version__.startswith("5."):
            raise RuntimeError(f"expected OpenCV 5.x, got {cv2.__version__}")
        if device_count != 1:
            raise RuntimeError(f"expected exactly one HIP device, got {device_count}")
        if not has_pointer_factory:
            raise RuntimeError("cv2.cuda_GpuMat.fromDevicePointer is missing")
        if not has_align_corners:
            raise RuntimeError("cv2.cuda.resize align_corners binding is missing")
        if not has_nms:
            raise RuntimeError("cv2.cuda.nms is missing")

        source = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        gpu_source = cv2.cuda_GpuMat()
        gpu_source.upload(source)
        wrapped = cv2.cuda_GpuMat.fromDevicePointer(
            gpu_source.cudaPtr(), 16, 16, cv2.CV_8UC3, gpu_source.step
        )
        resized = cv2.cuda.resize(
            wrapped,
            (8, 8),
            interpolation=cv2.INTER_LINEAR,
            align_corners=True,
        )
        result = resized.download()
        if result.shape != (8, 8, 3) or not np.isfinite(result).all():
            raise RuntimeError(f"unexpected HIP resize result: {result.shape}")
        report["kernel_smoke"] = {
            "source_pointer": int(gpu_source.cudaPtr()),
            "wrapped_pointer": int(wrapped.cudaPtr()),
            "pointer_alias": int(gpu_source.cudaPtr()) == int(wrapped.cudaPtr()),
            "result_shape": list(result.shape),
            "result_checksum": int(result.astype(np.uint64).sum()),
        }
        if not report["kernel_smoke"]["pointer_alias"]:
            raise RuntimeError("fromDevicePointer did not preserve the device pointer")
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 - the report must capture import/runtime failures
        report["failure"] = f"{type(error).__name__}: {error}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
