#!/usr/bin/env bash

zerocopy_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
zerocopy_opencv_prefix="$zerocopy_repo_dir/.local/opencv5-gfx1151"
zerocopy_rocm_path="${ROCM_PATH:-/opt/rocm}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$zerocopy_opencv_prefix/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$zerocopy_opencv_prefix/lib:$zerocopy_rocm_path/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$zerocopy_opencv_prefix/bin:$zerocopy_rocm_path/bin:$PATH"
export OPENCV_INSTALL="$zerocopy_opencv_prefix"

unset zerocopy_repo_dir zerocopy_opencv_prefix zerocopy_rocm_path
