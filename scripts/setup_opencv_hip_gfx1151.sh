#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
opencv_dir="$repo_dir/third_party/opencv-hip"
contrib_dir="$repo_dir/third_party/opencv_contrib-hip"
build_dir="$repo_dir/.build/opencv5-gfx1151"
install_dir="$repo_dir/.local/opencv5-gfx1151"
opencv_commit="e0387086b2103c23b3b25952b5fb700ca2e42132"
contrib_commit="467cbc6f99aa82ebda39a2e94d6125557bd84d0b"
rocm_path="${ROCM_PATH:-/opt/rocm}"
build_jobs="${BUILD_JOBS:-8}"

ensure_checkout() {
  local source_dir="$1"
  local repository="$2"
  local branch="$3"
  local commit="$4"

  if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none --single-branch --branch "$branch" "$repository" "$source_dir"
  fi
  if [[ -n "$(git -C "$source_dir" status --short)" ]]; then
    echo "Refusing to modify dirty third-party checkout: $source_dir" >&2
    exit 1
  fi
  if ! git -C "$source_dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch --depth 1 origin "$commit"
  fi
  git -C "$source_dir" checkout --detach "$commit"
  test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
}

command -v uv >/dev/null
command -v cmake >/dev/null
command -v ninja >/dev/null
test -x "$repo_dir/.venv/bin/python"
test -d "$rocm_path"

mkdir -p "$repo_dir/third_party" "$build_dir" "$install_dir"
ensure_checkout \
  "$opencv_dir" \
  https://github.com/zhangnju/opencv.git \
  5.x-hip \
  "$opencv_commit"
ensure_checkout \
  "$contrib_dir" \
  https://github.com/zhangnju/opencv_contrib.git \
  5.x-hip-zerocopy \
  "$contrib_commit"

python_executable="$repo_dir/.venv/bin/python"
python_include="$($python_executable -c 'import sysconfig; print(sysconfig.get_path("include"))')"
python_library="$($python_executable -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") + "/" + sysconfig.get_config_var("LDLIBRARY"))')"
python_numpy_include="$($python_executable -c 'import numpy; print(numpy.get_include())')"
python_install="$install_dir/python"

cmake -S "$opencv_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_PREFIX_PATH="$rocm_path" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_INSTALL_RPATH="$install_dir/lib;$rocm_path/lib" \
  -DOPENCV_EXTRA_MODULES_PATH="$contrib_dir/modules" \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DBUILD_LIST=core,imgproc,python3,cudev,cudaarithm,cudaimgproc,cudawarping \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python3=ON \
  -DINSTALL_C_EXAMPLES=OFF \
  -DINSTALL_PYTHON_EXAMPLES=OFF \
  -DWITH_HIP=ON \
  -DWITH_CUDA=OFF \
  -DWITH_ROCDECODE=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GTK=OFF \
  -DWITH_V4L=OFF \
  -DWITH_OPENCL=OFF \
  -DROCM_PATH="$rocm_path" \
  -Dhip_DIR="$rocm_path/lib/cmake/hip" \
  -Dhipblas_DIR="$rocm_path/lib/cmake/hipblas" \
  -Dhipfft_DIR="$rocm_path/lib/cmake/hipfft" \
  -DPYTHON3_EXECUTABLE="$python_executable" \
  -DPYTHON3_INCLUDE_DIR="$python_include" \
  -DPYTHON3_LIBRARY="$python_library" \
  -DPYTHON3_NUMPY_INCLUDE_DIRS="$python_numpy_include" \
  -DOPENCV_PYTHON3_INSTALL_PATH="$python_install"

cmake --build "$build_dir" --parallel "$build_jobs"
cmake --install "$build_dir"

mkdir -p "$repo_dir/native-lock"
cp "$build_dir/CMakeCache.txt" "$repo_dir/native-lock/opencv5-hip-CMakeCache.txt"
git -C "$opencv_dir" rev-parse HEAD > "$repo_dir/native-lock/opencv5-hip.commit"
git -C "$contrib_dir" rev-parse HEAD > "$repo_dir/native-lock/opencv_contrib-hip.commit"

# shellcheck source=env_zerocopy_gfx1151.sh
source "$repo_dir/scripts/env_zerocopy_gfx1151.sh"
exec uv run --frozen python "$repo_dir/scripts/check_opencv_hip.py" \
  --expected-prefix "$install_dir" \
  --output "$repo_dir/output/realtime/opencv5-hip-check.json"
