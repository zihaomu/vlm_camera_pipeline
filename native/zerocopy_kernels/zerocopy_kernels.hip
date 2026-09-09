#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

thread_local char last_error[512] = "ok";

void set_error(const char *operation, hipError_t status) {
    std::snprintf(
        last_error,
        sizeof(last_error),
        "%s: %s (%s)",
        operation,
        hipGetErrorName(status),
        hipGetErrorString(status));
}

__device__ float bilinear_channel(
    const std::uint8_t *source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int destination_x,
    int destination_y,
    int channel) {
    const float source_x =
        (static_cast<float>(destination_x) + 0.5F) * source_width / resized_width - 0.5F;
    const float source_y =
        (static_cast<float>(destination_y) + 0.5F) * source_height / resized_height - 0.5F;
    const int x0_unclamped = static_cast<int>(floorf(source_x));
    const int y0_unclamped = static_cast<int>(floorf(source_y));
    const int x0 = max(0, min(source_width - 1, x0_unclamped));
    const int y0 = max(0, min(source_height - 1, y0_unclamped));
    const int x1 = max(0, min(source_width - 1, x0_unclamped + 1));
    const int y1 = max(0, min(source_height - 1, y0_unclamped + 1));
    const float wx = source_x - floorf(source_x);
    const float wy = source_y - floorf(source_y);
    const auto *row0 = source + static_cast<std::size_t>(y0) * source_pitch;
    const auto *row1 = source + static_cast<std::size_t>(y1) * source_pitch;
    const float p00 = row0[x0 * 3 + channel];
    const float p01 = row0[x1 * 3 + channel];
    const float p10 = row1[x0 * 3 + channel];
    const float p11 = row1[x1 * 3 + channel];
    const float top = fmaf(wx, p01 - p00, p00);
    const float bottom = fmaf(wx, p11 - p10, p10);
    return fmaf(wy, bottom - top, top);
}

__device__ float bilinear_channel_align_corners(
    const std::uint8_t *source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int destination_x,
    int destination_y,
    int channel) {
    const float x_ratio = resized_width > 1
        ? static_cast<float>(source_width - 1) / static_cast<float>(resized_width - 1)
        : 0.0F;
    const float y_ratio = resized_height > 1
        ? static_cast<float>(source_height - 1) / static_cast<float>(resized_height - 1)
        : 0.0F;
    const float source_x = static_cast<float>(destination_x) * x_ratio;
    const float source_y = static_cast<float>(destination_y) * y_ratio;
    const int x0 = min(static_cast<int>(source_x), source_width - 1);
    const int y0 = min(static_cast<int>(source_y), source_height - 1);
    const int x1 = min(x0 + 1, source_width - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float wx = source_x - static_cast<float>(x0);
    const float wy = source_y - static_cast<float>(y0);
    const auto *row0 = source + static_cast<std::size_t>(y0) * source_pitch;
    const auto *row1 = source + static_cast<std::size_t>(y1) * source_pitch;
    const float p00 = row0[x0 * 3 + channel];
    const float p10 = row0[x1 * 3 + channel];
    const float p01 = row1[x0 * 3 + channel];
    const float p11 = row1[x1 * 3 + channel];
    const float top = p00 + (p10 - p00) * wx;
    const float bottom = p01 + (p11 - p01) * wx;
    return top + (bottom - top) * wy;
}

__global__ void qwen3_vl_preprocess_rgb8_f32_kernel(
    const std::uint8_t *source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    int source_is_bgr,
    float *destination,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= destination_width || y >= destination_height) {
        return;
    }

    const std::size_t plane_size =
        static_cast<std::size_t>(destination_width) * destination_height;
    const std::size_t output_index = static_cast<std::size_t>(y) * destination_width + x;
    const int resized_x = x - pad_left;
    const int resized_y = y - pad_top;
    if (resized_x < 0 || resized_y < 0 || resized_x >= resized_width || resized_y >= resized_height) {
        destination[output_index] = -1.0F;
        destination[plane_size + output_index] = -1.0F;
        destination[2 * plane_size + output_index] = -1.0F;
        return;
    }

    const int channels[3] = {source_is_bgr ? 2 : 0, 1, source_is_bgr ? 0 : 2};
    for (int output_channel = 0; output_channel < 3; ++output_channel) {
        const float interpolated = bilinear_channel_align_corners(
            source,
            source_pitch,
            source_width,
            source_height,
            resized_width,
            resized_height,
            resized_x,
            resized_y,
            channels[output_channel]);
        // llama.cpp resizes into an RGB8 image before normalization. Preserve its
        // float-to-uint8 truncation exactly, then write the final planar tensor.
        const auto pixel = static_cast<std::uint8_t>(interpolated);
        destination[static_cast<std::size_t>(output_channel) * plane_size + output_index] =
            (static_cast<float>(pixel) - 127.5F) / 127.5F;
    }
}

bool qwen3_vl_smart_resize_impl(
    int source_width,
    int source_height,
    int image_max_tokens,
    int &destination_width,
    int &destination_height) {
    constexpr int align = 16 * 2;
    constexpr int min_pixels = 8 * align * align;
    if (source_width <= 0 || source_height <= 0 || source_width > 16384 ||
        source_height > 16384 || image_max_tokens < 8 || image_max_tokens > 4096) {
        return false;
    }
    const int max_pixels = image_max_tokens * align * align;
    auto round_factor = [](float value) {
        return static_cast<int>(std::round(value / align)) * align;
    };
    auto ceil_factor = [](float value) {
        return static_cast<int>(std::ceil(value / align)) * align;
    };
    auto floor_factor = [](float value) {
        return static_cast<int>(std::floor(value / align)) * align;
    };
    int height = std::max(align, round_factor(static_cast<float>(source_height)));
    int width = std::max(align, round_factor(static_cast<float>(source_width)));
    if (static_cast<std::int64_t>(height) * width > max_pixels) {
        const float beta = std::sqrt(
            static_cast<float>(source_height) * source_width / max_pixels);
        height = std::max(align, floor_factor(source_height / beta));
        width = std::max(align, floor_factor(source_width / beta));
    } else if (static_cast<std::int64_t>(height) * width < min_pixels) {
        const float beta = std::sqrt(
            static_cast<float>(min_pixels) /
            (static_cast<float>(source_height) * source_width));
        height = ceil_factor(source_height * beta);
        width = ceil_factor(source_width * beta);
    }
    destination_width = width;
    destination_height = height;
    return true;
}

__global__ void letterbox_rgb8_to_bchw_f32_kernel(
    const std::uint8_t *source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    int source_is_bgr,
    float *destination,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= destination_width || y >= destination_height) {
        return;
    }

    const std::size_t plane_size =
        static_cast<std::size_t>(destination_width) * destination_height;
    const std::size_t output_index = static_cast<std::size_t>(y) * destination_width + x;
    const int resized_x = x - pad_left;
    const int resized_y = y - pad_top;
    if (resized_x < 0 || resized_y < 0 || resized_x >= resized_width || resized_y >= resized_height) {
        constexpr float padding = 114.0F / 255.0F;
        destination[output_index] = padding;
        destination[plane_size + output_index] = padding;
        destination[2 * plane_size + output_index] = padding;
        return;
    }

    const int red_channel = source_is_bgr ? 2 : 0;
    const int blue_channel = source_is_bgr ? 0 : 2;
    destination[output_index] = bilinear_channel(
        source,
        source_pitch,
        source_width,
        source_height,
        resized_width,
        resized_height,
        resized_x,
        resized_y,
        red_channel) / 255.0F;
    destination[plane_size + output_index] = bilinear_channel(
        source,
        source_pitch,
        source_width,
        source_height,
        resized_width,
        resized_height,
        resized_x,
        resized_y,
        1) / 255.0F;
    destination[2 * plane_size + output_index] = bilinear_channel(
        source,
        source_pitch,
        source_width,
        source_height,
        resized_width,
        resized_height,
        resized_x,
        resized_y,
        blue_channel) / 255.0F;
}

__global__ void unletterbox_detections_kernel(
    float *detections,
    int detection_count,
    float confidence_threshold,
    int source_width,
    int source_height,
    float gain,
    int pad_left,
    int pad_top) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= detection_count) {
        return;
    }
    float *detection = detections + static_cast<std::size_t>(index) * 6;
    float x1 = (detection[0] - pad_left) / gain;
    float y1 = (detection[1] - pad_top) / gain;
    float x2 = (detection[2] - pad_left) / gain;
    float y2 = (detection[3] - pad_top) / gain;
    x1 = fminf(fmaxf(x1, 0.0F), static_cast<float>(source_width));
    y1 = fminf(fmaxf(y1, 0.0F), static_cast<float>(source_height));
    x2 = fminf(fmaxf(x2, 0.0F), static_cast<float>(source_width));
    y2 = fminf(fmaxf(y2, 0.0F), static_cast<float>(source_height));
    const bool valid = detection[4] > confidence_threshold && x2 > x1 && y2 > y1;
    detection[0] = x1;
    detection[1] = y1;
    detection[2] = x2;
    detection[3] = y2;
    detection[4] = valid ? detection[4] : 0.0F;
}

}  // namespace

extern "C" const char *vlm_camera_zerocopy_last_error() {
    return last_error;
}

extern "C" int vlm_camera_qwen3_vl_smart_resize(
    int source_width,
    int source_height,
    int image_max_tokens,
    int *destination_width_out,
    int *destination_height_out) {
    if (destination_width_out == nullptr || destination_height_out == nullptr ||
        !qwen3_vl_smart_resize_impl(
            source_width,
            source_height,
            image_max_tokens,
            *destination_width_out,
            *destination_height_out)) {
        std::snprintf(last_error, sizeof(last_error), "invalid Qwen3-VL smart-resize arguments");
        return 1;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_qwen3_vl_preprocess_rgb8_f32(
    const void *source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    int source_is_bgr,
    void *destination,
    int destination_width,
    int destination_height,
    int image_max_tokens,
    void *source_ready_event_pointer,
    void *stream_pointer,
    int *resized_width_out,
    int *resized_height_out,
    int *pad_left_out,
    int *pad_top_out) {
    int expected_width = 0;
    int expected_height = 0;
    if (source == nullptr || destination == nullptr ||
        source_pitch < static_cast<std::size_t>(source_width) * 3 ||
        resized_width_out == nullptr || resized_height_out == nullptr ||
        pad_left_out == nullptr || pad_top_out == nullptr ||
        !qwen3_vl_smart_resize_impl(
            source_width,
            source_height,
            image_max_tokens,
            expected_width,
            expected_height) ||
        destination_width != expected_width || destination_height != expected_height) {
        std::snprintf(last_error, sizeof(last_error), "invalid Qwen3-VL preprocess arguments");
        return 1;
    }

    const float scale = std::min(
        static_cast<float>(destination_width) / source_width,
        static_cast<float>(destination_height) / source_height);
    const int resized_width = std::min(
        static_cast<int>(std::ceil(source_width * scale)), destination_width);
    const int resized_height = std::min(
        static_cast<int>(std::ceil(source_height * scale)), destination_height);
    const int pad_left = (destination_width - resized_width) / 2;
    const int pad_top = (destination_height - resized_height) / 2;
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (source_ready_event_pointer != nullptr) {
        const hipError_t wait_status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(source_ready_event_pointer), 0);
        if (wait_status != hipSuccess) {
            set_error("Qwen3-VL source event wait", wait_status);
            return 2;
        }
    }
    const dim3 block(16, 16);
    const dim3 grid(
        (destination_width + block.x - 1) / block.x,
        (destination_height + block.y - 1) / block.y);
    hipLaunchKernelGGL(
        qwen3_vl_preprocess_rgb8_f32_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const std::uint8_t *>(source),
        source_pitch,
        source_width,
        source_height,
        source_is_bgr,
        static_cast<float *>(destination),
        destination_width,
        destination_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("Qwen3-VL preprocess kernel launch", status);
        return 3;
    }
    *resized_width_out = resized_width;
    *resized_height_out = resized_height;
    *pad_left_out = pad_left;
    *pad_top_out = pad_top;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_yolo_letterbox_rgb8_f32(
    const void *source,
    std::size_t source_pitch,
    int source_width,
    int source_height,
    int source_is_bgr,
    void *destination,
    int destination_width,
    int destination_height,
    void *source_ready_event_pointer,
    void *stream_pointer,
    float *gain_out,
    int *resized_width_out,
    int *resized_height_out,
    int *pad_left_out,
    int *pad_top_out) {
    if (source == nullptr || destination == nullptr || source_width <= 0 || source_height <= 0 ||
        destination_width <= 0 || destination_height <= 0 ||
        source_pitch < static_cast<std::size_t>(source_width) * 3 || gain_out == nullptr ||
        resized_width_out == nullptr || resized_height_out == nullptr || pad_left_out == nullptr ||
        pad_top_out == nullptr) {
        std::snprintf(last_error, sizeof(last_error), "invalid letterbox arguments");
        return 1;
    }
    const float gain = std::min(
        static_cast<float>(destination_height) / source_height,
        static_cast<float>(destination_width) / source_width);
    const int resized_width = static_cast<int>(std::round(source_width * gain));
    const int resized_height = static_cast<int>(std::round(source_height * gain));
    const int pad_left = static_cast<int>(std::round((destination_width - resized_width) / 2.0F - 0.1F));
    const int pad_top = static_cast<int>(std::round((destination_height - resized_height) / 2.0F - 0.1F));
    const dim3 block(16, 16);
    const dim3 grid(
        (destination_width + block.x - 1) / block.x,
        (destination_height + block.y - 1) / block.y);
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (source_ready_event_pointer != nullptr) {
        const hipError_t wait_status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(source_ready_event_pointer), 0);
        if (wait_status != hipSuccess) {
            set_error("YOLO source event wait", wait_status);
            return 2;
        }
    }
    hipLaunchKernelGGL(
        letterbox_rgb8_to_bchw_f32_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const std::uint8_t *>(source),
        source_pitch,
        source_width,
        source_height,
        source_is_bgr,
        static_cast<float *>(destination),
        destination_width,
        destination_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("letterbox kernel launch", status);
        return 3;
    }
    *gain_out = gain;
    *resized_width_out = resized_width;
    *resized_height_out = resized_height;
    *pad_left_out = pad_left;
    *pad_top_out = pad_top;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_yolo_unletterbox_f32(
    void *detections,
    int detection_count,
    float confidence_threshold,
    int source_width,
    int source_height,
    float gain,
    int pad_left,
    int pad_top,
    void *stream_pointer) {
    if (detections == nullptr || detection_count <= 0 || source_width <= 0 || source_height <= 0 ||
        gain <= 0.0F || confidence_threshold < 0.0F || confidence_threshold > 1.0F) {
        std::snprintf(last_error, sizeof(last_error), "invalid detection arguments");
        return 1;
    }
    constexpr int block_size = 256;
    const int grid_size = (detection_count + block_size - 1) / block_size;
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    hipLaunchKernelGGL(
        unletterbox_detections_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        stream,
        static_cast<float *>(detections),
        detection_count,
        confidence_threshold,
        source_width,
        source_height,
        gain,
        pad_left,
        pad_top);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("unletterbox kernel launch", status);
        return 2;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}
