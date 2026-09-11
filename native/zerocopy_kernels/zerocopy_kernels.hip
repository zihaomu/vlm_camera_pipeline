#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

thread_local char last_error[512] = "ok";

constexpr int kHybridMaxHints = 8;
constexpr int kHybridBoxThickness = 3;
constexpr int kHybridBadgeWidth = 28;
constexpr int kHybridBadgeHeight = 18;
constexpr int kHybridGlyphWidth = 5;
constexpr int kHybridGlyphHeight = 7;
constexpr int kHybridGlyphScale = 2;
constexpr int kWebDetectionClassCount = 80;
constexpr int kWebDetectionLabelWidth = 224;
constexpr int kWebDetectionLabelHeight = 36;
constexpr int kWebDetectionBoxThickness = 4;

struct HybridHintRecord {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    std::int32_t class_id;
    std::int32_t source_index;
    std::int32_t rank;
};

struct HybridHintBuffer {
    std::uint32_t count;
    HybridHintRecord records[kHybridMaxHints];
};

static_assert(sizeof(HybridHintRecord) == 32, "HybridHintRecord ABI must remain fixed");
static_assert(sizeof(HybridHintBuffer) == 260, "HybridHintBuffer ABI must remain fixed");

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

__device__ bool hybrid_hint_precedes(
    float confidence,
    int source_index,
    const HybridHintRecord &other) {
    return confidence > other.confidence ||
        (confidence == other.confidence && source_index < other.source_index);
}

__global__ void hybrid_compact_hints_kernel(
    const float *detections,
    int detection_count,
    float confidence_threshold,
    int max_hints,
    HybridHintBuffer *output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    output->count = 0;
    for (int source_index = 0; source_index < detection_count; ++source_index) {
        const float *detection = detections + static_cast<std::size_t>(source_index) * 6;
        const float x1 = detection[0];
        const float y1 = detection[1];
        const float x2 = detection[2];
        const float y2 = detection[3];
        const float confidence = detection[4];
        const float class_value = detection[5];
        if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2) ||
            !isfinite(confidence) || !isfinite(class_value) ||
            confidence <= confidence_threshold || x2 <= x1 || y2 <= y1) {
            continue;
        }
        const int class_id = static_cast<int>(roundf(class_value));
        if (class_id < 0 || class_id >= 80) {
            continue;
        }

        const int count = static_cast<int>(output->count);
        int insertion = count;
        for (int index = 0; index < count; ++index) {
            if (hybrid_hint_precedes(confidence, source_index, output->records[index])) {
                insertion = index;
                break;
            }
        }
        if (insertion >= max_hints) {
            continue;
        }
        const int shifted_last = min(count, max_hints - 1);
        for (int index = shifted_last; index > insertion; --index) {
            output->records[index] = output->records[index - 1];
        }
        HybridHintRecord &record = output->records[insertion];
        record.x1 = x1;
        record.y1 = y1;
        record.x2 = x2;
        record.y2 = y2;
        record.confidence = confidence;
        record.class_id = class_id;
        record.source_index = source_index;
        record.rank = insertion + 1;
        output->count = static_cast<std::uint32_t>(min(count + 1, max_hints));
        for (int index = insertion + 1; index < static_cast<int>(output->count); ++index) {
            output->records[index].rank = index + 1;
        }
    }
}

struct HybridColor {
    std::uint8_t red;
    std::uint8_t green;
    std::uint8_t blue;
};

__device__ HybridColor hybrid_rank_color(int rank) {
    switch (rank) {
        case 1: return {255, 56, 56};
        case 2: return {255, 157, 151};
        case 3: return {255, 178, 29};
        case 4: return {72, 249, 10};
        case 5: return {0, 212, 187};
        case 6: return {0, 194, 255};
        case 7: return {100, 115, 255};
        default: return {255, 55, 199};
    }
}

__device__ std::uint8_t hybrid_glyph_row(char glyph, int row) {
    constexpr std::uint8_t hash[kHybridGlyphHeight] = {
        0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b01010};
    constexpr std::uint8_t one[kHybridGlyphHeight] = {
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110};
    constexpr std::uint8_t two[kHybridGlyphHeight] = {
        0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111};
    constexpr std::uint8_t three[kHybridGlyphHeight] = {
        0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110};
    constexpr std::uint8_t four[kHybridGlyphHeight] = {
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010};
    constexpr std::uint8_t five[kHybridGlyphHeight] = {
        0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110};
    constexpr std::uint8_t six[kHybridGlyphHeight] = {
        0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110};
    constexpr std::uint8_t seven[kHybridGlyphHeight] = {
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000};
    constexpr std::uint8_t eight[kHybridGlyphHeight] = {
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110};
    if (row < 0 || row >= kHybridGlyphHeight) {
        return 0;
    }
    switch (glyph) {
        case '#': return hash[row];
        case '1': return one[row];
        case '2': return two[row];
        case '3': return three[row];
        case '4': return four[row];
        case '5': return five[row];
        case '6': return six[row];
        case '7': return seven[row];
        case '8': return eight[row];
        default: return 0;
    }
}

__device__ bool hybrid_glyph_pixel(char glyph, int x, int y) {
    if (x < 0 || y < 0 || x >= kHybridGlyphWidth * kHybridGlyphScale ||
        y >= kHybridGlyphHeight * kHybridGlyphScale) {
        return false;
    }
    const int glyph_x = x / kHybridGlyphScale;
    const int glyph_y = y / kHybridGlyphScale;
    const std::uint8_t row = hybrid_glyph_row(glyph, glyph_y);
    return (row & (1U << (kHybridGlyphWidth - 1 - glyph_x))) != 0;
}

__device__ void hybrid_write_pixel(
    float *image,
    int width,
    int height,
    int x,
    int y,
    HybridColor color) {
    if (x < 0 || y < 0 || x >= width || y >= height) {
        return;
    }
    const std::size_t plane_size = static_cast<std::size_t>(width) * height;
    const std::size_t offset = static_cast<std::size_t>(y) * width + x;
    image[offset] = (static_cast<float>(color.red) - 127.5F) / 127.5F;
    image[plane_size + offset] = (static_cast<float>(color.green) - 127.5F) / 127.5F;
    image[2 * plane_size + offset] = (static_cast<float>(color.blue) - 127.5F) / 127.5F;
}

__device__ int hybrid_map_coordinate(
    float source_coordinate,
    int source_extent,
    int resized_extent,
    int pad_before) {
    if (source_extent <= 1 || resized_extent <= 1) {
        return pad_before;
    }
    return pad_before + static_cast<int>(roundf(
        source_coordinate * static_cast<float>(resized_extent - 1) /
        static_cast<float>(source_extent - 1)));
}

__global__ void hybrid_numbered_boxes_kernel(
    float *image,
    int image_width,
    int image_height,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    const HybridHintBuffer *hints) {
    if (blockIdx.x != 0) {
        return;
    }
    const int thread_index = static_cast<int>(threadIdx.x);
    const int content_left = pad_left;
    const int content_top = pad_top;
    const int content_right = pad_left + resized_width - 1;
    const int content_bottom = pad_top + resized_height - 1;
    const int count = min(static_cast<int>(hints->count), kHybridMaxHints);
    for (int hint_index = 0; hint_index < count; ++hint_index) {
        const HybridHintRecord &hint = hints->records[hint_index];
        const HybridColor color = hybrid_rank_color(hint.rank);
        const int x1 = max(content_left, min(content_right, hybrid_map_coordinate(
            hint.x1, source_width, resized_width, pad_left)));
        const int y1 = max(content_top, min(content_bottom, hybrid_map_coordinate(
            hint.y1, source_height, resized_height, pad_top)));
        const int x2 = max(content_left, min(content_right, hybrid_map_coordinate(
            hint.x2, source_width, resized_width, pad_left)));
        const int y2 = max(content_top, min(content_bottom, hybrid_map_coordinate(
            hint.y2, source_height, resized_height, pad_top)));
        if (x2 <= x1 || y2 <= y1) {
            __syncthreads();
            continue;
        }
        for (int offset = thread_index; offset <= x2 - x1; offset += blockDim.x) {
            for (int thickness = 0; thickness < kHybridBoxThickness; ++thickness) {
                hybrid_write_pixel(image, image_width, image_height, x1 + offset,
                    min(y2, y1 + thickness), color);
                hybrid_write_pixel(image, image_width, image_height, x1 + offset,
                    max(y1, y2 - thickness), color);
            }
        }
        for (int offset = thread_index; offset <= y2 - y1; offset += blockDim.x) {
            for (int thickness = 0; thickness < kHybridBoxThickness; ++thickness) {
                hybrid_write_pixel(image, image_width, image_height,
                    min(x2, x1 + thickness), y1 + offset, color);
                hybrid_write_pixel(image, image_width, image_height,
                    max(x1, x2 - thickness), y1 + offset, color);
            }
        }
        __syncthreads();

        const int badge_x = min(x1, max(content_left, content_right - kHybridBadgeWidth + 1));
        const int badge_y = y1 - kHybridBadgeHeight >= content_top
            ? y1 - kHybridBadgeHeight
            : y1;
        for (int pixel = thread_index;
             pixel < kHybridBadgeWidth * kHybridBadgeHeight;
             pixel += blockDim.x) {
            const int local_x = pixel % kHybridBadgeWidth;
            const int local_y = pixel / kHybridBadgeWidth;
            const bool border = local_x < 2 || local_y < 2 ||
                local_x >= kHybridBadgeWidth - 2 || local_y >= kHybridBadgeHeight - 2;
            HybridColor pixel_color = border ? color : HybridColor{16, 16, 16};
            const int glyph_y = local_y - 2;
            const bool hash_on = hybrid_glyph_pixel('#', local_x - 3, glyph_y);
            const bool digit_on = hybrid_glyph_pixel(
                static_cast<char>('0' + hint.rank), local_x - 16, glyph_y);
            if (hash_on || digit_on) {
                pixel_color = {255, 255, 255};
            }
            hybrid_write_pixel(
                image, image_width, image_height, badge_x + local_x, badge_y + local_y,
                pixel_color);
        }
        __syncthreads();
    }
}

struct WebRgbColor {
    std::uint8_t red;
    std::uint8_t green;
    std::uint8_t blue;
};

struct WebYuvColor {
    std::uint8_t y;
    std::uint8_t u;
    std::uint8_t v;
};

__device__ WebRgbColor web_detection_color(int class_id) {
    switch (class_id % 20) {
        case 0: return {255, 56, 56};
        case 1: return {255, 157, 151};
        case 2: return {255, 112, 31};
        case 3: return {255, 178, 29};
        case 4: return {207, 210, 49};
        case 5: return {72, 249, 10};
        case 6: return {146, 204, 23};
        case 7: return {61, 219, 134};
        case 8: return {26, 147, 52};
        case 9: return {0, 212, 187};
        case 10: return {44, 153, 168};
        case 11: return {0, 194, 255};
        case 12: return {52, 69, 147};
        case 13: return {100, 115, 255};
        case 14: return {0, 24, 236};
        case 15: return {132, 56, 255};
        case 16: return {82, 0, 133};
        case 17: return {203, 56, 255};
        case 18: return {255, 149, 200};
        default: return {255, 55, 199};
    }
}

__device__ WebYuvColor web_rgb_to_yuv(WebRgbColor color) {
    const int red = color.red;
    const int green = color.green;
    const int blue = color.blue;
    return {
        static_cast<std::uint8_t>(max(0, min(255,
            ((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16))),
        static_cast<std::uint8_t>(max(0, min(255,
            ((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128))),
        static_cast<std::uint8_t>(max(0, min(255,
            ((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128))),
    };
}

__device__ WebRgbColor web_label_pixel(WebRgbColor color, unsigned int alpha) {
    WebRgbColor background = {
        static_cast<std::uint8_t>(static_cast<unsigned int>(color.red) * 3 / 5),
        static_cast<std::uint8_t>(static_cast<unsigned int>(color.green) * 3 / 5),
        static_cast<std::uint8_t>(static_cast<unsigned int>(color.blue) * 3 / 5),
    };
    return {
        static_cast<std::uint8_t>(
            (static_cast<unsigned int>(background.red) * (255 - alpha) + 255 * alpha) / 255),
        static_cast<std::uint8_t>(
            (static_cast<unsigned int>(background.green) * (255 - alpha) + 255 * alpha) / 255),
        static_cast<std::uint8_t>(
            (static_cast<unsigned int>(background.blue) * (255 - alpha) + 255 * alpha) / 255),
    };
}

// One thread owns a complete YUYV pixel pair. This avoids a race between the
// two luma samples that share U/V and makes every overlay write a single aligned
// 32-bit store into the encoder-facing DMA-BUF.
__device__ void web_write_yuyv_pair(
    std::uint8_t *destination,
    std::size_t pitch,
    int pair_x,
    int y,
    WebRgbColor first,
    WebRgbColor second) {
    const WebYuvColor first_yuv = web_rgb_to_yuv(first);
    const WebYuvColor second_yuv = web_rgb_to_yuv(second);
    const std::uint8_t u = static_cast<std::uint8_t>(
        (static_cast<unsigned int>(first_yuv.u) + second_yuv.u + 1) / 2);
    const std::uint8_t v = static_cast<std::uint8_t>(
        (static_cast<unsigned int>(first_yuv.v) + second_yuv.v + 1) / 2);
    const std::uint32_t packed = static_cast<std::uint32_t>(first_yuv.y) |
        (static_cast<std::uint32_t>(u) << 8) |
        (static_cast<std::uint32_t>(second_yuv.y) << 16) |
        (static_cast<std::uint32_t>(v) << 24);
    auto *row = destination + static_cast<std::size_t>(y) * pitch;
    reinterpret_cast<std::uint32_t *>(row)[pair_x] = packed;
}

__global__ void web_yuyv_detection_overlay_kernel(
    std::uint8_t *destination,
    std::size_t pitch,
    int width,
    int height,
    const float *detections,
    int detection_count,
    float confidence_threshold,
    const std::uint8_t *class_label_alpha,
    const int *class_label_widths) {
    const int detection_index = static_cast<int>(blockIdx.x);
    const int thread_index = static_cast<int>(threadIdx.x);
    if (detection_index >= detection_count) {
        return;
    }
    const float *detection = detections + static_cast<std::size_t>(detection_index) * 6;
    if (!isfinite(detection[0]) || !isfinite(detection[1]) ||
        !isfinite(detection[2]) || !isfinite(detection[3]) ||
        !isfinite(detection[4]) || !isfinite(detection[5]) ||
        detection[4] <= confidence_threshold) {
        return;
    }
    const int x1 = max(0, min(width - 1, static_cast<int>(roundf(detection[0]))));
    const int y1 = max(0, min(height - 1, static_cast<int>(roundf(detection[1]))));
    const int x2 = max(0, min(width - 1, static_cast<int>(roundf(detection[2]))));
    const int y2 = max(0, min(height - 1, static_cast<int>(roundf(detection[3]))));
    const int class_id = static_cast<int>(roundf(detection[5]));
    if (x2 <= x1 || y2 <= y1 || class_id < 0 || class_id >= kWebDetectionClassCount) {
        return;
    }
    const WebRgbColor color = web_detection_color(class_id);
    const int first_pair = x1 / 2;
    const int last_pair = x2 / 2;
    for (int pair_x = first_pair + thread_index;
         pair_x <= last_pair;
         pair_x += static_cast<int>(blockDim.x)) {
        for (int thickness = 0; thickness < kWebDetectionBoxThickness; ++thickness) {
            web_write_yuyv_pair(
                destination, pitch, pair_x, min(y2, y1 + thickness), color, color);
            web_write_yuyv_pair(
                destination, pitch, pair_x, max(y1, y2 - thickness), color, color);
        }
    }
    for (int y = y1 + thread_index; y <= y2; y += static_cast<int>(blockDim.x)) {
        const int left_last_pair = min(x2, x1 + kWebDetectionBoxThickness - 1) / 2;
        const int right_first_pair = max(x1, x2 - kWebDetectionBoxThickness + 1) / 2;
        for (int pair_x = first_pair; pair_x <= left_last_pair; ++pair_x) {
            web_write_yuyv_pair(destination, pitch, pair_x, y, color, color);
        }
        for (int pair_x = right_first_pair; pair_x <= last_pair; ++pair_x) {
            web_write_yuyv_pair(destination, pitch, pair_x, y, color, color);
        }
    }
    __syncthreads();

    const int label_width = max(
        1, min(kWebDetectionLabelWidth, class_label_widths[class_id]));
    // Align labels to a packed-pixel boundary so one thread always owns both
    // samples and never modifies a neighbour through shared chroma.
    const int unclamped_label_x = min(x1, max(0, width - label_width));
    const int label_x = unclamped_label_x & ~1;
    const int label_y = y1 >= kWebDetectionLabelHeight
        ? y1 - kWebDetectionLabelHeight
        : y1;
    const int label_pairs = (label_width + 1) / 2;
    const std::size_t atlas_offset = static_cast<std::size_t>(class_id) *
        kWebDetectionLabelWidth * kWebDetectionLabelHeight;
    for (int pair_pixel = thread_index;
         pair_pixel < label_pairs * kWebDetectionLabelHeight;
         pair_pixel += static_cast<int>(blockDim.x)) {
        const int local_pair_x = pair_pixel % label_pairs;
        const int local_y = pair_pixel / label_pairs;
        const int local_x = local_pair_x * 2;
        const unsigned int first_alpha = class_label_alpha[
            atlas_offset + static_cast<std::size_t>(local_y) * kWebDetectionLabelWidth + local_x];
        const unsigned int second_alpha = local_x + 1 < label_width
            ? class_label_alpha[
                atlas_offset + static_cast<std::size_t>(local_y) *
                    kWebDetectionLabelWidth + local_x + 1]
            : 0;
        web_write_yuyv_pair(
            destination,
            pitch,
            label_x / 2 + local_pair_x,
            label_y + local_y,
            web_label_pixel(color, first_alpha),
            web_label_pixel(color, second_alpha));
    }
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

extern "C" int vlm_camera_hybrid_max_hints() {
    return kHybridMaxHints;
}

extern "C" std::size_t vlm_camera_hybrid_hint_record_size() {
    return sizeof(HybridHintRecord);
}

extern "C" std::size_t vlm_camera_hybrid_hint_buffer_size() {
    return sizeof(HybridHintBuffer);
}

extern "C" int vlm_camera_hybrid_compact_f32(
    const void *detections_f32,
    int detection_count,
    float confidence_threshold,
    int max_hints,
    void *hint_buffer,
    void *detections_ready_event_pointer,
    void *stream_pointer) {
    if (detections_f32 == nullptr || hint_buffer == nullptr || detection_count <= 0 ||
        confidence_threshold < 0.0F || confidence_threshold > 1.0F ||
        max_hints <= 0 || max_hints > kHybridMaxHints) {
        std::snprintf(last_error, sizeof(last_error), "invalid Hybrid compact arguments");
        return 1;
    }
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (detections_ready_event_pointer != nullptr) {
        const hipError_t status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(detections_ready_event_pointer), 0);
        if (status != hipSuccess) {
            set_error("Hybrid compact detection event wait", status);
            return 2;
        }
    }
    hipLaunchKernelGGL(
        hybrid_compact_hints_kernel,
        dim3(1),
        dim3(1),
        0,
        stream,
        static_cast<const float *>(detections_f32),
        detection_count,
        confidence_threshold,
        max_hints,
        static_cast<HybridHintBuffer *>(hint_buffer));
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("Hybrid compact kernel launch", status);
        return 3;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_hybrid_compact_and_overlay_f32(
    void *image_chw_f32,
    int image_width,
    int image_height,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    const void *detections_f32,
    int detection_count,
    float confidence_threshold,
    int max_hints,
    void *hint_buffer,
    void *base_ready_event_pointer,
    void *detections_ready_event_pointer,
    void *stream_pointer) {
    if (image_chw_f32 == nullptr || detections_f32 == nullptr || hint_buffer == nullptr ||
        image_width <= 0 || image_height <= 0 || source_width <= 0 || source_height <= 0 ||
        resized_width <= 0 || resized_height <= 0 || resized_width > image_width ||
        resized_height > image_height || pad_left < 0 || pad_top < 0 ||
        pad_left + resized_width > image_width || pad_top + resized_height > image_height ||
        detection_count <= 0 || confidence_threshold < 0.0F || confidence_threshold > 1.0F ||
        max_hints <= 0 || max_hints > kHybridMaxHints) {
        std::snprintf(last_error, sizeof(last_error), "invalid Hybrid overlay arguments");
        return 1;
    }
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (base_ready_event_pointer != nullptr) {
        const hipError_t status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(base_ready_event_pointer), 0);
        if (status != hipSuccess) {
            set_error("Hybrid base image event wait", status);
            return 2;
        }
    }
    if (detections_ready_event_pointer != nullptr) {
        const hipError_t status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(detections_ready_event_pointer), 0);
        if (status != hipSuccess) {
            set_error("Hybrid detection event wait", status);
            return 3;
        }
    }
    hipLaunchKernelGGL(
        hybrid_compact_hints_kernel,
        dim3(1),
        dim3(1),
        0,
        stream,
        static_cast<const float *>(detections_f32),
        detection_count,
        confidence_threshold,
        max_hints,
        static_cast<HybridHintBuffer *>(hint_buffer));
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("Hybrid hint compaction kernel launch", status);
        return 4;
    }
    hipLaunchKernelGGL(
        hybrid_numbered_boxes_kernel,
        dim3(1),
        dim3(256),
        0,
        stream,
        static_cast<float *>(image_chw_f32),
        image_width,
        image_height,
        source_width,
        source_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        static_cast<const HybridHintBuffer *>(hint_buffer));
    status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("Hybrid numbered-box kernel launch", status);
        return 5;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_web_overlay_class_count() {
    return kWebDetectionClassCount;
}

extern "C" int vlm_camera_web_overlay_label_width() {
    return kWebDetectionLabelWidth;
}

extern "C" int vlm_camera_web_overlay_label_height() {
    return kWebDetectionLabelHeight;
}

extern "C" int vlm_camera_web_overlay_yuyv_f32(
    void *destination_yuyv,
    std::size_t destination_pitch,
    int width,
    int height,
    const void *detections_f32,
    int detection_count,
    float confidence_threshold,
    const void *class_label_alpha,
    const void *class_label_widths,
    void *detections_ready_event_pointer,
    void *stream_pointer) {
    if (destination_yuyv == nullptr || detections_f32 == nullptr ||
        class_label_alpha == nullptr || class_label_widths == nullptr ||
        width <= 0 || height <= 0 || (width & 1) != 0 ||
        destination_pitch < static_cast<std::size_t>(width) * 2 ||
        detection_count <= 0 || detection_count > 4096 ||
        confidence_threshold < 0.0F || confidence_threshold > 1.0F) {
        std::snprintf(last_error, sizeof(last_error), "invalid Web YUYV overlay arguments");
        return 1;
    }
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (detections_ready_event_pointer != nullptr) {
        const hipError_t wait_status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(detections_ready_event_pointer), 0);
        if (wait_status != hipSuccess) {
            set_error("Web YUYV overlay detection event wait", wait_status);
            return 2;
        }
    }
    hipLaunchKernelGGL(
        web_yuyv_detection_overlay_kernel,
        dim3(detection_count),
        dim3(256),
        0,
        stream,
        static_cast<std::uint8_t *>(destination_yuyv),
        destination_pitch,
        width,
        height,
        static_cast<const float *>(detections_f32),
        detection_count,
        confidence_threshold,
        static_cast<const std::uint8_t *>(class_label_alpha),
        static_cast<const int *>(class_label_widths));
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        set_error("Web YUYV overlay kernel launch", launch_status);
        return 3;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}
