#include "egl_present.h"

#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <fontconfig/fontconfig.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime.h>
#include <drm/drm_fourcc.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

thread_local char last_error[1024] = "ok";

constexpr int kPresentPoolSize = 2;
constexpr int kSubtitleHeight = 112;
constexpr int kSubtitleMargin = 24;
constexpr int kFontPixelHeight = 30;
constexpr int kDetectionClassCount = 80;
constexpr int kDetectionLabelWidth = 224;
constexpr int kDetectionLabelHeight = 36;
constexpr int kDetectionLabelPaddingX = 6;
constexpr int kDetectionLabelFontPixelHeight = 25;
constexpr int kPerformanceHudWidth = 600;
constexpr int kPerformanceHudHeight = 76;
constexpr int kPerformanceHudMargin = 16;
constexpr int kPerformanceHudTextPadding = 12;
constexpr int kPerformanceHudFontPixelHeight = 24;

constexpr std::array<const char *, kDetectionClassCount> kCocoClassNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

void set_error(const char *operation, const char *detail) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s", operation, detail);
}

void set_hip_error(const char *operation, hipError_t status) {
    std::snprintf(
        last_error,
        sizeof(last_error),
        "%s: %s (%s)",
        operation,
        hipGetErrorName(status),
        hipGetErrorString(status));
}

bool check_hip(const char *operation, hipError_t status) {
    if (status == hipSuccess) {
        return true;
    }
    set_hip_error(operation, status);
    return false;
}

bool check_egl(const char *operation, EGLBoolean status) {
    if (status == EGL_TRUE) {
        return true;
    }
    std::snprintf(last_error, sizeof(last_error), "%s: EGL error 0x%x", operation, eglGetError());
    return false;
}

std::string gl_shader_log(GLuint object, bool program) {
    GLint length = 0;
    if (program) {
        glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
    } else {
        glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
    }
    if (length <= 1) {
        return "no shader log";
    }
    std::string result(static_cast<std::size_t>(length), '\0');
    if (program) {
        glGetProgramInfoLog(object, length, nullptr, result.data());
    } else {
        glGetShaderInfoLog(object, length, nullptr, result.data());
    }
    return result;
}

GLuint compile_shader(GLenum type, const char *source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE) {
        return shader;
    }
    const std::string log = gl_shader_log(shader, false);
    set_error("OpenGL shader compile", log.c_str());
    glDeleteShader(shader);
    return 0;
}

GLuint create_program() {
    constexpr const char *vertex_source = R"GLSL(
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        out vec2 uv;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            uv = texcoord;
        }
    )GLSL";
    constexpr const char *fragment_source = R"GLSL(
        #version 330 core
        in vec2 uv;
        out vec4 color;
        uniform sampler2D frame_texture;
        void main() {
            color = texture(frame_texture, uv);
        }
    )GLSL";
    const GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    if (vertex == 0) {
        return 0;
    }
    const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (fragment == 0) {
        glDeleteShader(vertex);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE) {
        return program;
    }
    const std::string log = gl_shader_log(program, true);
    set_error("OpenGL program link", log.c_str());
    glDeleteProgram(program);
    return 0;
}

std::vector<std::uint32_t> decode_utf8(const char *text) {
    std::vector<std::uint32_t> result;
    if (text == nullptr) {
        return result;
    }
    const auto *cursor = reinterpret_cast<const unsigned char *>(text);
    while (*cursor != 0) {
        std::uint32_t value = 0xfffd;
        std::size_t length = 1;
        if (*cursor < 0x80) {
            value = *cursor;
        } else if ((*cursor & 0xe0) == 0xc0 && cursor[1] != 0) {
            value = ((*cursor & 0x1f) << 6) | (cursor[1] & 0x3f);
            length = 2;
        } else if ((*cursor & 0xf0) == 0xe0 && cursor[1] != 0 && cursor[2] != 0) {
            value = ((*cursor & 0x0f) << 12) | ((cursor[1] & 0x3f) << 6) |
                (cursor[2] & 0x3f);
            length = 3;
        } else if ((*cursor & 0xf8) == 0xf0 && cursor[1] != 0 && cursor[2] != 0 &&
                   cursor[3] != 0) {
            value = ((*cursor & 0x07) << 18) | ((cursor[1] & 0x3f) << 12) |
                ((cursor[2] & 0x3f) << 6) | (cursor[3] & 0x3f);
            length = 4;
        }
        result.push_back(value);
        cursor += length;
    }
    return result;
}

std::string resolve_font(const char *requested) {
    if (requested != nullptr && requested[0] != '\0') {
        return requested;
    }
    if (FcInit() != FcTrue) {
        return {};
    }
    FcPattern *pattern = FcNameParse(reinterpret_cast<const FcChar8 *>("Noto Sans CJK SC"));
    if (pattern == nullptr) {
        return {};
    }
    FcConfigSubstitute(nullptr, pattern, FcMatchPattern);
    FcDefaultSubstitute(pattern);
    FcResult result = FcResultNoMatch;
    FcPattern *match = FcFontMatch(nullptr, pattern, &result);
    FcPatternDestroy(pattern);
    if (match == nullptr) {
        return {};
    }
    FcChar8 *path = nullptr;
    const FcResult get_result = FcPatternGetString(match, FC_FILE, 0, &path);
    std::string resolved;
    if (get_result == FcResultMatch && path != nullptr) {
        resolved = reinterpret_cast<const char *>(path);
    }
    FcPatternDestroy(match);
    return resolved;
}

__global__ void rgb8_to_rgba_kernel(
    const std::uint8_t *source,
    std::size_t source_pitch,
    int width,
    int height,
    int source_is_bgr,
    uchar4 *destination) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const auto *pixel = source + static_cast<std::size_t>(y) * source_pitch + x * 3;
    const std::uint8_t red = pixel[source_is_bgr ? 2 : 0];
    const std::uint8_t green = pixel[1];
    const std::uint8_t blue = pixel[source_is_bgr ? 0 : 2];
    destination[static_cast<std::size_t>(y) * width + x] = make_uchar4(red, green, blue, 255);
}

__device__ uchar4 detection_color(int class_id) {
    switch (class_id % 20) {
        case 0: return make_uchar4(255, 56, 56, 255);
        case 1: return make_uchar4(255, 157, 151, 255);
        case 2: return make_uchar4(255, 112, 31, 255);
        case 3: return make_uchar4(255, 178, 29, 255);
        case 4: return make_uchar4(207, 210, 49, 255);
        case 5: return make_uchar4(72, 249, 10, 255);
        case 6: return make_uchar4(146, 204, 23, 255);
        case 7: return make_uchar4(61, 219, 134, 255);
        case 8: return make_uchar4(26, 147, 52, 255);
        case 9: return make_uchar4(0, 212, 187, 255);
        case 10: return make_uchar4(44, 153, 168, 255);
        case 11: return make_uchar4(0, 194, 255, 255);
        case 12: return make_uchar4(52, 69, 147, 255);
        case 13: return make_uchar4(100, 115, 255, 255);
        case 14: return make_uchar4(0, 24, 236, 255);
        case 15: return make_uchar4(132, 56, 255, 255);
        case 16: return make_uchar4(82, 0, 133, 255);
        case 17: return make_uchar4(203, 56, 255, 255);
        case 18: return make_uchar4(255, 149, 200, 255);
        default: return make_uchar4(255, 55, 199, 255);
    }
}

__global__ void detection_boxes_kernel(
    uchar4 *destination,
    int width,
    int height,
    const float *detections,
    int detection_count,
    float confidence_threshold,
    const std::uint8_t *class_label_alpha,
    const int *class_label_widths) {
    const int detection_index = static_cast<int>(blockIdx.x);
    const int edge_index = static_cast<int>(threadIdx.x);
    if (detection_index >= detection_count) {
        return;
    }
    const float *detection = detections + static_cast<std::size_t>(detection_index) * 6;
    if (detection[4] <= confidence_threshold) {
        return;
    }
    const int x1 = max(0, min(width - 1, static_cast<int>(roundf(detection[0]))));
    const int y1 = max(0, min(height - 1, static_cast<int>(roundf(detection[1]))));
    const int x2 = max(0, min(width - 1, static_cast<int>(roundf(detection[2]))));
    const int y2 = max(0, min(height - 1, static_cast<int>(roundf(detection[3]))));
    if (x2 <= x1 || y2 <= y1) {
        return;
    }
    const int class_id = max(0, static_cast<int>(roundf(detection[5])));
    const uchar4 color = detection_color(class_id);
    constexpr int thickness = 4;
    for (int offset = edge_index; offset <= x2 - x1; offset += blockDim.x) {
        for (int t = 0; t < thickness; ++t) {
            const int top = min(height - 1, y1 + t);
            const int bottom = max(0, y2 - t);
            destination[static_cast<std::size_t>(top) * width + x1 + offset] = color;
            destination[static_cast<std::size_t>(bottom) * width + x1 + offset] = color;
        }
    }
    for (int offset = edge_index; offset <= y2 - y1; offset += blockDim.x) {
        for (int t = 0; t < thickness; ++t) {
            const int left = min(width - 1, x1 + t);
            const int right = max(0, x2 - t);
            destination[static_cast<std::size_t>(y1 + offset) * width + left] = color;
            destination[static_cast<std::size_t>(y1 + offset) * width + right] = color;
        }
    }
    if (class_id >= kDetectionClassCount || class_label_alpha == nullptr ||
        class_label_widths == nullptr) {
        return;
    }
    const int label_width = max(1, min(kDetectionLabelWidth, class_label_widths[class_id]));
    const int label_x = min(x1, max(0, width - label_width));
    const int label_y = y1 >= kDetectionLabelHeight ? y1 - kDetectionLabelHeight : y1;
    const std::size_t atlas_offset = static_cast<std::size_t>(class_id) *
        kDetectionLabelWidth * kDetectionLabelHeight;
    for (int pixel_index = edge_index;
         pixel_index < label_width * kDetectionLabelHeight;
         pixel_index += blockDim.x) {
        const int local_x = pixel_index % label_width;
        const int local_y = pixel_index / label_width;
        const int destination_x = label_x + local_x;
        const int destination_y = label_y + local_y;
        if (destination_x >= width || destination_y >= height) {
            continue;
        }
        const unsigned int glyph_alpha = class_label_alpha[
            atlas_offset + static_cast<std::size_t>(local_y) * kDetectionLabelWidth + local_x];
        uchar4 label_pixel = make_uchar4(
            static_cast<unsigned char>(static_cast<unsigned int>(color.x) * 3 / 5),
            static_cast<unsigned char>(static_cast<unsigned int>(color.y) * 3 / 5),
            static_cast<unsigned char>(static_cast<unsigned int>(color.z) * 3 / 5),
            255);
        if (glyph_alpha != 0) {
            label_pixel.x = static_cast<unsigned char>(
                (static_cast<unsigned int>(label_pixel.x) * (255 - glyph_alpha) +
                 255 * glyph_alpha) / 255);
            label_pixel.y = static_cast<unsigned char>(
                (static_cast<unsigned int>(label_pixel.y) * (255 - glyph_alpha) +
                 255 * glyph_alpha) / 255);
            label_pixel.z = static_cast<unsigned char>(
                (static_cast<unsigned int>(label_pixel.z) * (255 - glyph_alpha) +
                 255 * glyph_alpha) / 255);
        }
        destination[static_cast<std::size_t>(destination_y) * width + destination_x] = label_pixel;
    }
}

__global__ void subtitle_kernel(
    uchar4 *destination,
    int width,
    int height,
    const std::uint8_t *alpha,
    int subtitle_height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || local_y >= subtitle_height) {
        return;
    }
    const int y = height - subtitle_height + local_y;
    const std::size_t destination_index = static_cast<std::size_t>(y) * width + x;
    uchar4 pixel = destination[destination_index];
    pixel.x = static_cast<unsigned char>(static_cast<unsigned int>(pixel.x) * 3 / 8);
    pixel.y = static_cast<unsigned char>(static_cast<unsigned int>(pixel.y) * 3 / 8);
    pixel.z = static_cast<unsigned char>(static_cast<unsigned int>(pixel.z) * 3 / 8);
    const unsigned int glyph_alpha = alpha[static_cast<std::size_t>(local_y) * width + x];
    if (glyph_alpha != 0) {
        pixel.x = static_cast<unsigned char>(
            (static_cast<unsigned int>(pixel.x) * (255 - glyph_alpha) + 255 * glyph_alpha) / 255);
        pixel.y = static_cast<unsigned char>(
            (static_cast<unsigned int>(pixel.y) * (255 - glyph_alpha) + 244 * glyph_alpha) / 255);
        pixel.z = static_cast<unsigned char>(
            (static_cast<unsigned int>(pixel.z) * (255 - glyph_alpha) + 188 * glyph_alpha) / 255);
    }
    pixel.w = 255;
    destination[destination_index] = pixel;
}

__global__ void performance_hud_kernel(
    uchar4 *destination,
    int width,
    int height,
    const std::uint8_t *alpha) {
    const int local_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (local_x >= kPerformanceHudWidth || local_y >= kPerformanceHudHeight) {
        return;
    }
    const int x = kPerformanceHudMargin + local_x;
    const int y = kPerformanceHudMargin + local_y;
    if (x >= width || y >= height) {
        return;
    }
    const std::size_t destination_index = static_cast<std::size_t>(y) * width + x;
    uchar4 pixel = destination[destination_index];
    pixel.x = static_cast<unsigned char>(static_cast<unsigned int>(pixel.x) / 5);
    pixel.y = static_cast<unsigned char>(static_cast<unsigned int>(pixel.y) / 5);
    pixel.z = static_cast<unsigned char>(static_cast<unsigned int>(pixel.z) / 5);
    const bool border = local_x < 2 || local_y < 2 ||
        local_x >= kPerformanceHudWidth - 2 || local_y >= kPerformanceHudHeight - 2;
    if (border) {
        pixel = make_uchar4(42, 196, 112, 255);
    }
    const unsigned int glyph_alpha =
        alpha[static_cast<std::size_t>(local_y) * kPerformanceHudWidth + local_x];
    if (glyph_alpha != 0) {
        pixel.x = static_cast<unsigned char>(
            (static_cast<unsigned int>(pixel.x) * (255 - glyph_alpha) +
             210 * glyph_alpha) / 255);
        pixel.y = static_cast<unsigned char>(
            (static_cast<unsigned int>(pixel.y) * (255 - glyph_alpha) +
             255 * glyph_alpha) / 255);
        pixel.z = static_cast<unsigned char>(
            (static_cast<unsigned int>(pixel.z) * (255 - glyph_alpha) +
             225 * glyph_alpha) / 255);
    }
    pixel.w = 255;
    destination[destination_index] = pixel;
}

struct PresentSlot {
    void *device_pointer = nullptr;
    int dmabuf_fd = -1;
    std::uint64_t dmabuf_offset = 0;
    EGLImageKHR egl_image = EGL_NO_IMAGE_KHR;
    GLuint texture = 0;
    hipEvent_t hip_ready = nullptr;
    GLsync fence = nullptr;
};

}  // namespace

struct vlm_camera_egl_presenter {
    int frame_width = 0;
    int frame_height = 0;
    int window_width = 0;
    int window_height = 0;
    int hip_device = 0;
    bool visible = false;
    bool close_requested = false;
    bool fullscreen = false;
    bool subtitle_active = false;
    bool performance_hud_active = false;
    bool performance_hud_visible = false;
    std::uint64_t frames_presented = 0;
    std::uint64_t subtitle_updates = 0;
    std::uint64_t subtitle_h2d_bytes = 0;
    std::uint64_t class_label_h2d_bytes = 0;
    std::uint64_t performance_hud_updates = 0;
    std::uint64_t performance_hud_h2d_bytes = 0;
    int next_slot = 0;
    std::thread::id gl_thread;
    std::mutex mutex;

    Display *x_display = nullptr;
    Window x_window = 0;
    Colormap x_colormap = 0;
    Atom wm_delete = None;
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLSurface egl_surface = EGL_NO_SURFACE;
    EGLContext egl_context = EGL_NO_CONTEXT;

    GLuint program = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    std::array<PresentSlot, kPresentPoolSize> slots{};

    FT_Library freetype = nullptr;
    FT_Face face = nullptr;
    std::string font_path;
    std::uint8_t *subtitle_host = nullptr;
    std::uint8_t *subtitle_device = nullptr;
    std::uint8_t *class_label_host = nullptr;
    std::uint8_t *class_label_device = nullptr;
    std::uint8_t *performance_hud_host = nullptr;
    std::uint8_t *performance_hud_device = nullptr;
    hipStream_t subtitle_stream = nullptr;
    hipEvent_t subtitle_ready = nullptr;
    hipEvent_t class_label_ready = nullptr;
    hipEvent_t performance_hud_ready = nullptr;
    bool hsa_initialized = false;

    std::string window_title;
    std::string gl_vendor;
    std::string gl_renderer;
    std::string gl_version;
    std::string interop_device_proof;
};

namespace {

void destroy_presenter(vlm_camera_egl_presenter *presenter) {
    if (presenter == nullptr) {
        return;
    }
    if (presenter->egl_display != EGL_NO_DISPLAY && presenter->egl_context != EGL_NO_CONTEXT) {
        eglMakeCurrent(
            presenter->egl_display,
            presenter->egl_surface,
            presenter->egl_surface,
            presenter->egl_context);
    }
    for (auto &slot : presenter->slots) {
        if (slot.fence != nullptr) {
            glDeleteSync(slot.fence);
            slot.fence = nullptr;
        }
        if (slot.hip_ready != nullptr) {
            (void) hipEventSynchronize(slot.hip_ready);
            (void) hipEventDestroy(slot.hip_ready);
            slot.hip_ready = nullptr;
        }
        if (slot.texture != 0) {
            glDeleteTextures(1, &slot.texture);
            slot.texture = 0;
        }
        if (slot.egl_image != EGL_NO_IMAGE_KHR && presenter->egl_display != EGL_NO_DISPLAY) {
            eglDestroyImageKHR(presenter->egl_display, slot.egl_image);
            slot.egl_image = EGL_NO_IMAGE_KHR;
        }
        if (slot.dmabuf_fd >= 0) {
            close(slot.dmabuf_fd);
            slot.dmabuf_fd = -1;
        }
        if (slot.device_pointer != nullptr) {
            (void) hipFree(slot.device_pointer);
            slot.device_pointer = nullptr;
        }
    }
    if (presenter->vbo != 0) {
        glDeleteBuffers(1, &presenter->vbo);
    }
    if (presenter->vao != 0) {
        glDeleteVertexArrays(1, &presenter->vao);
    }
    if (presenter->program != 0) {
        glDeleteProgram(presenter->program);
    }
    if (presenter->subtitle_ready != nullptr) {
        (void) hipEventDestroy(presenter->subtitle_ready);
    }
    if (presenter->class_label_ready != nullptr) {
        (void) hipEventDestroy(presenter->class_label_ready);
    }
    if (presenter->performance_hud_ready != nullptr) {
        (void) hipEventDestroy(presenter->performance_hud_ready);
    }
    if (presenter->subtitle_stream != nullptr) {
        (void) hipStreamDestroy(presenter->subtitle_stream);
    }
    if (presenter->subtitle_device != nullptr) {
        (void) hipFree(presenter->subtitle_device);
    }
    if (presenter->subtitle_host != nullptr) {
        (void) hipHostFree(presenter->subtitle_host);
    }
    if (presenter->class_label_device != nullptr) {
        (void) hipFree(presenter->class_label_device);
    }
    if (presenter->class_label_host != nullptr) {
        (void) hipHostFree(presenter->class_label_host);
    }
    if (presenter->performance_hud_device != nullptr) {
        (void) hipFree(presenter->performance_hud_device);
    }
    if (presenter->performance_hud_host != nullptr) {
        (void) hipHostFree(presenter->performance_hud_host);
    }
    if (presenter->face != nullptr) {
        FT_Done_Face(presenter->face);
    }
    if (presenter->freetype != nullptr) {
        FT_Done_FreeType(presenter->freetype);
    }
    if (presenter->hsa_initialized) {
        (void) hsa_shut_down();
        presenter->hsa_initialized = false;
    }
    if (presenter->egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(presenter->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (presenter->egl_context != EGL_NO_CONTEXT) {
            eglDestroyContext(presenter->egl_display, presenter->egl_context);
        }
        if (presenter->egl_surface != EGL_NO_SURFACE) {
            eglDestroySurface(presenter->egl_display, presenter->egl_surface);
        }
        eglTerminate(presenter->egl_display);
    }
    if (presenter->x_display != nullptr) {
        if (presenter->x_window != 0) {
            XDestroyWindow(presenter->x_display, presenter->x_window);
        }
        if (presenter->x_colormap != 0) {
            XFreeColormap(presenter->x_display, presenter->x_colormap);
        }
        XCloseDisplay(presenter->x_display);
    }
    delete presenter;
}

bool create_window_and_context(
    vlm_camera_egl_presenter *presenter,
    float scale,
    const char *title) {
    presenter->x_display = XOpenDisplay(nullptr);
    if (presenter->x_display == nullptr) {
        set_error("XOpenDisplay", "DISPLAY is unavailable");
        return false;
    }
    presenter->egl_display = eglGetDisplay(
        reinterpret_cast<EGLNativeDisplayType>(presenter->x_display));
    if (presenter->egl_display == EGL_NO_DISPLAY) {
        set_error("eglGetDisplay", "no EGL display");
        return false;
    }
    EGLint major = 0;
    EGLint minor = 0;
    if (!check_egl("eglInitialize", eglInitialize(presenter->egl_display, &major, &minor))) {
        return false;
    }
    if (!check_egl("eglBindAPI", eglBindAPI(EGL_OPENGL_API))) {
        return false;
    }
    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE,
        EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_NONE,
    };
    EGLConfig config = nullptr;
    EGLint config_count = 0;
    if (!check_egl(
            "eglChooseConfig",
            eglChooseConfig(
                presenter->egl_display, config_attributes, &config, 1, &config_count)) ||
        config_count != 1) {
        set_error("eglChooseConfig", "no OpenGL window configuration");
        return false;
    }
    EGLint visual_id = 0;
    if (!check_egl(
            "eglGetConfigAttrib(EGL_NATIVE_VISUAL_ID)",
            eglGetConfigAttrib(
                presenter->egl_display, config, EGL_NATIVE_VISUAL_ID, &visual_id))) {
        return false;
    }
    XVisualInfo visual_template{};
    visual_template.visualid = static_cast<VisualID>(visual_id);
    int visual_count = 0;
    XVisualInfo *visual_info = XGetVisualInfo(
        presenter->x_display, VisualIDMask, &visual_template, &visual_count);
    if (visual_info == nullptr || visual_count < 1) {
        set_error("XGetVisualInfo", "EGL visual is unavailable in X11");
        return false;
    }
    const int screen = DefaultScreen(presenter->x_display);
    presenter->x_colormap = XCreateColormap(
        presenter->x_display,
        RootWindow(presenter->x_display, screen),
        visual_info->visual,
        AllocNone);
    XSetWindowAttributes attributes{};
    attributes.colormap = presenter->x_colormap;
    attributes.event_mask = StructureNotifyMask | KeyPressMask | ExposureMask;
    presenter->window_width = std::max(1, static_cast<int>(std::lround(presenter->frame_width * scale)));
    presenter->window_height = std::max(
        1, static_cast<int>(std::lround(presenter->frame_height * scale)));
    presenter->x_window = XCreateWindow(
        presenter->x_display,
        RootWindow(presenter->x_display, screen),
        0,
        0,
        static_cast<unsigned int>(presenter->window_width),
        static_cast<unsigned int>(presenter->window_height),
        0,
        visual_info->depth,
        InputOutput,
        visual_info->visual,
        CWColormap | CWEventMask,
        &attributes);
    XFree(visual_info);
    if (presenter->x_window == 0) {
        set_error("XCreateWindow", "returned no window");
        return false;
    }
    presenter->window_title = title;
    XStoreName(presenter->x_display, presenter->x_window, presenter->window_title.c_str());
    const Atom utf8_string = XInternAtom(presenter->x_display, "UTF8_STRING", False);
    const Atom net_wm_name = XInternAtom(presenter->x_display, "_NET_WM_NAME", False);
    XChangeProperty(
        presenter->x_display,
        presenter->x_window,
        net_wm_name,
        utf8_string,
        8,
        PropModeReplace,
        reinterpret_cast<const unsigned char *>(presenter->window_title.c_str()),
        static_cast<int>(presenter->window_title.size()));
    presenter->wm_delete = XInternAtom(presenter->x_display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(presenter->x_display, presenter->x_window, &presenter->wm_delete, 1);
    if (presenter->visible) {
        XMapWindow(presenter->x_display, presenter->x_window);
    }
    XFlush(presenter->x_display);

    presenter->egl_surface = eglCreateWindowSurface(
        presenter->egl_display,
        config,
        reinterpret_cast<EGLNativeWindowType>(presenter->x_window),
        nullptr);
    if (presenter->egl_surface == EGL_NO_SURFACE) {
        set_error("eglCreateWindowSurface", "failed");
        return false;
    }
    const EGLint context_attributes[] = {
        EGL_CONTEXT_MAJOR_VERSION,
        3,
        EGL_CONTEXT_MINOR_VERSION,
        3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    };
    presenter->egl_context = eglCreateContext(
        presenter->egl_display, config, EGL_NO_CONTEXT, context_attributes);
    if (presenter->egl_context == EGL_NO_CONTEXT) {
        set_error("eglCreateContext", "OpenGL 3.3 core context unavailable");
        return false;
    }
    if (!check_egl(
            "eglMakeCurrent",
            eglMakeCurrent(
                presenter->egl_display,
                presenter->egl_surface,
                presenter->egl_surface,
                presenter->egl_context))) {
        return false;
    }
    eglSwapInterval(presenter->egl_display, 0);
    return true;
}

bool create_gl_objects(vlm_camera_egl_presenter *presenter) {
    const auto *vendor = glGetString(GL_VENDOR);
    const auto *renderer = glGetString(GL_RENDERER);
    const auto *version = glGetString(GL_VERSION);
    if (vendor == nullptr || renderer == nullptr || version == nullptr) {
        set_error("glGetString", "OpenGL context is incomplete");
        return false;
    }
    presenter->gl_vendor = reinterpret_cast<const char *>(vendor);
    presenter->gl_renderer = reinterpret_cast<const char *>(renderer);
    presenter->gl_version = reinterpret_cast<const char *>(version);
    presenter->program = create_program();
    if (presenter->program == 0) {
        return false;
    }
    constexpr std::array<float, 16> vertices = {
        -1.0F, -1.0F, 0.0F, 1.0F,
         1.0F, -1.0F, 1.0F, 1.0F,
        -1.0F,  1.0F, 0.0F, 0.0F,
         1.0F,  1.0F, 1.0F, 0.0F,
    };
    glGenVertexArrays(1, &presenter->vao);
    glBindVertexArray(presenter->vao);
    glGenBuffers(1, &presenter->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, presenter->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        reinterpret_cast<const void *>(2 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    const char *egl_extensions = eglQueryString(presenter->egl_display, EGL_EXTENSIONS);
    if (egl_extensions == nullptr ||
        std::strstr(egl_extensions, "EGL_EXT_image_dma_buf_import") == nullptr) {
        set_error("EGL capability", "EGL_EXT_image_dma_buf_import is unavailable");
        return false;
    }
    const std::size_t frame_bytes =
        static_cast<std::size_t>(presenter->frame_width) * presenter->frame_height * 4;
    for (auto &slot : presenter->slots) {
        if (!check_hip(
                "hipMalloc(present DMA-BUF)",
                hipMalloc(&slot.device_pointer, frame_bytes)) ||
            !check_hip(
                "hipEventCreateWithFlags(present)",
                hipEventCreateWithFlags(&slot.hip_ready, hipEventDisableTiming))) {
            return false;
        }
        const hsa_status_t export_status = hsa_amd_portable_export_dmabuf(
            slot.device_pointer,
            frame_bytes,
            &slot.dmabuf_fd,
            &slot.dmabuf_offset);
        if (export_status != HSA_STATUS_SUCCESS || slot.dmabuf_fd < 0 ||
            slot.dmabuf_offset > static_cast<std::uint64_t>(std::numeric_limits<EGLint>::max())) {
            std::snprintf(
                last_error,
                sizeof(last_error),
                "hsa_amd_portable_export_dmabuf failed: status=%d fd=%d offset=%llu",
                static_cast<int>(export_status),
                slot.dmabuf_fd,
                static_cast<unsigned long long>(slot.dmabuf_offset));
            return false;
        }
        const EGLint image_attributes[] = {
            EGL_WIDTH,
            presenter->frame_width,
            EGL_HEIGHT,
            presenter->frame_height,
            EGL_LINUX_DRM_FOURCC_EXT,
            static_cast<EGLint>(DRM_FORMAT_ABGR8888),
            EGL_DMA_BUF_PLANE0_FD_EXT,
            slot.dmabuf_fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            static_cast<EGLint>(slot.dmabuf_offset),
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            presenter->frame_width * 4,
            EGL_NONE,
        };
        slot.egl_image = eglCreateImageKHR(
            presenter->egl_display,
            EGL_NO_CONTEXT,
            EGL_LINUX_DMA_BUF_EXT,
            nullptr,
            image_attributes);
        if (slot.egl_image == EGL_NO_IMAGE_KHR) {
            std::snprintf(
                last_error,
                sizeof(last_error),
                "eglCreateImageKHR(DMA-BUF): EGL error 0x%x",
                eglGetError());
            return false;
        }
        glGenTextures(1, &slot.texture);
        glBindTexture(GL_TEXTURE_2D, slot.texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, slot.egl_image);
        if (glGetError() != GL_NO_ERROR) {
            set_error("glEGLImageTargetTexture2DOES", "DMA-BUF texture import failed");
            return false;
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    presenter->interop_device_proof =
        "hsa_amd_portable_export_dmabuf + EGL_EXT_image_dma_buf_import";
    return true;
}

bool create_text_resources(vlm_camera_egl_presenter *presenter, const char *font_path) {
    presenter->font_path = resolve_font(font_path);
    if (presenter->font_path.empty()) {
        set_error("font lookup", "no font supporting the requested subtitle path was found");
        return false;
    }
    if (FT_Init_FreeType(&presenter->freetype) != 0 ||
        FT_New_Face(presenter->freetype, presenter->font_path.c_str(), 0, &presenter->face) != 0 ||
        FT_Set_Pixel_Sizes(presenter->face, 0, kFontPixelHeight) != 0) {
        set_error("FreeType initialization", presenter->font_path.c_str());
        return false;
    }
    const std::size_t subtitle_bytes =
        static_cast<std::size_t>(presenter->frame_width) * kSubtitleHeight;
    const std::size_t class_label_atlas_bytes = static_cast<std::size_t>(
        kDetectionClassCount) * kDetectionLabelWidth * kDetectionLabelHeight;
    const std::size_t class_label_width_bytes =
        static_cast<std::size_t>(kDetectionClassCount) * sizeof(int);
    const std::size_t class_label_resource_bytes =
        class_label_atlas_bytes + class_label_width_bytes;
    constexpr std::size_t performance_hud_bytes =
        static_cast<std::size_t>(kPerformanceHudWidth) * kPerformanceHudHeight;
    if (!check_hip(
            "hipHostMalloc(subtitle)",
            hipHostMalloc(
                reinterpret_cast<void **>(&presenter->subtitle_host),
                subtitle_bytes,
                hipHostMallocDefault)) ||
        !check_hip(
            "hipMalloc(subtitle)",
            hipMalloc(reinterpret_cast<void **>(&presenter->subtitle_device), subtitle_bytes)) ||
        !check_hip(
            "hipHostMalloc(class labels)",
            hipHostMalloc(
                reinterpret_cast<void **>(&presenter->class_label_host),
                class_label_resource_bytes,
                hipHostMallocDefault)) ||
        !check_hip(
            "hipMalloc(class labels)",
            hipMalloc(
                reinterpret_cast<void **>(&presenter->class_label_device),
                class_label_resource_bytes)) ||
        !check_hip(
            "hipHostMalloc(performance HUD)",
            hipHostMalloc(
                reinterpret_cast<void **>(&presenter->performance_hud_host),
                performance_hud_bytes,
                hipHostMallocDefault)) ||
        !check_hip(
            "hipMalloc(performance HUD)",
            hipMalloc(
                reinterpret_cast<void **>(&presenter->performance_hud_device),
                performance_hud_bytes)) ||
        !check_hip(
            "hipStreamCreateWithFlags(subtitle)",
            hipStreamCreateWithFlags(&presenter->subtitle_stream, hipStreamNonBlocking)) ||
        !check_hip(
            "hipEventCreateWithFlags(subtitle)",
            hipEventCreateWithFlags(&presenter->subtitle_ready, hipEventDisableTiming)) ||
        !check_hip(
            "hipEventCreateWithFlags(class labels)",
            hipEventCreateWithFlags(&presenter->class_label_ready, hipEventDisableTiming)) ||
        !check_hip(
            "hipEventCreateWithFlags(performance HUD)",
            hipEventCreateWithFlags(
                &presenter->performance_hud_ready, hipEventDisableTiming))) {
        return false;
    }
    std::memset(presenter->subtitle_host, 0, subtitle_bytes);
    std::memset(presenter->class_label_host, 0, class_label_resource_bytes);
    std::memset(presenter->performance_hud_host, 0, performance_hud_bytes);
    if (FT_Set_Pixel_Sizes(presenter->face, 0, kDetectionLabelFontPixelHeight) != 0) {
        set_error("FreeType class-label size", presenter->font_path.c_str());
        return false;
    }
    auto *label_widths = reinterpret_cast<int *>(
        presenter->class_label_host + class_label_atlas_bytes);
    for (int class_id = 0; class_id < kDetectionClassCount; ++class_id) {
        auto *label_alpha = presenter->class_label_host +
            static_cast<std::size_t>(class_id) * kDetectionLabelWidth * kDetectionLabelHeight;
        int pen_x = kDetectionLabelPaddingX;
        constexpr int baseline = kDetectionLabelFontPixelHeight + 4;
        for (const unsigned char character : std::string(kCocoClassNames[class_id])) {
            if (FT_Load_Char(presenter->face, character, FT_LOAD_RENDER) != 0) {
                continue;
            }
            FT_GlyphSlot glyph = presenter->face->glyph;
            const int advance = std::max(1, static_cast<int>(glyph->advance.x >> 6));
            if (pen_x + advance + kDetectionLabelPaddingX >= kDetectionLabelWidth) {
                break;
            }
            const int glyph_x = pen_x + glyph->bitmap_left;
            const int glyph_y = baseline - glyph->bitmap_top;
            for (unsigned int row = 0; row < glyph->bitmap.rows; ++row) {
                const int y = glyph_y + static_cast<int>(row);
                if (y < 0 || y >= kDetectionLabelHeight) {
                    continue;
                }
                const auto *bitmap_row = glyph->bitmap.buffer +
                    static_cast<std::ptrdiff_t>(row) * glyph->bitmap.pitch;
                for (unsigned int column = 0; column < glyph->bitmap.width; ++column) {
                    const int x = glyph_x + static_cast<int>(column);
                    if (x < 0 || x >= kDetectionLabelWidth) {
                        continue;
                    }
                    auto &destination = label_alpha[
                        static_cast<std::size_t>(y) * kDetectionLabelWidth + x];
                    destination = std::max(destination, bitmap_row[column]);
                }
            }
            pen_x += advance;
        }
        label_widths[class_id] = std::min(
            kDetectionLabelWidth, pen_x + kDetectionLabelPaddingX);
    }
    if (FT_Set_Pixel_Sizes(presenter->face, 0, kFontPixelHeight) != 0) {
        set_error("FreeType subtitle size", presenter->font_path.c_str());
        return false;
    }
    if (!check_hip(
            "hipMemsetAsync(subtitle)",
            hipMemsetAsync(
                presenter->subtitle_device, 0, subtitle_bytes, presenter->subtitle_stream)) ||
        !check_hip(
            "hipEventRecord(subtitle)",
            hipEventRecord(presenter->subtitle_ready, presenter->subtitle_stream)) ||
        !check_hip(
            "hipMemcpyAsync(class-label atlas H2D)",
            hipMemcpyAsync(
                presenter->class_label_device,
                presenter->class_label_host,
                class_label_resource_bytes,
                hipMemcpyHostToDevice,
                presenter->subtitle_stream)) ||
        !check_hip(
            "hipEventRecord(class-label atlas)",
            hipEventRecord(presenter->class_label_ready, presenter->subtitle_stream)) ||
        !check_hip(
            "hipMemsetAsync(performance HUD)",
            hipMemsetAsync(
                presenter->performance_hud_device,
                0,
                performance_hud_bytes,
                presenter->subtitle_stream)) ||
        !check_hip(
            "hipEventRecord(performance HUD)",
            hipEventRecord(presenter->performance_hud_ready, presenter->subtitle_stream))) {
        return false;
    }
    presenter->class_label_h2d_bytes = class_label_resource_bytes;
    return true;
}

void process_events(vlm_camera_egl_presenter *presenter) {
    while (XPending(presenter->x_display) > 0) {
        XEvent event{};
        XNextEvent(presenter->x_display, &event);
        if (event.type == ClientMessage &&
            static_cast<Atom>(event.xclient.data.l[0]) == presenter->wm_delete) {
            presenter->close_requested = true;
        } else if (event.type == ConfigureNotify) {
            presenter->window_width = std::max(1, event.xconfigure.width);
            presenter->window_height = std::max(1, event.xconfigure.height);
        } else if (event.type == KeyPress) {
            const KeySym key = XLookupKeysym(&event.xkey, 0);
            if (key == XK_Escape) {
                presenter->close_requested = true;
            } else if (key == XK_F11) {
                const Atom state = XInternAtom(presenter->x_display, "_NET_WM_STATE", False);
                const Atom fullscreen =
                    XInternAtom(presenter->x_display, "_NET_WM_STATE_FULLSCREEN", False);
                XEvent message{};
                message.type = ClientMessage;
                message.xclient.window = presenter->x_window;
                message.xclient.message_type = state;
                message.xclient.format = 32;
                message.xclient.data.l[0] = presenter->fullscreen ? 0 : 1;
                message.xclient.data.l[1] = static_cast<long>(fullscreen);
                message.xclient.data.l[3] = 1;
                XSendEvent(
                    presenter->x_display,
                    DefaultRootWindow(presenter->x_display),
                    False,
                    SubstructureRedirectMask | SubstructureNotifyMask,
                    &message);
                presenter->fullscreen = !presenter->fullscreen;
            } else if (key == XK_h || key == XK_H) {
                presenter->performance_hud_visible =
                    !presenter->performance_hud_visible;
            }
        }
    }
}

void set_aspect_viewport(vlm_camera_egl_presenter *presenter) {
    const float source_aspect =
        static_cast<float>(presenter->frame_width) / presenter->frame_height;
    const float window_aspect =
        static_cast<float>(presenter->window_width) / presenter->window_height;
    int viewport_width = presenter->window_width;
    int viewport_height = presenter->window_height;
    if (window_aspect > source_aspect) {
        viewport_width = static_cast<int>(std::lround(viewport_height * source_aspect));
    } else {
        viewport_height = static_cast<int>(std::lround(viewport_width / source_aspect));
    }
    glViewport(
        (presenter->window_width - viewport_width) / 2,
        (presenter->window_height - viewport_height) / 2,
        viewport_width,
        viewport_height);
}

bool wait_slot_fence(PresentSlot &slot) {
    if (slot.fence == nullptr) {
        return true;
    }
    constexpr GLuint64 timeout_ns = 1000000000ULL;
    const GLenum wait_result = glClientWaitSync(
        slot.fence, GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns);
    glDeleteSync(slot.fence);
    slot.fence = nullptr;
    if (wait_result == GL_ALREADY_SIGNALED || wait_result == GL_CONDITION_SATISFIED) {
        return true;
    }
    set_error("OpenGL present-slot fence", "timeout or wait failure");
    return false;
}

}  // namespace

extern "C" const char *vlm_camera_egl_presenter_last_error() {
    return last_error;
}

extern "C" int vlm_camera_egl_presenter_create(
    int frame_width,
    int frame_height,
    float window_scale,
    int visible,
    int hip_device,
    const char *title,
    const char *font_path,
    vlm_camera_egl_presenter **presenter_out) {
    if (presenter_out == nullptr || frame_width <= 0 || frame_height <= 0 ||
        frame_width > 8192 || frame_height > 8192 || window_scale <= 0.05F ||
        window_scale > 4.0F || hip_device < 0) {
        set_error("presenter create", "invalid arguments");
        return 1;
    }
    *presenter_out = nullptr;
    auto *presenter = new (std::nothrow) vlm_camera_egl_presenter();
    if (presenter == nullptr) {
        set_error("presenter create", "allocation failed");
        return 2;
    }
    presenter->frame_width = frame_width;
    presenter->frame_height = frame_height;
    presenter->visible = visible != 0;
    presenter->hip_device = hip_device;
    presenter->gl_thread = std::this_thread::get_id();
    if (!create_window_and_context(
            presenter,
            window_scale,
            title == nullptr ? "YOLO26 + VLM - GPU Camera Demo" : title)) {
        destroy_presenter(presenter);
        return 3;
    }
    if (!check_hip("hipSetDevice", hipSetDevice(hip_device))) {
        destroy_presenter(presenter);
        return 4;
    }
    int device_count = 0;
    if (!check_hip("hipGetDeviceCount", hipGetDeviceCount(&device_count)) ||
        device_count != 1 || hip_device != 0) {
        set_error(
            "HIP/EGL device match",
            "the initial implementation requires one uniquely identifiable ROCm device");
        destroy_presenter(presenter);
        return 4;
    }
    const hsa_status_t hsa_status = hsa_init();
    if (hsa_status != HSA_STATUS_SUCCESS) {
        std::snprintf(
            last_error,
            sizeof(last_error),
            "hsa_init failed with status %d",
            static_cast<int>(hsa_status));
        destroy_presenter(presenter);
        return 4;
    }
    presenter->hsa_initialized = true;
    if (!create_gl_objects(presenter) || !create_text_resources(presenter, font_path)) {
        destroy_presenter(presenter);
        return 5;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    *presenter_out = presenter;
    return 0;
}

extern "C" int vlm_camera_egl_presenter_set_subtitle(
    vlm_camera_egl_presenter *presenter,
    const char *subtitle_utf8) {
    if (presenter == nullptr || subtitle_utf8 == nullptr) {
        set_error("set subtitle", "invalid arguments");
        return 1;
    }
    std::lock_guard<std::mutex> guard(presenter->mutex);
    if (!check_hip("hipSetDevice(subtitle)", hipSetDevice(presenter->hip_device))) {
        return 2;
    }
    const std::size_t subtitle_bytes =
        static_cast<std::size_t>(presenter->frame_width) * kSubtitleHeight;
    std::memset(presenter->subtitle_host, 0, subtitle_bytes);
    const std::vector<std::uint32_t> codepoints = decode_utf8(subtitle_utf8);
    int pen_x = kSubtitleMargin;
    int baseline = kFontPixelHeight + 12;
    constexpr int line_height = kFontPixelHeight + 12;
    const int maximum_x = presenter->frame_width - kSubtitleMargin;
    const auto is_word_separator = [](std::uint32_t codepoint) {
        return codepoint == ' ' || codepoint == '\t' || codepoint == '\n';
    };
    for (std::size_t index = 0; index < codepoints.size(); ++index) {
        const std::uint32_t codepoint = codepoints[index];
        if (codepoint == '\n') {
            pen_x = kSubtitleMargin;
            baseline += line_height;
            continue;
        }
        const bool starts_word = !is_word_separator(codepoint) &&
            (index == 0 || is_word_separator(codepoints[index - 1]));
        if (starts_word && pen_x > kSubtitleMargin) {
            int word_width = 0;
            for (std::size_t lookahead = index; lookahead < codepoints.size(); ++lookahead) {
                if (is_word_separator(codepoints[lookahead])) {
                    break;
                }
                if (FT_Load_Char(
                        presenter->face,
                        codepoints[lookahead],
                        FT_LOAD_DEFAULT) == 0) {
                    word_width += std::max(
                        1,
                        static_cast<int>(presenter->face->glyph->advance.x >> 6));
                }
            }
            if (pen_x + word_width >= maximum_x) {
                pen_x = kSubtitleMargin;
                baseline += line_height;
            }
        }
        if (baseline >= kSubtitleHeight - 4) {
            break;
        }
        if (FT_Load_Char(presenter->face, codepoint, FT_LOAD_RENDER) != 0) {
            continue;
        }
        FT_GlyphSlot glyph = presenter->face->glyph;
        const int advance = std::max(1, static_cast<int>(glyph->advance.x >> 6));
        if (pen_x + advance >= maximum_x) {
            pen_x = kSubtitleMargin;
            baseline += line_height;
        }
        if (baseline >= kSubtitleHeight - 4) {
            break;
        }
        const int glyph_x = pen_x + glyph->bitmap_left;
        const int glyph_y = baseline - glyph->bitmap_top;
        for (unsigned int row = 0; row < glyph->bitmap.rows; ++row) {
            const int y = glyph_y + static_cast<int>(row);
            if (y < 0 || y >= kSubtitleHeight) {
                continue;
            }
            const auto *bitmap_row = glyph->bitmap.buffer +
                static_cast<std::ptrdiff_t>(row) * glyph->bitmap.pitch;
            for (unsigned int column = 0; column < glyph->bitmap.width; ++column) {
                const int x = glyph_x + static_cast<int>(column);
                if (x < 0 || x >= presenter->frame_width) {
                    continue;
                }
                auto &destination = presenter->subtitle_host[
                    static_cast<std::size_t>(y) * presenter->frame_width + x];
                destination = std::max(destination, bitmap_row[column]);
            }
        }
        pen_x += advance;
    }
    if (!check_hip(
            "hipMemcpyAsync(subtitle H2D)",
            hipMemcpyAsync(
                presenter->subtitle_device,
                presenter->subtitle_host,
                subtitle_bytes,
                hipMemcpyHostToDevice,
                presenter->subtitle_stream)) ||
        !check_hip(
            "hipEventRecord(subtitle ready)",
            hipEventRecord(presenter->subtitle_ready, presenter->subtitle_stream))) {
        return 3;
    }
    presenter->subtitle_active = !codepoints.empty();
    presenter->subtitle_updates += 1;
    presenter->subtitle_h2d_bytes += subtitle_bytes;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_egl_presenter_set_performance_hud_text(
    vlm_camera_egl_presenter *presenter,
    const char *text_utf8) {
    if (presenter == nullptr || text_utf8 == nullptr) {
        set_error("set performance HUD", "invalid arguments");
        return 1;
    }
    std::lock_guard<std::mutex> guard(presenter->mutex);
    if (!check_hip("hipSetDevice(performance HUD)", hipSetDevice(presenter->hip_device))) {
        return 2;
    }
    if (FT_Set_Pixel_Sizes(presenter->face, 0, kPerformanceHudFontPixelHeight) != 0) {
        set_error("FreeType performance HUD size", presenter->font_path.c_str());
        return 3;
    }
    constexpr std::size_t performance_hud_bytes =
        static_cast<std::size_t>(kPerformanceHudWidth) * kPerformanceHudHeight;
    std::memset(presenter->performance_hud_host, 0, performance_hud_bytes);
    const std::vector<std::uint32_t> codepoints = decode_utf8(text_utf8);
    int pen_x = kPerformanceHudTextPadding;
    int baseline = kPerformanceHudFontPixelHeight + 5;
    constexpr int line_height = kPerformanceHudFontPixelHeight + 9;
    const int maximum_x = kPerformanceHudWidth - kPerformanceHudTextPadding;
    for (const std::uint32_t codepoint : codepoints) {
        if (codepoint == '\n') {
            pen_x = kPerformanceHudTextPadding;
            baseline += line_height;
            continue;
        }
        if (baseline >= kPerformanceHudHeight - 3) {
            break;
        }
        if (FT_Load_Char(presenter->face, codepoint, FT_LOAD_RENDER) != 0) {
            continue;
        }
        FT_GlyphSlot glyph = presenter->face->glyph;
        const int advance = std::max(1, static_cast<int>(glyph->advance.x >> 6));
        if (pen_x + advance >= maximum_x) {
            pen_x = kPerformanceHudTextPadding;
            baseline += line_height;
        }
        if (baseline >= kPerformanceHudHeight - 3) {
            break;
        }
        const int glyph_x = pen_x + glyph->bitmap_left;
        const int glyph_y = baseline - glyph->bitmap_top;
        for (unsigned int row = 0; row < glyph->bitmap.rows; ++row) {
            const int y = glyph_y + static_cast<int>(row);
            if (y < 0 || y >= kPerformanceHudHeight) {
                continue;
            }
            const auto *bitmap_row = glyph->bitmap.buffer +
                static_cast<std::ptrdiff_t>(row) * glyph->bitmap.pitch;
            for (unsigned int column = 0; column < glyph->bitmap.width; ++column) {
                const int x = glyph_x + static_cast<int>(column);
                if (x < 0 || x >= kPerformanceHudWidth) {
                    continue;
                }
                auto &destination = presenter->performance_hud_host[
                    static_cast<std::size_t>(y) * kPerformanceHudWidth + x];
                destination = std::max(destination, bitmap_row[column]);
            }
        }
        pen_x += advance;
    }
    if (FT_Set_Pixel_Sizes(presenter->face, 0, kFontPixelHeight) != 0) {
        set_error("FreeType subtitle size", presenter->font_path.c_str());
        return 3;
    }
    if (!check_hip(
            "hipMemcpyAsync(performance HUD H2D)",
            hipMemcpyAsync(
                presenter->performance_hud_device,
                presenter->performance_hud_host,
                performance_hud_bytes,
                hipMemcpyHostToDevice,
                presenter->subtitle_stream)) ||
        !check_hip(
            "hipEventRecord(performance HUD ready)",
            hipEventRecord(presenter->performance_hud_ready, presenter->subtitle_stream))) {
        return 4;
    }
    presenter->performance_hud_active = !codepoints.empty();
    presenter->performance_hud_updates += 1;
    presenter->performance_hud_h2d_bytes += performance_hud_bytes;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_egl_presenter_set_performance_hud_visible(
    vlm_camera_egl_presenter *presenter,
    int visible) {
    if (presenter == nullptr) {
        set_error("set performance HUD visibility", "invalid presenter");
        return 1;
    }
    std::lock_guard<std::mutex> guard(presenter->mutex);
    presenter->performance_hud_visible = visible != 0;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_egl_presenter_present_rgb8(
    vlm_camera_egl_presenter *presenter,
    const void *source_rgb8,
    std::size_t source_pitch,
    int source_is_bgr,
    const void *detections_f32,
    int detection_count,
    float confidence_threshold,
    void *source_ready_event,
    void *detections_ready_event,
    void *stream_pointer) {
    if (presenter == nullptr || source_rgb8 == nullptr ||
        source_pitch < static_cast<std::size_t>(presenter->frame_width) * 3 ||
        (detection_count > 0 && detections_f32 == nullptr) || detection_count < 0 ||
        detection_count > 4096 || confidence_threshold < 0.0F || confidence_threshold > 1.0F) {
        set_error("present frame", "invalid arguments");
        return 1;
    }
    if (presenter->gl_thread != std::this_thread::get_id()) {
        set_error("present frame", "EGL presentation must run on the creator thread");
        return 2;
    }
    std::lock_guard<std::mutex> guard(presenter->mutex);
    process_events(presenter);
    if (presenter->close_requested) {
        return 1;
    }
    if (!check_hip("hipSetDevice(present)", hipSetDevice(presenter->hip_device))) {
        return 3;
    }
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (source_ready_event != nullptr &&
        !check_hip(
            "hipStreamWaitEvent(source)",
            hipStreamWaitEvent(stream, reinterpret_cast<hipEvent_t>(source_ready_event), 0))) {
        return 4;
    }
    if (detections_ready_event != nullptr &&
        !check_hip(
            "hipStreamWaitEvent(detections)",
            hipStreamWaitEvent(
                stream, reinterpret_cast<hipEvent_t>(detections_ready_event), 0))) {
        return 5;
    }
    if (detection_count > 0 &&
        !check_hip(
            "hipStreamWaitEvent(class labels)",
            hipStreamWaitEvent(stream, presenter->class_label_ready, 0))) {
        return 6;
    }
    if (presenter->subtitle_active &&
        !check_hip(
            "hipStreamWaitEvent(subtitle)",
            hipStreamWaitEvent(stream, presenter->subtitle_ready, 0))) {
        return 7;
    }
    if (presenter->performance_hud_active && presenter->performance_hud_visible &&
        !check_hip(
            "hipStreamWaitEvent(performance HUD)",
            hipStreamWaitEvent(stream, presenter->performance_hud_ready, 0))) {
        return 8;
    }
    PresentSlot &slot = presenter->slots[static_cast<std::size_t>(presenter->next_slot)];
    presenter->next_slot = (presenter->next_slot + 1) % kPresentPoolSize;
    if (!wait_slot_fence(slot)) {
        return 9;
    }
    if (slot.device_pointer == nullptr) {
        set_error("HIP/EGL present slot", "device allocation is unavailable");
        return 10;
    }
    const dim3 block(16, 16);
    const dim3 grid(
        (presenter->frame_width + block.x - 1) / block.x,
        (presenter->frame_height + block.y - 1) / block.y);
    hipLaunchKernelGGL(
        rgb8_to_rgba_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const std::uint8_t *>(source_rgb8),
        source_pitch,
        presenter->frame_width,
        presenter->frame_height,
        source_is_bgr,
        static_cast<uchar4 *>(slot.device_pointer));
    if (detections_f32 != nullptr && detection_count > 0) {
        hipLaunchKernelGGL(
            detection_boxes_kernel,
            dim3(detection_count),
            dim3(256),
            0,
            stream,
            static_cast<uchar4 *>(slot.device_pointer),
            presenter->frame_width,
            presenter->frame_height,
            static_cast<const float *>(detections_f32),
            detection_count,
            confidence_threshold,
            presenter->class_label_device,
            reinterpret_cast<const int *>(
                presenter->class_label_device +
                static_cast<std::size_t>(kDetectionClassCount) *
                    kDetectionLabelWidth * kDetectionLabelHeight));
    }
    if (presenter->subtitle_active) {
        const dim3 subtitle_grid(
            (presenter->frame_width + block.x - 1) / block.x,
            (kSubtitleHeight + block.y - 1) / block.y);
        hipLaunchKernelGGL(
            subtitle_kernel,
            subtitle_grid,
            block,
            0,
            stream,
            static_cast<uchar4 *>(slot.device_pointer),
            presenter->frame_width,
            presenter->frame_height,
            presenter->subtitle_device,
            kSubtitleHeight);
    }
    if (presenter->performance_hud_active && presenter->performance_hud_visible) {
        const dim3 performance_hud_grid(
            (kPerformanceHudWidth + block.x - 1) / block.x,
            (kPerformanceHudHeight + block.y - 1) / block.y);
        hipLaunchKernelGGL(
            performance_hud_kernel,
            performance_hud_grid,
            block,
            0,
            stream,
            static_cast<uchar4 *>(slot.device_pointer),
            presenter->frame_width,
            presenter->frame_height,
            presenter->performance_hud_device);
    }
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        set_hip_error("present compositor kernel", launch_status);
        return 11;
    }
    if (!check_hip(
            "hipEventRecord(present ready)",
            hipEventRecord(slot.hip_ready, stream)) ||
        !check_hip(
            "hipEventSynchronize(present ready)",
            hipEventSynchronize(slot.hip_ready))) {
        return 12;
    }

    glBindTexture(GL_TEXTURE_2D, slot.texture);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    set_aspect_viewport(presenter);
    glUseProgram(presenter->program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, slot.texture);
    glUniform1i(glGetUniformLocation(presenter->program, "frame_texture"), 0);
    glBindVertexArray(presenter->vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    if (glGetError() != GL_NO_ERROR ||
        !check_egl("eglSwapBuffers", eglSwapBuffers(presenter->egl_display, presenter->egl_surface))) {
        return 13;
    }
    slot.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    presenter->frames_presented += 1;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_egl_presenter_should_close(vlm_camera_egl_presenter *presenter) {
    if (presenter == nullptr) {
        return 1;
    }
    std::lock_guard<std::mutex> guard(presenter->mutex);
    if (presenter->gl_thread == std::this_thread::get_id()) {
        process_events(presenter);
    }
    return presenter->close_requested ? 1 : 0;
}

extern "C" int vlm_camera_egl_presenter_get_info(
    vlm_camera_egl_presenter *presenter,
    vlm_camera_egl_presenter_info *info_out) {
    if (presenter == nullptr || info_out == nullptr) {
        set_error("get presenter info", "invalid arguments");
        return 1;
    }
    std::lock_guard<std::mutex> guard(presenter->mutex);
    *info_out = {
        presenter->frame_width,
        presenter->frame_height,
        presenter->window_width,
        presenter->window_height,
        presenter->hip_device,
        kPresentPoolSize,
        presenter->frames_presented,
        presenter->subtitle_updates,
        presenter->subtitle_h2d_bytes,
        presenter->class_label_h2d_bytes,
        presenter->performance_hud_updates,
        presenter->performance_hud_h2d_bytes,
        kDetectionClassCount,
        kDetectionLabelWidth,
        kDetectionLabelHeight,
        presenter->performance_hud_visible ? 1 : 0,
        kPerformanceHudWidth,
        kPerformanceHudHeight,
        presenter->gl_vendor.c_str(),
        presenter->gl_renderer.c_str(),
        presenter->gl_version.c_str(),
        presenter->interop_device_proof.c_str(),
        presenter->font_path.c_str(),
        presenter->window_title.c_str(),
        "COCO80 startup atlas; GPU box/label compositor",
        "two-line GPU compositor; H toggles visibility",
    };
    return 0;
}

extern "C" void vlm_camera_egl_presenter_destroy(vlm_camera_egl_presenter *presenter) {
    if (presenter == nullptr) {
        return;
    }
    destroy_presenter(presenter);
}
