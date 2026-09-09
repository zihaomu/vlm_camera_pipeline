#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vlm_camera_egl_presenter vlm_camera_egl_presenter;

typedef struct vlm_camera_egl_presenter_info {
    int frame_width;
    int frame_height;
    int window_width;
    int window_height;
    int hip_device;
    int present_pool_size;
    uint64_t frames_presented;
    uint64_t subtitle_updates;
    uint64_t subtitle_h2d_bytes;
    uint64_t class_label_h2d_bytes;
    uint64_t performance_hud_updates;
    uint64_t performance_hud_h2d_bytes;
    int class_label_count;
    int class_label_width;
    int class_label_height;
    int performance_hud_visible;
    int performance_hud_width;
    int performance_hud_height;
    const char *gl_vendor;
    const char *gl_renderer;
    const char *gl_version;
    const char *interop_device_proof;
    const char *font_path;
    const char *window_title;
    const char *class_label_mode;
    const char *performance_hud_mode;
} vlm_camera_egl_presenter_info;

// Creates an EGL/OpenGL window and two HIP allocations exported as DMA-BUF EGLImages.
// A hidden X11 window is still a real EGL window surface and is useful for CI probes.
int vlm_camera_egl_presenter_create(
    int frame_width,
    int frame_height,
    float window_scale,
    int visible,
    int hip_device,
    const char *title,
    const char *font_path,
    vlm_camera_egl_presenter **presenter_out);

// Renders UTF-8 into a host-side glyph mask and asynchronously uploads only that UI resource.
// Video pixels are never accepted by this API on the host.
int vlm_camera_egl_presenter_set_subtitle(
    vlm_camera_egl_presenter *presenter,
    const char *subtitle_utf8);

// Updates a small performance UI mask. This is control-plane text, never frame pixels.
int vlm_camera_egl_presenter_set_performance_hud_text(
    vlm_camera_egl_presenter *presenter,
    const char *text_utf8);

// The H key also toggles this state inside the native X11 event loop.
int vlm_camera_egl_presenter_set_performance_hud_visible(
    vlm_camera_egl_presenter *presenter,
    int visible);

// source_rgb8 and detections_f32 must be device pointers on hip_device. detections_f32 has
// [detection_count, 6] rows in xyxy/conf/class layout. COCO class-name glyphs are uploaded once
// at presenter creation and boxes plus labels are composited on the GPU. Events and stream may be null.
// Returns 0 when a frame was presented, 1 when a close event was consumed,
// and a value >= 2 on error.
int vlm_camera_egl_presenter_present_rgb8(
    vlm_camera_egl_presenter *presenter,
    const void *source_rgb8,
    size_t source_pitch,
    int source_is_bgr,
    const void *detections_f32,
    int detection_count,
    float confidence_threshold,
    void *source_ready_event,
    void *detections_ready_event,
    void *stream);

int vlm_camera_egl_presenter_should_close(vlm_camera_egl_presenter *presenter);

int vlm_camera_egl_presenter_get_info(
    vlm_camera_egl_presenter *presenter,
    vlm_camera_egl_presenter_info *info_out);

void vlm_camera_egl_presenter_destroy(vlm_camera_egl_presenter *presenter);

const char *vlm_camera_egl_presenter_last_error(void);

#ifdef __cplusplus
}
#endif
