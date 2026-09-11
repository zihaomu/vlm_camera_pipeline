#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vlm_camera_gpu_capture vlm_camera_gpu_capture;

typedef struct vlm_camera_gpu_capture_options {
    const char *device;
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    uint32_t camera_buffers;
    uint32_t clean_rgb_buffers;
    uint32_t web_stream_buffers;
    int hip_device;
    uint32_t horizontal_flip;
} vlm_camera_gpu_capture_options;

typedef struct vlm_camera_gpu_capture_info {
    uint32_t width;
    uint32_t height;
    uint32_t fourcc;
    uint32_t bytes_per_line;
    uint32_t size_image;
    size_t camera_allocation_bytes;
    uint32_t web_stream_fourcc;
    uint32_t web_stream_bytes_per_line;
    uint32_t web_stream_size_image;
    uint32_t fps_numerator;
    uint32_t fps_denominator;
    uint32_t camera_buffers;
    uint32_t clean_rgb_buffers;
    uint32_t web_stream_buffers;
    int hip_device;
    uint32_t horizontal_flip;
    uint64_t frames_acquired;
    uint64_t frames_requeued;
    uint64_t frames_dropped_no_clean_slot;
    uint64_t web_stream_frames_converted;
    uint64_t web_stream_frames_released;
    uint64_t web_stream_frames_dropped_no_slot;
    uint64_t web_stream_gpu_bytes_written;
    const char *driver;
    const char *card;
    const char *memory_path;
    const char *color_conversion;
} vlm_camera_gpu_capture_info;

typedef struct vlm_camera_gpu_frame {
    uint64_t frame_id;
    uint64_t captured_monotonic_ns;
    uint64_t generation;
    uint32_t sequence;
    uint32_t slot_index;
    uint32_t width;
    uint32_t height;
    uint32_t bytes_used;
    size_t rgb_pitch;
    size_t rgb_allocation_bytes;
    void *rgb_device_pointer;
    void *ready_event;
    uint64_t web_stream_generation;
    uint32_t web_stream_slot_index;
    int32_t web_stream_dmabuf_fd;
    uint32_t web_stream_fourcc;
    void *web_stream_device_pointer;
    size_t web_stream_pitch;
    size_t web_stream_size_bytes;
    size_t web_stream_allocation_bytes;
} vlm_camera_gpu_frame;

// Creates a strict HIP-owned DMA-BUF capture ring. The running ISP4 driver must
// already include the locked foreign DMA-BUF PRIME/GART import patch.
int vlm_camera_gpu_capture_create(
    const vlm_camera_gpu_capture_options *options,
    vlm_camera_gpu_capture **capture_out);

// Returns 0 on success, 1 when all bounded clean RGB slots are leased, 2 on
// poll timeout, and a value >= 10 for a fatal capture/runtime error.
int vlm_camera_gpu_capture_acquire_rgb8(
    vlm_camera_gpu_capture *capture,
    uint32_t timeout_ms,
    vlm_camera_gpu_frame *frame_out);

// Releasing a frame optionally waits on the supplied per-resource HIP event.
// It never performs a device-wide synchronization.
int vlm_camera_gpu_capture_release_rgb8(
    vlm_camera_gpu_capture *capture,
    uint32_t slot_index,
    uint64_t generation,
    void *consumer_done_event);

// Releases a browser encoder YUYV DMA-BUF slot. The caller must retain it until
// the hardware encoder has emitted the corresponding access unit.
int vlm_camera_gpu_capture_release_web_stream(
    vlm_camera_gpu_capture *capture,
    uint32_t slot_index,
    uint64_t generation);

int vlm_camera_gpu_capture_get_info(
    vlm_camera_gpu_capture *capture,
    vlm_camera_gpu_capture_info *info_out);

void vlm_camera_gpu_capture_destroy(vlm_camera_gpu_capture *capture);

// Public for an offline numerical test. Both pointers must be HIP device
// pointers; this function launches only a GPU kernel and performs no copy.
int vlm_camera_nv12_to_rgb8(
    const void *source_nv12,
    size_t source_y_pitch,
    size_t source_uv_offset,
    size_t source_uv_pitch,
    int width,
    int height,
    void *destination_rgb8,
    size_t destination_pitch,
    int horizontal_flip,
    void *source_ready_event,
    void *stream);

// Public for the encoder-facing packed-format numerical test. All pixels stay
// on the GPU in production; the checker alone downloads the tiny test image.
int vlm_camera_nv12_to_yuyv(
    const void *source_nv12,
    size_t source_y_pitch,
    size_t source_uv_offset,
    size_t source_uv_pitch,
    int width,
    int height,
    void *destination_yuyv,
    size_t destination_pitch,
    int horizontal_flip,
    void *stream);

const char *vlm_camera_gpu_capture_last_error(void);

#ifdef __cplusplus
}
#endif
