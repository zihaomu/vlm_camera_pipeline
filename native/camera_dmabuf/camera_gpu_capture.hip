#include "camera_gpu_capture.h"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local char last_error[512] = "ok";

// ROCr may suballocate small hipMalloc requests from a larger BO. Portable
// DMA-BUF export then returns the allocation's byte offset inside that BO.
// V4L2's single-plane DMABUF ABI has no offset field, so force these camera
// allocations above the small-allocation pool and keep rejecting any export
// that does not begin at byte zero.
constexpr std::size_t kCameraExportAllocationFloor = 4U * 1024U * 1024U;

void set_error(const char *operation, const std::string &detail) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s", operation, detail.c_str());
}

void set_errno_error(const char *operation) {
    set_error(operation, "errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")");
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

void hip_check(hipError_t status, const char *operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorName(status) + " (" +
            hipGetErrorString(status) + ")");
    }
}

std::string hsa_error(hsa_status_t status) {
    const char *text = nullptr;
    if (hsa_status_string(status, &text) == HSA_STATUS_SUCCESS && text != nullptr) {
        return std::string(text) + " (" + std::to_string(status) + ")";
    }
    return "status " + std::to_string(status);
}

void ioctl_check(int fd, unsigned long request, void *argument, const char *operation) {
    int status;
    do {
        status = ioctl(fd, request, argument);
    } while (status < 0 && errno == EINTR);
    if (status < 0) {
        throw std::runtime_error(
            std::string(operation) + ": errno=" + std::to_string(errno) + " (" +
            std::strerror(errno) + ")");
    }
}

std::uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(value.tv_nsec);
}

std::size_t page_align(std::size_t value) {
    const long raw_page_size = sysconf(_SC_PAGESIZE);
    const std::size_t page_size = raw_page_size > 0
        ? static_cast<std::size_t>(raw_page_size)
        : 4096U;
    return (value + page_size - 1) & ~(page_size - 1);
}

struct CameraBuffer {
    void *device_pointer = nullptr;
    std::size_t allocation_bytes = 0;
    int dmabuf_fd = -1;
};

struct CleanSlot {
    void *device_pointer = nullptr;
    std::size_t pitch = 0;
    std::size_t allocation_bytes = 0;
    hipEvent_t ready = nullptr;
    std::uint64_t generation = 0;
    bool in_use = false;
};

__device__ std::uint8_t clamp_u8(int value) {
    return static_cast<std::uint8_t>(max(0, min(255, value)));
}

__global__ void nv12_to_rgb8_bt601_limited_kernel(
    const std::uint8_t *source,
    std::size_t source_y_pitch,
    std::size_t source_uv_offset,
    std::size_t source_uv_pitch,
    int width,
    int height,
    std::uint8_t *destination,
    std::size_t destination_pitch) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const auto *y_row = source + static_cast<std::size_t>(y) * source_y_pitch;
    const auto *uv_row = source + source_uv_offset +
        static_cast<std::size_t>(y / 2) * source_uv_pitch;
    const int uv_x = (x / 2) * 2;
    const int c = max(0, static_cast<int>(y_row[x]) - 16);
    const int d = static_cast<int>(uv_row[uv_x]) - 128;
    const int e = static_cast<int>(uv_row[uv_x + 1]) - 128;
    const int red = (298 * c + 409 * e + 128) >> 8;
    const int green = (298 * c - 100 * d - 208 * e + 128) >> 8;
    const int blue = (298 * c + 516 * d + 128) >> 8;
    auto *pixel = destination + static_cast<std::size_t>(y) * destination_pitch +
        static_cast<std::size_t>(x) * 3;
    pixel[0] = clamp_u8(red);
    pixel[1] = clamp_u8(green);
    pixel[2] = clamp_u8(blue);
}

int launch_nv12_to_rgb8(
    const void *source_nv12,
    std::size_t source_y_pitch,
    std::size_t source_uv_offset,
    std::size_t source_uv_pitch,
    int width,
    int height,
    void *destination_rgb8,
    std::size_t destination_pitch,
    void *source_ready_event,
    void *stream_pointer) {
    if (source_nv12 == nullptr || destination_rgb8 == nullptr || width <= 0 || height <= 0 ||
        (width & 1) != 0 || (height & 1) != 0 ||
        source_y_pitch < static_cast<std::size_t>(width) ||
        source_uv_pitch < static_cast<std::size_t>(width) ||
        source_uv_offset < source_y_pitch * static_cast<std::size_t>(height) ||
        destination_pitch < static_cast<std::size_t>(width) * 3) {
        set_error("NV12 to RGB8", "invalid device pointers, dimensions, pitch, or plane offset");
        return 1;
    }
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    if (source_ready_event != nullptr) {
        const hipError_t wait_status = hipStreamWaitEvent(
            stream, reinterpret_cast<hipEvent_t>(source_ready_event), 0);
        if (wait_status != hipSuccess) {
            set_hip_error("NV12 source event wait", wait_status);
            return 2;
        }
    }
    const dim3 block(16, 16);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1) / block.y);
    hipLaunchKernelGGL(
        nv12_to_rgb8_bt601_limited_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const std::uint8_t *>(source_nv12),
        source_y_pitch,
        source_uv_offset,
        source_uv_pitch,
        width,
        height,
        static_cast<std::uint8_t *>(destination_rgb8),
        destination_pitch);
    const hipError_t launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
        set_hip_error("NV12 to RGB8 kernel", launch_status);
        return 3;
    }
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

}  // namespace

struct vlm_camera_gpu_capture {
    int fd = -1;
    bool streaming = false;
    bool hsa_initialized = false;
    int hip_device = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t fourcc = 0;
    std::uint32_t bytes_per_line = 0;
    std::uint32_t size_image = 0;
    std::size_t camera_allocation_bytes = 0;
    std::uint32_t fps_numerator = 0;
    std::uint32_t fps_denominator = 0;
    std::uint64_t frames_acquired = 0;
    std::uint64_t frames_requeued = 0;
    std::uint64_t frames_dropped_no_clean_slot = 0;
    std::string driver;
    std::string card;
    std::vector<CameraBuffer> camera_buffers;
    std::vector<CleanSlot> clean_slots;
    hipStream_t ingest_stream = nullptr;
    bool v4l2_buffers_requested = false;
};

namespace {

void queue_camera_buffer(vlm_camera_gpu_capture *capture, std::uint32_t index) {
    if (index >= capture->camera_buffers.size()) {
        throw std::runtime_error("camera returned an out-of-range buffer index");
    }
    const CameraBuffer &buffer = capture->camera_buffers[index];
    v4l2_buffer queue{};
    queue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    queue.memory = V4L2_MEMORY_DMABUF;
    queue.index = index;
    queue.m.fd = buffer.dmabuf_fd;
    queue.length = static_cast<std::uint32_t>(buffer.allocation_bytes);
    ioctl_check(capture->fd, VIDIOC_QBUF, &queue, "VIDIOC_QBUF(DMABUF)");
}

void destroy_capture(vlm_camera_gpu_capture *capture) noexcept {
    if (capture == nullptr) {
        return;
    }
    if (capture->ingest_stream != nullptr) {
        static_cast<void>(hipStreamSynchronize(capture->ingest_stream));
    }
    if (capture->streaming && capture->fd >= 0) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        static_cast<void>(ioctl(capture->fd, VIDIOC_STREAMOFF, &type));
        capture->streaming = false;
    }
    if (capture->fd >= 0 && capture->v4l2_buffers_requested) {
        v4l2_requestbuffers request{};
        request.count = 0;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_DMABUF;
        static_cast<void>(ioctl(capture->fd, VIDIOC_REQBUFS, &request));
        capture->v4l2_buffers_requested = false;
    }
    for (CameraBuffer &buffer : capture->camera_buffers) {
        if (buffer.dmabuf_fd >= 0) {
            close(buffer.dmabuf_fd);
            buffer.dmabuf_fd = -1;
        }
        if (buffer.device_pointer != nullptr) {
            static_cast<void>(hipFree(buffer.device_pointer));
            buffer.device_pointer = nullptr;
        }
    }
    for (CleanSlot &slot : capture->clean_slots) {
        if (slot.ready != nullptr) {
            static_cast<void>(hipEventDestroy(slot.ready));
            slot.ready = nullptr;
        }
        if (slot.device_pointer != nullptr) {
            static_cast<void>(hipFree(slot.device_pointer));
            slot.device_pointer = nullptr;
        }
    }
    if (capture->ingest_stream != nullptr) {
        static_cast<void>(hipStreamDestroy(capture->ingest_stream));
        capture->ingest_stream = nullptr;
    }
    if (capture->hsa_initialized) {
        static_cast<void>(hsa_shut_down());
        capture->hsa_initialized = false;
    }
    if (capture->fd >= 0) {
        close(capture->fd);
        capture->fd = -1;
    }
    delete capture;
}

void initialize_capture(
    vlm_camera_gpu_capture *capture,
    const vlm_camera_gpu_capture_options &options) {
    hip_check(hipSetDevice(options.hip_device), "hipSetDevice(camera)");
    int device_count = 0;
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount(camera)");
    if (device_count != 1 || options.hip_device != 0) {
        throw std::runtime_error("strict capture currently requires one ROCm GPU at device 0");
    }
    capture->hip_device = options.hip_device;
    capture->fd = open(options.device, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (capture->fd < 0) {
        throw std::runtime_error(
            "open camera: errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    }

    v4l2_capability capability{};
    ioctl_check(capture->fd, VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP");
    const std::uint32_t capabilities = capability.device_caps != 0
        ? capability.device_caps
        : capability.capabilities;
    if ((capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
        (capabilities & V4L2_CAP_STREAMING) == 0) {
        throw std::runtime_error("camera lacks VIDEO_CAPTURE or STREAMING capability");
    }
    capture->driver = reinterpret_cast<const char *>(capability.driver);
    capture->card = reinterpret_cast<const char *>(capability.card);
    if (capture->driver != "amd_isp_capture") {
        throw std::runtime_error("strict capture expected the amd_isp_capture driver");
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = options.width;
    format.fmt.pix.height = options.height;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    ioctl_check(capture->fd, VIDIOC_S_FMT, &format, "VIDIOC_S_FMT(NV12)");
    capture->width = format.fmt.pix.width;
    capture->height = format.fmt.pix.height;
    capture->fourcc = format.fmt.pix.pixelformat;
    capture->bytes_per_line = format.fmt.pix.bytesperline;
    capture->size_image = format.fmt.pix.sizeimage;
    if (capture->fourcc != V4L2_PIX_FMT_NV12 || capture->width != options.width ||
        capture->height != options.height || (capture->width & 1U) != 0 ||
        (capture->height & 1U) != 0 || capture->bytes_per_line < capture->width) {
        throw std::runtime_error("camera did not accept the fixed even-sized NV12 geometry");
    }
    const std::size_t uv_offset =
        static_cast<std::size_t>(capture->bytes_per_line) * capture->height;
    const std::size_t required_image_bytes = uv_offset +
        static_cast<std::size_t>(capture->bytes_per_line) * (capture->height / 2U);
    if (capture->size_image < required_image_bytes) {
        throw std::runtime_error("camera sizeimage is smaller than its negotiated NV12 planes");
    }

    v4l2_streamparm params{};
    params.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    params.parm.capture.timeperframe.numerator = 1;
    params.parm.capture.timeperframe.denominator = options.fps;
    ioctl_check(capture->fd, VIDIOC_S_PARM, &params, "VIDIOC_S_PARM(30fps)");
    capture->fps_numerator = params.parm.capture.timeperframe.numerator;
    capture->fps_denominator = params.parm.capture.timeperframe.denominator;
    if (capture->fps_numerator != 1 || capture->fps_denominator != options.fps) {
        throw std::runtime_error("camera did not accept the fixed frame rate");
    }

    hsa_status_t hsa_status = hsa_init();
    if (hsa_status != HSA_STATUS_SUCCESS) {
        throw std::runtime_error("hsa_init failed with status " + std::to_string(hsa_status));
    }
    capture->hsa_initialized = true;
    bool dmabuf_supported = false;
    hsa_status = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED, &dmabuf_supported);
    if (hsa_status != HSA_STATUS_SUCCESS || !dmabuf_supported) {
        throw std::runtime_error("HSA runtime does not support portable DMA-BUF export");
    }

    v4l2_requestbuffers request{};
    request.count = options.camera_buffers;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_DMABUF;
    ioctl_check(capture->fd, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS(DMABUF)");
    capture->v4l2_buffers_requested = true;
    if ((request.capabilities & V4L2_BUF_CAP_SUPPORTS_DMABUF) == 0 ||
        request.count < options.camera_buffers || request.count > 32) {
        throw std::runtime_error("camera did not provide the requested bounded DMA-BUF queue");
    }

    capture->camera_allocation_bytes = std::max(
        page_align(capture->size_image),
        kCameraExportAllocationFloor);
    capture->camera_buffers.reserve(request.count);
    for (std::uint32_t index = 0; index < request.count; ++index) {
        CameraBuffer buffer;
        buffer.allocation_bytes = capture->camera_allocation_bytes;
        hip_check(
            hipMalloc(&buffer.device_pointer, buffer.allocation_bytes),
            "hipMalloc(camera DMA-BUF)");
        int exported_fd = -1;
        std::uint64_t export_offset = 0;
        hsa_status = hsa_amd_portable_export_dmabuf(
            buffer.device_pointer,
            buffer.allocation_bytes,
            &exported_fd,
            &export_offset);
        if (hsa_status != HSA_STATUS_SUCCESS || exported_fd < 0) {
            if (exported_fd >= 0) {
                close(exported_fd);
            }
            static_cast<void>(hipFree(buffer.device_pointer));
            throw std::runtime_error(
                "HSA camera DMA-BUF export failed: " + hsa_error(hsa_status) +
                ", fd=" + std::to_string(exported_fd) +
                ", offset=" + std::to_string(export_offset));
        }
        if (export_offset != 0) {
            close(exported_fd);
            static_cast<void>(hipFree(buffer.device_pointer));
            throw std::runtime_error(
                "HSA camera DMA-BUF export returned non-zero offset=" +
                std::to_string(export_offset) +
                " for dedicated allocation_bytes=" +
                std::to_string(buffer.allocation_bytes));
        }
        buffer.dmabuf_fd = exported_fd;
        static_cast<void>(fcntl(exported_fd, F_SETFD, FD_CLOEXEC));
        capture->camera_buffers.push_back(buffer);
    }

    const std::size_t rgb_pitch = static_cast<std::size_t>(capture->width) * 3;
    const std::size_t rgb_bytes = rgb_pitch * capture->height;
    capture->clean_slots.reserve(options.clean_rgb_buffers);
    for (std::uint32_t index = 0; index < options.clean_rgb_buffers; ++index) {
        CleanSlot slot;
        slot.pitch = rgb_pitch;
        slot.allocation_bytes = rgb_bytes;
        hip_check(hipMalloc(&slot.device_pointer, rgb_bytes), "hipMalloc(clean RGB slot)");
        const hipError_t event_status =
            hipEventCreateWithFlags(&slot.ready, hipEventDisableTiming);
        if (event_status != hipSuccess) {
            static_cast<void>(hipFree(slot.device_pointer));
            slot.device_pointer = nullptr;
            hip_check(event_status, "hipEventCreate(clean RGB slot)");
        }
        capture->clean_slots.push_back(slot);
    }
    hip_check(
        hipStreamCreateWithFlags(&capture->ingest_stream, hipStreamNonBlocking),
        "hipStreamCreate(camera ingest)");

    for (std::uint32_t index = 0; index < capture->camera_buffers.size(); ++index) {
        queue_camera_buffer(capture, index);
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl_check(capture->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
    capture->streaming = true;
}

}  // namespace

extern "C" const char *vlm_camera_gpu_capture_last_error() {
    return last_error;
}

extern "C" int vlm_camera_nv12_to_rgb8(
    const void *source_nv12,
    std::size_t source_y_pitch,
    std::size_t source_uv_offset,
    std::size_t source_uv_pitch,
    int width,
    int height,
    void *destination_rgb8,
    std::size_t destination_pitch,
    void *source_ready_event,
    void *stream) {
    return launch_nv12_to_rgb8(
        source_nv12,
        source_y_pitch,
        source_uv_offset,
        source_uv_pitch,
        width,
        height,
        destination_rgb8,
        destination_pitch,
        source_ready_event,
        stream);
}

extern "C" int vlm_camera_gpu_capture_create(
    const vlm_camera_gpu_capture_options *options,
    vlm_camera_gpu_capture **capture_out) {
    if (options == nullptr || capture_out == nullptr || options->device == nullptr ||
        options->width == 0 || options->height == 0 || options->fps == 0 ||
        options->camera_buffers < 4 || options->clean_rgb_buffers < 3 ||
        options->camera_buffers > 32 || options->clean_rgb_buffers > 16) {
        set_error("create GPU camera", "invalid options or bounded pool sizes");
        return 1;
    }
    *capture_out = nullptr;
    auto *capture = new (std::nothrow) vlm_camera_gpu_capture();
    if (capture == nullptr) {
        set_error("create GPU camera", "allocation failed");
        return 2;
    }
    try {
        initialize_capture(capture, *options);
    } catch (const std::exception &error) {
        set_error("create GPU camera", error.what());
        destroy_capture(capture);
        return 3;
    }
    *capture_out = capture;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_gpu_capture_acquire_rgb8(
    vlm_camera_gpu_capture *capture,
    std::uint32_t timeout_ms,
    vlm_camera_gpu_frame *frame_out) {
    if (capture == nullptr || frame_out == nullptr || !capture->streaming || timeout_ms == 0) {
        set_error("acquire GPU camera frame", "invalid capture, output, or timeout");
        return 10;
    }
    CleanSlot *clean_slot = nullptr;
    std::uint32_t clean_index = 0;
    for (std::uint32_t index = 0; index < capture->clean_slots.size(); ++index) {
        if (!capture->clean_slots[index].in_use) {
            clean_slot = &capture->clean_slots[index];
            clean_index = index;
            break;
        }
    }
    if (clean_slot == nullptr) {
        capture->frames_dropped_no_clean_slot += 1;
        set_error("acquire GPU camera frame", "all bounded clean RGB slots are leased");
        return 1;
    }
    clean_slot->in_use = true;
    clean_slot->generation += 1;

    pollfd camera_poll{};
    camera_poll.fd = capture->fd;
    camera_poll.events = POLLIN | POLLPRI;
    int poll_status;
    do {
        poll_status = poll(&camera_poll, 1, static_cast<int>(timeout_ms));
    } while (poll_status < 0 && errno == EINTR);
    if (poll_status == 0) {
        clean_slot->in_use = false;
        set_error("acquire GPU camera frame", "poll timeout");
        return 2;
    }
    if (poll_status < 0 ||
        (camera_poll.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        clean_slot->in_use = false;
        if (poll_status < 0) {
            set_errno_error("poll camera");
        } else {
            set_error("poll camera", "device returned error flags");
        }
        return 11;
    }

    v4l2_buffer dequeued{};
    dequeued.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    dequeued.memory = V4L2_MEMORY_DMABUF;
    int dequeue_status;
    do {
        dequeue_status = ioctl(capture->fd, VIDIOC_DQBUF, &dequeued);
    } while (dequeue_status < 0 && errno == EINTR);
    if (dequeue_status < 0 || dequeued.index >= capture->camera_buffers.size()) {
        clean_slot->in_use = false;
        if (dequeue_status < 0) {
            set_errno_error("VIDIOC_DQBUF(DMABUF)");
        } else {
            set_error("VIDIOC_DQBUF(DMABUF)", "out-of-range buffer index");
        }
        return 12;
    }

    CameraBuffer &camera_buffer = capture->camera_buffers[dequeued.index];
    const std::size_t uv_offset =
        static_cast<std::size_t>(capture->bytes_per_line) * capture->height;
    const int conversion_status = launch_nv12_to_rgb8(
        camera_buffer.device_pointer,
        capture->bytes_per_line,
        uv_offset,
        capture->bytes_per_line,
        static_cast<int>(capture->width),
        static_cast<int>(capture->height),
        clean_slot->device_pointer,
        clean_slot->pitch,
        nullptr,
        capture->ingest_stream);
    if (conversion_status != 0) {
        clean_slot->in_use = false;
        try {
            queue_camera_buffer(capture, dequeued.index);
        } catch (...) {
        }
        return 13;
    }
    hipError_t hip_status = hipEventRecord(clean_slot->ready, capture->ingest_stream);
    if (hip_status == hipSuccess) {
        // This per-slot boundary is required before ISP may overwrite the NV12
        // buffer. It is not a device-wide synchronization.
        hip_status = hipEventSynchronize(clean_slot->ready);
    }
    if (hip_status != hipSuccess) {
        clean_slot->in_use = false;
        set_hip_error("camera ingest event", hip_status);
        try {
            queue_camera_buffer(capture, dequeued.index);
        } catch (...) {
        }
        return 14;
    }
    try {
        queue_camera_buffer(capture, dequeued.index);
    } catch (const std::exception &error) {
        clean_slot->in_use = false;
        set_error("requeue GPU camera buffer", error.what());
        return 15;
    }

    capture->frames_acquired += 1;
    capture->frames_requeued += 1;
    std::uint64_t captured_ns =
        static_cast<std::uint64_t>(dequeued.timestamp.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(dequeued.timestamp.tv_usec) * 1000ULL;
    if (captured_ns == 0) {
        captured_ns = monotonic_ns();
    }
    *frame_out = {
        capture->frames_acquired,
        captured_ns,
        clean_slot->generation,
        dequeued.sequence,
        clean_index,
        capture->width,
        capture->height,
        dequeued.bytesused,
        clean_slot->pitch,
        clean_slot->allocation_bytes,
        clean_slot->device_pointer,
        clean_slot->ready,
    };
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_gpu_capture_release_rgb8(
    vlm_camera_gpu_capture *capture,
    std::uint32_t slot_index,
    std::uint64_t generation,
    void *consumer_done_event) {
    if (capture == nullptr || slot_index >= capture->clean_slots.size()) {
        set_error("release GPU camera frame", "invalid capture or slot index");
        return 1;
    }
    CleanSlot &slot = capture->clean_slots[slot_index];
    if (!slot.in_use || slot.generation != generation) {
        set_error("release GPU camera frame", "stale or duplicate frame lease");
        return 2;
    }
    if (consumer_done_event != nullptr) {
        const hipError_t status =
            hipEventSynchronize(reinterpret_cast<hipEvent_t>(consumer_done_event));
        if (status != hipSuccess) {
            set_hip_error("consumer frame event", status);
            return 3;
        }
    }
    slot.in_use = false;
    std::snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

extern "C" int vlm_camera_gpu_capture_get_info(
    vlm_camera_gpu_capture *capture,
    vlm_camera_gpu_capture_info *info_out) {
    if (capture == nullptr || info_out == nullptr) {
        set_error("get GPU camera info", "invalid arguments");
        return 1;
    }
    *info_out = {
        capture->width,
        capture->height,
        capture->fourcc,
        capture->bytes_per_line,
        capture->size_image,
        capture->camera_allocation_bytes,
        capture->fps_numerator,
        capture->fps_denominator,
        static_cast<std::uint32_t>(capture->camera_buffers.size()),
        static_cast<std::uint32_t>(capture->clean_slots.size()),
        capture->hip_device,
        capture->frames_acquired,
        capture->frames_requeued,
        capture->frames_dropped_no_clean_slot,
        capture->driver.c_str(),
        capture->card.c_str(),
        "hipMalloc->HSA-DMA-BUF->V4L2-DMABUF->ISP-PRIME/GART",
        "HIP-NV12-to-RGB8-BT.601-limited",
    };
    return 0;
}

extern "C" void vlm_camera_gpu_capture_destroy(vlm_camera_gpu_capture *capture) {
    destroy_capture(capture);
}
