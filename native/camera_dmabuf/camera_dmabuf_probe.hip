#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kCameraExportAllocationFloor = 4ULL * 1024ULL * 1024ULL;

struct Options {
    std::string device = "/dev/video0";
    std::string output;
    std::string mode = "driver-export";
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t frames = 8;
    uint32_t timeout_ms = 2000;
    uint32_t requested_buffers = 4;
};

struct Buffer {
    uint32_t index = 0;
    uint32_t length = 0;
    uint64_t fd_size = 0;
    uint64_t import_size = 0;
    int exported_fd = -1;
    hipExternalMemory_t external_memory = nullptr;
    void * device_ptr = nullptr;
    bool hsa_interop_mapping = false;
    bool hip_owned = false;
    uint64_t export_offset = 0;
    std::string import_backend;
    std::string hip_import_error;
};

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

std::string errno_message(const std::string & operation) {
    return operation + ": errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")";
}

void xioctl(int fd, unsigned long request, void * arg, const char * operation) {
    int rc;
    do {
        rc = ioctl(fd, request, arg);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        throw std::runtime_error(errno_message(operation));
    }
}

void hip_check(hipError_t rc, const char * operation) {
    if (rc != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": hip=" + std::to_string(static_cast<int>(rc)) +
            " (" + hipGetErrorString(rc) + ")");
    }
}

std::string hsa_error(hsa_status_t status) {
    const char * text = nullptr;
    if (hsa_status_string(status, &text) == HSA_STATUS_SUCCESS && text != nullptr) {
        return std::string(text);
    }
    return "hsa_status=" + std::to_string(static_cast<int>(status));
}

struct HsaAgentSearch {
    hsa_agent_t gpu{};
    bool found = false;
};

hsa_status_t find_gpu_agent(hsa_agent_t agent, void * data) {
    auto * search = static_cast<HsaAgentSearch *>(data);
    hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
    const hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    if (!search->found && type == HSA_DEVICE_TYPE_GPU) {
        search->gpu = agent;
        search->found = true;
    }
    return HSA_STATUS_SUCCESS;
}

std::string fourcc_string(uint32_t value) {
    char text[5] = {
        static_cast<char>(value & 0xff),
        static_cast<char>((value >> 8) & 0xff),
        static_cast<char>((value >> 16) & 0xff),
        static_cast<char>((value >> 24) & 0xff),
        '\0',
    };
    return std::string(text);
}

uint32_t parse_u32(const std::string & value, const char * name) {
    size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed, 10);
    if (consumed != value.size() || parsed > UINT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<uint32_t>(parsed);
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto next = [&](const char * name) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[i];
        };
        if (arg == "--device") {
            options.device = next("--device");
        } else if (arg == "--output") {
            options.output = next("--output");
        } else if (arg == "--mode") {
            options.mode = next("--mode");
        } else if (arg == "--width") {
            options.width = parse_u32(next("--width"), "--width");
        } else if (arg == "--height") {
            options.height = parse_u32(next("--height"), "--height");
        } else if (arg == "--frames") {
            options.frames = parse_u32(next("--frames"), "--frames");
        } else if (arg == "--timeout-ms") {
            options.timeout_ms = parse_u32(next("--timeout-ms"), "--timeout-ms");
        } else if (arg == "--buffers") {
            options.requested_buffers = parse_u32(next("--buffers"), "--buffers");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: camera_dmabuf_probe [--device /dev/video0] [--width 1280]"
                << " [--height 720] [--buffers 4] [--frames 8]"
                << " [--mode driver-export|gpu-export]"
                << " [--timeout-ms 2000] [--output report.json]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.width == 0 || options.height == 0 || options.frames == 0 ||
        options.requested_buffers < 2) {
        throw std::runtime_error("width/height/frames must be positive and buffers must be >= 2");
    }
    if (options.mode != "driver-export" && options.mode != "gpu-export") {
        throw std::runtime_error("--mode must be driver-export or gpu-export");
    }
    return options;
}

__global__ void checksum_kernel(const uint8_t * data, size_t size, unsigned long long * sum) {
    unsigned long long local = 0;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size;
         i += stride) {
        local += data[i];
    }
    atomicAdd(sum, local);
}

class Probe {
  public:
    explicit Probe(Options options) : options_(std::move(options)) {}

    ~Probe() {
        cleanup();
    }

    int run() {
        try {
            stage_ = "open";
            fd_ = open(options_.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
            if (fd_ < 0) {
                throw std::runtime_error(errno_message("open camera"));
            }

            query_and_configure();
            if (options_.mode == "gpu-export") {
                allocate_gpu_buffers_for_v4l2();
            } else {
                allocate_driver_buffers_and_import();
            }
            capture_and_checksum();
            passed_ = true;
            stage_ = "complete";
        } catch (const std::exception & error) {
            failure_ = error.what();
        }

        const std::string report = make_report();
        std::cout << report << '\n';
        if (!options_.output.empty()) {
            std::ofstream stream(options_.output, std::ios::out | std::ios::trunc);
            if (!stream) {
                std::cerr << "failed to open report path: " << options_.output << '\n';
                return 2;
            }
            stream << report << '\n';
        }
        return passed_ ? 0 : 1;
    }

  private:
    void query_and_configure() {
        stage_ = "querycap";
        v4l2_capability capability{};
        xioctl(fd_, VIDIOC_QUERYCAP, &capability, "VIDIOC_QUERYCAP");
        driver_ = reinterpret_cast<const char *>(capability.driver);
        card_ = reinterpret_cast<const char *>(capability.card);
        const uint32_t caps = capability.device_caps ? capability.device_caps : capability.capabilities;
        if ((caps & V4L2_CAP_VIDEO_CAPTURE) == 0 || (caps & V4L2_CAP_STREAMING) == 0) {
            throw std::runtime_error("camera lacks VIDEO_CAPTURE or STREAMING capability");
        }

        stage_ = "set_format";
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = options_.width;
        format.fmt.pix.height = options_.height;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        xioctl(fd_, VIDIOC_S_FMT, &format, "VIDIOC_S_FMT");
        width_ = format.fmt.pix.width;
        height_ = format.fmt.pix.height;
        pixel_format_ = format.fmt.pix.pixelformat;
        bytes_per_line_ = format.fmt.pix.bytesperline;
        size_image_ = format.fmt.pix.sizeimage;
        if (pixel_format_ != V4L2_PIX_FMT_NV12) {
            throw std::runtime_error("camera did not accept NV12; got " + fourcc_string(pixel_format_));
        }

        stage_ = "set_fps";
        v4l2_streamparm params{};
        params.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        params.parm.capture.timeperframe.numerator = 1;
        params.parm.capture.timeperframe.denominator = 30;
        xioctl(fd_, VIDIOC_S_PARM, &params, "VIDIOC_S_PARM");
        fps_numerator_ = params.parm.capture.timeperframe.numerator;
        fps_denominator_ = params.parm.capture.timeperframe.denominator;
    }

    void allocate_driver_buffers_and_import() {
        memory_type_ = V4L2_MEMORY_MMAP;
        stage_ = "request_buffers";
        v4l2_requestbuffers request{};
        request.count = options_.requested_buffers;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        xioctl(fd_, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS(MMAP)");
        request_capabilities_ = request.capabilities;
        allocated_buffers_ = request.count;
        if (request.count < 2) {
            throw std::runtime_error("driver allocated fewer than two V4L2 buffers");
        }

        buffers_.reserve(request.count);
        for (uint32_t index = 0; index < request.count; ++index) {
            stage_ = "query_buffer_" + std::to_string(index);
            v4l2_buffer query{};
            query.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            query.memory = V4L2_MEMORY_MMAP;
            query.index = index;
            xioctl(fd_, VIDIOC_QUERYBUF, &query, "VIDIOC_QUERYBUF");

            Buffer buffer;
            buffer.index = index;
            buffer.length = query.length;

            stage_ = "export_buffer_" + std::to_string(index);
            v4l2_exportbuffer export_buffer{};
            export_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            export_buffer.index = index;
            export_buffer.flags = O_CLOEXEC;
            xioctl(fd_, VIDIOC_EXPBUF, &export_buffer, "VIDIOC_EXPBUF");
            buffer.exported_fd = export_buffer.fd;
            ++exported_buffers_;

            const off_t seek_size = lseek(buffer.exported_fd, 0, SEEK_END);
            if (seek_size > 0) {
                buffer.fd_size = static_cast<uint64_t>(seek_size);
            } else {
                struct stat fd_stat{};
                if (fstat(buffer.exported_fd, &fd_stat) == 0 && fd_stat.st_size > 0) {
                    buffer.fd_size = static_cast<uint64_t>(fd_stat.st_size);
                }
            }
            const long page_size_raw = sysconf(_SC_PAGESIZE);
            const uint64_t page_size = page_size_raw > 0
                ? static_cast<uint64_t>(page_size_raw)
                : 4096ULL;
            const uint64_t page_aligned_length =
                (static_cast<uint64_t>(buffer.length) + page_size - 1) & ~(page_size - 1);
            buffer.import_size = buffer.fd_size > 0 ? buffer.fd_size : page_aligned_length;
            buffers_.push_back(buffer);
            Buffer & imported_buffer = buffers_.back();

            stage_ = "hip_import_buffer_" + std::to_string(index);
            const int import_fd = dup(imported_buffer.exported_fd);
            if (import_fd < 0) {
                throw std::runtime_error(errno_message("dup DMA-BUF fd"));
            }

            hipExternalMemoryHandleDesc handle_desc{};
            handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
            handle_desc.handle.fd = import_fd;
            handle_desc.size = imported_buffer.import_size;
            hipError_t import_rc = hipImportExternalMemory(
                &imported_buffer.external_memory, &handle_desc);
            if (import_rc == hipSuccess) {
                hipExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.offset = 0;
                buffer_desc.size = imported_buffer.import_size;
                const hipError_t map_rc = hipExternalMemoryGetMappedBuffer(
                    &imported_buffer.device_ptr,
                    imported_buffer.external_memory,
                    &buffer_desc);
                if (map_rc != hipSuccess) {
                    imported_buffer.hip_import_error = std::string("mapped-buffer: ") +
                        hipGetErrorString(map_rc);
                    static_cast<void>(hipDestroyExternalMemory(imported_buffer.external_memory));
                    imported_buffer.external_memory = nullptr;
                } else {
                    imported_buffer.import_backend = "hip_external_memory";
                    ++hip_imported_buffers_;
                }
            } else {
                close(import_fd);
                imported_buffer.hip_import_error = hipGetErrorString(import_rc);
            }

            if (imported_buffer.device_ptr == nullptr) {
                stage_ = "hsa_interop_import_buffer_" + std::to_string(index);
                ensure_hsa_agent();
                const hsa_handle_t interop_handle =
                    static_cast<hsa_handle_t>(imported_buffer.exported_fd);
                size_t mapped_size = 0;
                void * mapped_pointer = nullptr;
                const hsa_status_t map_status = hsa_amd_interop_map_buffer(
                    1,
                    &hsa_gpu_agent_,
                    interop_handle,
                    0,
                    &mapped_size,
                    &mapped_pointer,
                    nullptr,
                    nullptr);
                if (map_status != HSA_STATUS_SUCCESS) {
                    throw std::runtime_error(
                        "hipImportExternalMemory failed (" + imported_buffer.hip_import_error +
                        "); hsa_amd_interop_map_buffer failed (" + hsa_error(map_status) + ")");
                }
                if (mapped_pointer == nullptr || mapped_size < imported_buffer.length) {
                    if (mapped_pointer != nullptr) {
                        static_cast<void>(hsa_amd_interop_unmap_buffer(mapped_pointer));
                    }
                    throw std::runtime_error("HSA interop returned an invalid pointer or size");
                }
                imported_buffer.device_ptr = mapped_pointer;
                imported_buffer.import_size = mapped_size;
                imported_buffer.hsa_interop_mapping = true;
                imported_buffer.import_backend = "hsa_amd_interop_map_buffer";
                ++hsa_imported_buffers_;
            }
        }
    }

    void allocate_gpu_buffers_for_v4l2() {
        memory_type_ = V4L2_MEMORY_DMABUF;
        ensure_hsa_agent();

        bool dmabuf_supported = false;
        const hsa_status_t info_status = hsa_system_get_info(
            HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED, &dmabuf_supported);
        if (info_status != HSA_STATUS_SUCCESS) {
            throw std::runtime_error(
                "HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED: " + hsa_error(info_status));
        }
        hsa_dmabuf_supported_ = dmabuf_supported;
        if (!hsa_dmabuf_supported_) {
            throw std::runtime_error("HSA runtime reports DMA-BUF export unsupported");
        }

        stage_ = "request_dmabuf_buffers";
        v4l2_requestbuffers request{};
        request.count = options_.requested_buffers;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_DMABUF;
        xioctl(fd_, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS(DMABUF)");
        request_capabilities_ = request.capabilities;
        allocated_buffers_ = request.count;
        if ((request.capabilities & V4L2_BUF_CAP_SUPPORTS_DMABUF) == 0) {
            throw std::runtime_error(
                "V4L2 queue did not report V4L2_BUF_CAP_SUPPORTS_DMABUF");
        }
        if (request.count < 2) {
            throw std::runtime_error("driver accepted fewer than two DMA-BUF queue slots");
        }

        const long page_size_raw = sysconf(_SC_PAGESIZE);
        const uint64_t page_size = page_size_raw > 0
            ? static_cast<uint64_t>(page_size_raw)
            : 4096ULL;
        const uint64_t allocation_size = std::max(
            (static_cast<uint64_t>(size_image_) + page_size - 1) & ~(page_size - 1),
            kCameraExportAllocationFloor);

        buffers_.reserve(request.count);
        for (uint32_t index = 0; index < request.count; ++index) {
            stage_ = "hip_allocate_buffer_" + std::to_string(index);
            Buffer buffer;
            buffer.index = index;
            buffer.length = static_cast<uint32_t>(allocation_size);
            buffer.import_size = allocation_size;
            buffer.hip_owned = true;
            hip_check(hipMalloc(&buffer.device_ptr, allocation_size), "hipMalloc camera buffer");
            buffers_.push_back(buffer);
            Buffer & exported_buffer = buffers_.back();

            stage_ = "hsa_export_buffer_" + std::to_string(index);
            int dmabuf_fd = -1;
            uint64_t export_offset = 0;
            const hsa_status_t export_status = hsa_amd_portable_export_dmabuf(
                exported_buffer.device_ptr,
                allocation_size,
                &dmabuf_fd,
                &export_offset);
            if (export_status != HSA_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "hsa_amd_portable_export_dmabuf: " + hsa_error(export_status));
            }
            exported_buffer.exported_fd = dmabuf_fd;
            exported_buffer.export_offset = export_offset;
            static_cast<void>(fcntl(dmabuf_fd, F_SETFD, FD_CLOEXEC));
            ++exported_buffers_;
            ++hsa_exported_buffers_;

            const off_t seek_size = lseek(dmabuf_fd, 0, SEEK_END);
            if (seek_size > 0) {
                exported_buffer.fd_size = static_cast<uint64_t>(seek_size);
            }
            if (export_offset != 0) {
                throw std::runtime_error(
                    "HSA-exported camera buffer has non-zero DMA-BUF offset " +
                    std::to_string(export_offset));
            }
            if (exported_buffer.fd_size > 0 && exported_buffer.fd_size < allocation_size) {
                throw std::runtime_error("HSA-exported DMA-BUF is smaller than camera payload");
            }
            exported_buffer.import_backend = "hip_allocation_exported_to_v4l2";
        }
    }

    void ensure_hsa_agent() {
        if (hsa_initialized_) {
            return;
        }
        const hsa_status_t init_status = hsa_init();
        if (init_status != HSA_STATUS_SUCCESS) {
            throw std::runtime_error("hsa_init: " + hsa_error(init_status));
        }
        hsa_initialized_ = true;
        HsaAgentSearch search;
        const hsa_status_t iterate_status = hsa_iterate_agents(find_gpu_agent, &search);
        if (iterate_status != HSA_STATUS_SUCCESS || !search.found) {
            throw std::runtime_error(
                "cannot find HSA GPU agent: " + hsa_error(iterate_status));
        }
        hsa_gpu_agent_ = search.gpu;
    }

    void capture_and_checksum() {
        stage_ = "allocate_checksum";
        hip_check(
            hipMalloc(reinterpret_cast<void **>(&device_sum_), sizeof(unsigned long long)),
            "hipMalloc checksum");
        hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate");

        stage_ = "initialize_gpu_buffers";
        for (const Buffer & buffer : buffers_) {
            const unsigned char pattern = static_cast<unsigned char>(0xa5U - (buffer.index & 0x0fU));
            hip_check(
                hipMemsetAsync(
                    buffer.device_ptr,
                    pattern,
                    std::min<size_t>(size_image_, buffer.length),
                    stream_),
                "hipMemsetAsync camera buffer sentinel");
        }
        hip_check(hipStreamSynchronize(stream_), "hipStreamSynchronize camera sentinels");
        for (const Buffer & buffer : buffers_) {
            initial_checksums_.push_back(checksum_device(
                buffer.device_ptr, std::min<size_t>(size_image_, buffer.length)));
        }

        for (const Buffer & buffer : buffers_) {
            v4l2_buffer queue{};
            queue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            queue.memory = memory_type_;
            queue.index = buffer.index;
            if (memory_type_ == V4L2_MEMORY_DMABUF) {
                queue.m.fd = buffer.exported_fd;
                queue.length = buffer.length;
            }
            xioctl(fd_, VIDIOC_QBUF, &queue, "VIDIOC_QBUF(initial)");
        }

        stage_ = "stream_on";
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
        streaming_ = true;

        for (uint32_t frame = 0; frame < options_.frames; ++frame) {
            stage_ = "poll_frame_" + std::to_string(frame);
            pollfd poll_fd{};
            poll_fd.fd = fd_;
            poll_fd.events = POLLIN | POLLPRI;
            int poll_rc;
            do {
                poll_rc = poll(&poll_fd, 1, static_cast<int>(options_.timeout_ms));
            } while (poll_rc < 0 && errno == EINTR);
            if (poll_rc == 0) {
                throw std::runtime_error("camera poll timeout");
            }
            if (poll_rc < 0) {
                throw std::runtime_error(errno_message("poll camera"));
            }
            if ((poll_fd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                throw std::runtime_error("camera poll returned error flags " +
                                         std::to_string(poll_fd.revents));
            }

            stage_ = "dequeue_frame_" + std::to_string(frame);
            v4l2_buffer dequeued{};
            dequeued.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            dequeued.memory = memory_type_;
            xioctl(fd_, VIDIOC_DQBUF, &dequeued, "VIDIOC_DQBUF");
            if (dequeued.index >= buffers_.size()) {
                throw std::runtime_error("driver returned out-of-range buffer index");
            }
            const Buffer & buffer = buffers_[dequeued.index];
            const size_t valid_bytes = dequeued.bytesused > 0
                ? std::min<size_t>(dequeued.bytesused, buffer.length)
                : std::min<size_t>(size_image_, buffer.length);
            if (valid_bytes == 0) {
                throw std::runtime_error("camera returned an empty frame");
            }

            stage_ = "gpu_checksum_frame_" + std::to_string(frame);
            const unsigned long long host_sum = checksum_device(buffer.device_ptr, valid_bytes);
            checksums_.push_back(host_sum);
            if (host_sum != 0 && host_sum != initial_checksums_[dequeued.index]) {
                ++frames_changed_from_sentinel_;
            }
            sequences_.push_back(dequeued.sequence);
            bytes_used_.push_back(valid_bytes);

            stage_ = "requeue_frame_" + std::to_string(frame);
            if (memory_type_ == V4L2_MEMORY_DMABUF) {
                dequeued.m.fd = buffer.exported_fd;
                dequeued.length = buffer.length;
            }
            xioctl(fd_, VIDIOC_QBUF, &dequeued, "VIDIOC_QBUF(requeue)");
        }
        frames_sampled_ = static_cast<uint32_t>(checksums_.size());
        if (frames_changed_from_sentinel_ != frames_sampled_) {
            throw std::runtime_error(
                "one or more GPU-visible frames are zero or unchanged from the sentinel");
        }
    }

    unsigned long long checksum_device(const void * pointer, size_t size) {
        hip_check(
            hipMemsetAsync(device_sum_, 0, sizeof(unsigned long long), stream_),
            "hipMemsetAsync checksum");
        hipLaunchKernelGGL(
            checksum_kernel,
            dim3(256),
            dim3(256),
            0,
            stream_,
            static_cast<const uint8_t *>(pointer),
            size,
            device_sum_);
        hip_check(hipGetLastError(), "checksum_kernel launch");
        unsigned long long host_sum = 0;
        hip_check(
            hipMemcpyAsync(
                &host_sum,
                device_sum_,
                sizeof(host_sum),
                hipMemcpyDeviceToHost,
                stream_),
            "hipMemcpyAsync checksum scalar");
        hip_check(hipStreamSynchronize(stream_), "hipStreamSynchronize checksum");
        return host_sum;
    }

    void cleanup() noexcept {
        if (streaming_ && fd_ >= 0) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        if (fd_ >= 0 && allocated_buffers_ > 0) {
            v4l2_requestbuffers request{};
            request.count = 0;
            request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            request.memory = memory_type_;
            static_cast<void>(ioctl(fd_, VIDIOC_REQBUFS, &request));
            allocated_buffers_ = 0;
        }
        if (stream_ != nullptr) {
            static_cast<void>(hipStreamDestroy(stream_));
            stream_ = nullptr;
        }
        if (device_sum_ != nullptr) {
            static_cast<void>(hipFree(device_sum_));
            device_sum_ = nullptr;
        }
        for (Buffer & buffer : buffers_) {
            if (buffer.hip_owned && buffer.exported_fd >= 0) {
                close(buffer.exported_fd);
                buffer.exported_fd = -1;
            }
            if (buffer.device_ptr != nullptr) {
                if (buffer.hsa_interop_mapping) {
                    static_cast<void>(hsa_amd_interop_unmap_buffer(buffer.device_ptr));
                } else {
                    static_cast<void>(hipFree(buffer.device_ptr));
                }
                buffer.device_ptr = nullptr;
            }
            if (buffer.external_memory != nullptr) {
                static_cast<void>(hipDestroyExternalMemory(buffer.external_memory));
                buffer.external_memory = nullptr;
            }
            if (buffer.exported_fd >= 0) {
                close(buffer.exported_fd);
                buffer.exported_fd = -1;
            }
        }
        buffers_.clear();
        if (hsa_initialized_) {
            static_cast<void>(hsa_shut_down());
            hsa_initialized_ = false;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    std::string make_report() const {
        std::ostringstream out;
        out << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"passed\": " << (passed_ ? "true" : "false") << ",\n"
            << "  \"zero_host_copy_scope\": \"camera DMA-BUF export/import probe\",\n"
            << "  \"diagnostic_scalar_d2h_bytes_per_frame\": 8,\n"
            << "  \"device\": \"" << json_escape(options_.device) << "\",\n"
            << "  \"mode\": \"" << json_escape(options_.mode) << "\",\n"
            << "  \"driver\": \"" << json_escape(driver_) << "\",\n"
            << "  \"card\": \"" << json_escape(card_) << "\",\n"
            << "  \"format\": {\"fourcc\": \"" << json_escape(fourcc_string(pixel_format_))
            << "\", \"width\": " << width_ << ", \"height\": " << height_
            << ", \"bytes_per_line\": " << bytes_per_line_
            << ", \"size_image\": " << size_image_ << "},\n"
            << "  \"fps\": {\"numerator\": " << fps_numerator_
            << ", \"denominator\": " << fps_denominator_ << "},\n"
            << "  \"requested_buffers\": " << options_.requested_buffers << ",\n"
            << "  \"allocated_buffers\": " << allocated_buffers_ << ",\n"
            << "  \"request_buffer_capabilities\": " << request_capabilities_ << ",\n"
            << "  \"v4l2_supports_dmabuf\": "
            << ((request_capabilities_ & V4L2_BUF_CAP_SUPPORTS_DMABUF) ? "true" : "false")
            << ",\n"
            << "  \"hsa_dmabuf_supported\": "
            << (hsa_dmabuf_supported_ ? "true" : "false") << ",\n"
            << "  \"exported_buffers\": " << exported_buffers_ << ",\n"
            << "  \"hsa_exported_buffers\": " << hsa_exported_buffers_ << ",\n"
            << "  \"hip_imported_buffers\": " << hip_imported_buffers_ << ",\n"
            << "  \"hsa_interop_imported_buffers\": " << hsa_imported_buffers_ << ",\n"
            << "  \"buffers\": [";
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (i != 0) out << ", ";
            out << "{\"index\": " << buffers_[i].index
                << ", \"query_length\": " << buffers_[i].length
                << ", \"fd_size\": " << buffers_[i].fd_size
                << ", \"import_size\": " << buffers_[i].import_size
                << ", \"export_offset\": " << buffers_[i].export_offset
                << ", \"exported\": " << (buffers_[i].exported_fd >= 0 ? "true" : "false")
                << ", \"import_backend\": \""
                << json_escape(buffers_[i].import_backend) << "\""
                << ", \"hip_import_error\": \""
                << json_escape(buffers_[i].hip_import_error) << "\"}";
        }
        out << "],\n"
            << "  \"frames_requested\": " << options_.frames << ",\n"
            << "  \"frames_sampled\": " << frames_sampled_ << ",\n"
            << "  \"frames_changed_from_sentinel\": "
            << frames_changed_from_sentinel_ << ",\n"
            << "  \"initial_gpu_checksums\": [";
        for (size_t i = 0; i < initial_checksums_.size(); ++i) {
            if (i != 0) out << ", ";
            out << initial_checksums_[i];
        }
        out << "],\n"
            << "  \"sequences\": [";
        for (size_t i = 0; i < sequences_.size(); ++i) {
            if (i != 0) out << ", ";
            out << sequences_[i];
        }
        out << "],\n  \"bytes_used\": [";
        for (size_t i = 0; i < bytes_used_.size(); ++i) {
            if (i != 0) out << ", ";
            out << bytes_used_[i];
        }
        out << "],\n  \"gpu_checksums\": [";
        for (size_t i = 0; i < checksums_.size(); ++i) {
            if (i != 0) out << ", ";
            out << checksums_[i];
        }
        out << "],\n"
            << "  \"failure_stage\": \"" << json_escape(passed_ ? "" : stage_) << "\",\n"
            << "  \"failure\": \"" << json_escape(failure_) << "\"\n"
            << "}";
        return out.str();
    }

    Options options_;
    int fd_ = -1;
    bool streaming_ = false;
    bool passed_ = false;
    std::string stage_ = "not_started";
    std::string failure_;
    std::string driver_;
    std::string card_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t pixel_format_ = 0;
    uint32_t bytes_per_line_ = 0;
    uint32_t size_image_ = 0;
    uint32_t fps_numerator_ = 0;
    uint32_t fps_denominator_ = 0;
    uint32_t allocated_buffers_ = 0;
    uint32_t request_capabilities_ = 0;
    uint32_t exported_buffers_ = 0;
    uint32_t hip_imported_buffers_ = 0;
    uint32_t hsa_imported_buffers_ = 0;
    uint32_t hsa_exported_buffers_ = 0;
    uint32_t frames_sampled_ = 0;
    uint32_t frames_changed_from_sentinel_ = 0;
    std::vector<Buffer> buffers_;
    std::vector<unsigned long long> checksums_;
    std::vector<unsigned long long> initial_checksums_;
    std::vector<uint32_t> sequences_;
    std::vector<size_t> bytes_used_;
    hipStream_t stream_ = nullptr;
    unsigned long long * device_sum_ = nullptr;
    bool hsa_initialized_ = false;
    bool hsa_dmabuf_supported_ = false;
    hsa_agent_t hsa_gpu_agent_{};
    v4l2_memory memory_type_ = V4L2_MEMORY_MMAP;
};

}  // namespace

int main(int argc, char ** argv) {
    try {
        return Probe(parse_options(argc, argv)).run();
    } catch (const std::exception & error) {
        std::cerr << "camera_dmabuf_probe: " << error.what() << '\n';
        return 2;
    }
}
