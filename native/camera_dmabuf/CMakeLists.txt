cmake_minimum_required(VERSION 3.21)

project(camera_dmabuf_probe LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

add_executable(camera_dmabuf_probe camera_dmabuf_probe.hip)
find_library(HSA_RUNTIME_LIBRARY hsa-runtime64
  PATHS /opt/rocm/lib /opt/rocm/lib64
  REQUIRED)
target_link_libraries(camera_dmabuf_probe PRIVATE ${HSA_RUNTIME_LIBRARY})
target_compile_options(camera_dmabuf_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall;-Wextra>)

add_library(vlm_camera_gpu_capture SHARED camera_gpu_capture.hip)
target_link_libraries(vlm_camera_gpu_capture PRIVATE ${HSA_RUNTIME_LIBRARY})
target_compile_options(vlm_camera_gpu_capture PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall;-Wextra>)
set_target_properties(vlm_camera_gpu_capture PROPERTIES
  OUTPUT_NAME vlm_camera_gpu_capture
  VERSION 0.1.0
  SOVERSION 0)

install(TARGETS camera_dmabuf_probe vlm_camera_gpu_capture
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
install(FILES camera_gpu_capture.h DESTINATION include)
