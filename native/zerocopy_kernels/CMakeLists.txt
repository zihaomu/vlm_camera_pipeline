cmake_minimum_required(VERSION 3.21)

project(vlm_camera_zerocopy_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

add_library(vlm_camera_zerocopy_kernels SHARED zerocopy_kernels.hip)
target_compile_options(vlm_camera_zerocopy_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall;-Wextra>)
set_target_properties(vlm_camera_zerocopy_kernels PROPERTIES
  OUTPUT_NAME vlm_camera_zerocopy_kernels
  VERSION 0.1.0
  SOVERSION 0)

install(TARGETS vlm_camera_zerocopy_kernels
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
