cmake_minimum_required(VERSION 3.21)

project(vlm_camera_egl_present LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(EGL REQUIRED IMPORTED_TARGET egl)
pkg_check_modules(EPOXY REQUIRED IMPORTED_TARGET epoxy)
pkg_check_modules(FREETYPE REQUIRED IMPORTED_TARGET freetype2)
pkg_check_modules(FONTCONFIG REQUIRED IMPORTED_TARGET fontconfig)
pkg_check_modules(X11 REQUIRED IMPORTED_TARGET x11)
find_library(HSA_RUNTIME_LIBRARY hsa-runtime64
  PATHS /opt/rocm/lib /opt/rocm/lib64
  REQUIRED)

add_library(vlm_camera_egl_present SHARED egl_present.hip)
target_include_directories(vlm_camera_egl_present PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vlm_camera_egl_present PRIVATE
  PkgConfig::EGL
  PkgConfig::EPOXY
  PkgConfig::FREETYPE
  PkgConfig::FONTCONFIG
  PkgConfig::X11
  ${HSA_RUNTIME_LIBRARY})
target_compile_options(vlm_camera_egl_present PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall;-Wextra>)
set_target_properties(vlm_camera_egl_present PROPERTIES
  OUTPUT_NAME vlm_camera_egl_present
  VERSION 0.1.0
  SOVERSION 0)

install(TARGETS vlm_camera_egl_present
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES egl_present.h DESTINATION include)
