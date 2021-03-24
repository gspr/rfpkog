cmake_minimum_required(VERSION 3.12)

project(rfpkog VERSION 0.1.2
  DESCRIPTION "Really Fast Persistence Kernels On GPUs"
  LANGUAGES CXX
  )

include(TestBigEndian)
include(GNUInstallDirs)
test_big_endian(BIG_ENDIAN)

find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)

find_program(PANDOC pandoc)
if(PANDOC STREQUAL "PANDOC-NOTFOUND")
  set(PANDOC_FOUND 0)
else()
  message(STATUS "Pandoc found: ${PANDOC}")
  set(PANDOC_FOUND 1)
endif()

set(RFPKOG_DATADIR "${CMAKE_INSTALL_FULL_DATADIR}/rfpkog" CACHE PATH "Installation directory for data files and kernel source.")

add_executable(rfpkog
  src/cl_headers.hpp
  src/dtype.hpp
  src/main.cpp
  src/misc.cpp src/misc.hpp 
  src/options.cpp src/options.hpp
  src/pd.hpp
  src/rfpkog.hpp
  )

set_target_properties(rfpkog PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  )

target_compile_definitions(rfpkog PRIVATE KERNEL_FILENAME="${RFPKOG_DATADIR}/kernels.cl")
target_compile_definitions(rfpkog PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(rfpkog PRIVATE CL_HPP_TARGET_OPENCL_VERSION=120)
target_compile_definitions(rfpkog PRIVATE CL_HPP_CL_1_2_DEFAULT_BUILD)

if(BIG_ENDIAN)
  target_compile_definitions(rfpkog PRIVATE BIGENDIAN)
endif()
target_include_directories(rfpkog PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(rfpkog PRIVATE Threads::Threads OpenCL::OpenCL)

install(TARGETS rfpkog RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES src/kernels.cl DESTINATION "${RFPKOG_DATADIR}")

if(PANDOC_FOUND)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/man" COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/man")
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/man/rfpkog.1"
    COMMAND "${PANDOC}" --standalone --to man -o ${CMAKE_CURRENT_BINARY_DIR}/man/rfpkog.1 ${CMAKE_CURRENT_SOURCE_DIR}/man/rfpkog.1.md)
  add_custom_target(manpage ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/man" "${CMAKE_CURRENT_BINARY_DIR}/man/rfpkog.1")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/man/rfpkog.1" DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")
endif()

