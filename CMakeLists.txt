cmake_minimum_required(VERSION 3.12)

project(rfpkog VERSION 0.1
  DESCRIPTION "Really Fast Persistence Kernels On GPUs"
  LANGUAGES CXX
  )

include(TestBigEndian)
include(GNUInstallDirs)
test_big_endian(BIG_ENDIAN)

find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)

set(RFPKOG_DATADIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/rfpkog" CACHE STRING "Installation directory for data files.")

add_executable(rfpkog
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

target_compile_definitions(rfpkog PRIVATE KERNEL_FILENAME="${RFPKOG_DATADIR}/heat_kernel.cl")

if(BIG_ENDIAN)
  target_compile_definitions(rfpkog PRIVATE BIGENDIAN)
endif()
target_include_directories(rfpkog PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(rfpkog PRIVATE Threads::Threads OpenCL::OpenCL)

