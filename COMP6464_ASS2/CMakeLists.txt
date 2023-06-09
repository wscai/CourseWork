# CMake build system for cloth simulation
cmake_minimum_required(VERSION 3.12)
project(cloth VERSION 1.0.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# ##############################################################################
# enable C++  ##
# ##############################################################################
enable_language(CXX)

# ##############################################################################
# setup general C++ compilation flags#
# ##############################################################################

# add explictly debug symbols 
set(CMAKE_CXX_FLAGS " -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS " -msse2 -mavx -qopenmp")
# add g++ specific flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}  -fpic -Wall -Wextra -Wno-deprecated-declarations -Wpedantic"
  )
endif()

# add icc specific flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}   -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout"
  )
endif()

find_program(PAPI papi_avail)
if(PAPI)
    message("-- Found PAPI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpapi -DPAPI")
else()
    message("-- PAPI not found.")
endif()

# OpenGL
find_package(OpenGL REQUIRED)
message(STATUS "OpenGL included at ${OPENGL_INCLUDE_DIR}")

# GLUT
find_package(GLUT REQUIRED)
message(STATUS "GLUT included at ${GLUT_INCLUDE_DIR}")

# OpenMP
find_package(OpenMP)
message(STATUS "OpenMP included at ${OPENMP_INCLUDE_DIR}")

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

add_executable(
  opengl_main
  opengl_main.cpp
  cloth_code.cpp)

add_executable(
  kernel_main
  kernel_main.cpp
  cloth_code.cpp)

add_executable(
  kernel_opt
  kernel_main.cpp
  cloth_opt.cpp)

add_executable(
  kernel_sse
  kernel_main.cpp
  cloth_sse.cpp)
  
add_executable(
  kernel_vect_omp
  kernel_main.cpp
  cloth_vect_omp.cpp)

add_executable(
  kernel_omp
  kernel_main.cpp
  cloth_omp.cpp)

if(OpenMP_FOUND)
  target_link_libraries(kernel_main m OpenMP::OpenMP_CXX)
  target_link_libraries(opengl_main m ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}
                        OpenMP::OpenMP_CXX)
else()
  target_link_libraries(kernel_main m)
  target_link_libraries(opengl_main m ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
endif()
