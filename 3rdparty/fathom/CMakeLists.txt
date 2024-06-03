cmake_minimum_required(VERSION 3.15)
project(fathom)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${ENYO_LIBRARY_PATH}/include /opt/homebrew/include/ ${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(.)

include_directories(src)
add_library(fathom_lib STATIC src/tbprobe.c)
set_target_properties(fathom_lib PROPERTIES OUTPUT_NAME "fathom")
target_compile_definitions(fathom_lib PRIVATE USE_INCREMENTAL_NNUE=false)
