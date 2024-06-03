cmake_minimum_required(VERSION 3.15)

project(replay)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${ENYO_LIBRARY_PATH}/include /opt/homebrew/include/ ${CMAKE_SOURCE_DIR})

add_executable(replay replay.cpp)

find_package(fmt)
target_link_libraries(replay PRIVATE fmt::fmt-header-only)

if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    if(APPLE)
        find_library(PROFILER_LIBRARY NAMES profiler HINTS /opt/homebrew/lib)
        if(NOT PROFILER_LIBRARY)
            message(FATAL_ERROR "Google CPU Profiler library not found.")
        endif()
        message(STATUS "Profiler library found at: ${PROFILER_LIBRARY}")
    endif()
    target_link_libraries(replay PRIVATE ${PROFILER_LIBRARY})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(replay PRIVATE enyo_lib)
else()
    target_link_libraries(replay PRIVATE enyo_lib)
endif()

string(ASCII 27 Esc)
set(COLOR_RESET "${Esc}[0m")
set(COLOR_CYAN "${Esc}[36m")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} build: ${CMAKE_BUILD_TYPE}")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} flags: ${CMAKE_CXX_FLAGS}")
get_target_property(LINK_LIBRARIES ${PROJECT_NAME} LINK_LIBRARIES)
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} libs: ${LINK_LIBRARIES}")
