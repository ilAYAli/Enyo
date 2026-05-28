cmake_minimum_required(VERSION 3.16)

project(enyo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENYO_USE_SYZYGY "Enable Syzygy tablebase probing via Pyrrhic" ON)

#---[ git versioning/output ]--------------------------------------------------
string(ASCII 27 Esc)
set(COLOR_RESET  "${Esc}[0m")
set(COLOR_CYAN   "${Esc}[36m")
set(COLOR_YELLOW "${Esc}[33m")

#---[ binary tagging ]---------------------------------------------------------
# Allow explicit version override via cache variables:
# cmake -DENYO_GIT_HASH=abc123 -DENYO_GIT_DIRTY=0 ..

if(NOT DEFINED ENYO_GIT_HASH)
    execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ENYO_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT ENYO_GIT_HASH)
        set(ENYO_GIT_HASH "unknown")
    endif()
endif()

if(NOT DEFINED ENYO_GIT_DIRTY)
    execute_process(
        COMMAND git diff --quiet --ignore-submodules HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_DIFF_RESULT
        ERROR_QUIET
    )
    if(GIT_DIFF_RESULT EQUAL 0)
        set(ENYO_GIT_DIRTY 0)
    else()
        set(ENYO_GIT_DIRTY 1)
    endif()
endif()

message(STATUS "Git hash: ${ENYO_GIT_HASH} dirty: ${ENYO_GIT_DIRTY}")

# build timestamp
string(TIMESTAMP BUILD_DATE "%d/%m-%y")

#---[ build version ]----------------------------------------------------------
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wfatal-errors")

# Set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release:
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG" CACHE STRING "Release flags" FORCE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -flto=auto -ffast-math -fno-pie")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -march=native -mtune=native -flto -ffast-math")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mdynamic-no-pic")
    endif()
endif()

# Debug:
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
    add_compile_definitions(CONSTEXPR_ASSERT=1)
endif()

# Profile:
# pprof --text ./build/enyo gmon.out
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    find_library(GPERFTOOLS_PROFILER_LIBRARY profiler HINTS /opt/homebrew/lib)
    find_path(GPERFTOOLS_INCLUDE_DIR gperftools/profiler.h HINTS /opt/homebrew/include)
    if(GPERFTOOLS_PROFILER_LIBRARY AND GPERFTOOLS_INCLUDE_DIR)
        message(STATUS "Profiler library found at: ${GPERFTOOLS_PROFILER_LIBRARY}")
        set(PROFILER_LIBRARY ${GPERFTOOLS_PROFILER_LIBRARY})
        include_directories(${GPERFTOOLS_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "gperftools profiler library not found. Make sure it's installed.")
    endif()
endif()

# enable propagation of CXX flags:
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} CACHE INTERNAL "cxx flags")

#---[ NNUE network ]-----------------------------------------------------------
add_compile_definitions(EVAL_FILE="${CMAKE_CURRENT_SOURCE_DIR}/net/default.net")

#---[ dependencies ]-----------------------------------------------------------
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

set(ENYO_CHESS_SOURCES
    src/fen.cpp
    src/board.cpp
    magic/magic.cpp
    src/pgn.cpp
    src/zobrist.cpp
)

set(ENYO_ENGINE_SOURCES
    src/main.cpp
    src/exepath.cpp
    src/thread.cpp
    src/uci.cpp
    src/nnue.cpp
    src/nnue_model.cpp
    src/nnue_model_board.cpp
    src/search.cpp
    src/probe.cpp
)

if(ENYO_USE_SYZYGY)
    if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/Pyrrhic/tbprobe.c" AND EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/Pyrrhic/tbprobe.h")
        add_library(pyrrhic_lib STATIC 3rdparty/Pyrrhic/tbprobe.c)
        target_include_directories(pyrrhic_lib PUBLIC 3rdparty/Pyrrhic)
    else()
        message(WARNING "ENYO_USE_SYZYGY is ON, but 3rdparty/Pyrrhic sources were not found. Building without Syzygy support.")
        set(ENYO_USE_SYZYGY OFF)
    endif()
endif()

#---[ targets ]----------------------------------------------------------------
add_custom_target(update_version_info
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DINPUT_FILE=${CMAKE_SOURCE_DIR}/version_info.hpp.in
        -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/version_info.hpp
        -DBUILD_DATE_VALUE=${BUILD_DATE}
        -DBUILD_TYPE_VALUE=${CMAKE_BUILD_TYPE}
        -DENYO_GIT_HASH=${ENYO_GIT_HASH}
        -DENYO_GIT_DIRTY=${ENYO_GIT_DIRTY}
        -P ${CMAKE_SOURCE_DIR}/cmake/update_version_info.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/version_info.hpp
    DEPENDS ${CMAKE_SOURCE_DIR}/version_info.hpp.in
            ${CMAKE_SOURCE_DIR}/cmake/update_version_info.cmake
    COMMENT "Generating version_info.hpp"
)

add_library(version_lib OBJECT src/version.cpp)
add_dependencies(version_lib update_version_info)
target_include_directories(version_lib PRIVATE ${CMAKE_BINARY_DIR})

add_library(enyo_chess STATIC
    ${ENYO_CHESS_SOURCES}
)

target_include_directories(enyo_chess PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(enyo_chess PUBLIC
    fmt::fmt-header-only
    nlohmann_json::nlohmann_json
)

add_executable(enyo
    ${ENYO_ENGINE_SOURCES}
)

target_compile_definitions(enyo PRIVATE ENYO_USE_SYZYGY=$<BOOL:${ENYO_USE_SYZYGY}>)
target_link_libraries(enyo PRIVATE enyo_chess version_lib)

if(WIN32)
    target_link_libraries(enyo PRIVATE ws2_32)
    target_link_options(enyo PRIVATE -Wl,--stack,67108864)
endif()

if(ENYO_USE_SYZYGY)
    target_link_libraries(enyo PRIVATE pyrrhic_lib)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_link_libraries(enyo PRIVATE ${PROFILER_LIBRARY})
endif()

target_include_directories(enyo PRIVATE
    ${CMAKE_BINARY_DIR}
    3rdparty/Pyrrhic)

option(BUILD_TESTS "Build the gtest target" ON)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)

    set(ENYO_TEST_SOURCES
        test/tests.cpp
        src/exepath.cpp
        src/thread.cpp
        src/uci.cpp
        src/nnue.cpp
        src/nnue_model.cpp
        src/nnue_model_board.cpp
        src/search.cpp
        src/probe.cpp
    )

    add_executable(test
        ${ENYO_TEST_SOURCES}
    )
    target_compile_definitions(test PRIVATE ENYO_USE_SYZYGY=$<BOOL:${ENYO_USE_SYZYGY}>)
    target_include_directories(test PRIVATE ${CMAKE_BINARY_DIR})
    target_link_libraries(test PRIVATE enyo_chess version_lib GTest::gtest)

    if(WIN32)
        target_link_libraries(test PRIVATE ws2_32)
        target_link_options(test PRIVATE -Wl,--stack,67108864)
    endif()

    if(ENYO_USE_SYZYGY)
        target_link_libraries(test PRIVATE pyrrhic_lib)
    endif()
endif()

#---[ install targets ]--------------------------------------------------------
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_SOURCE_DIR}
)

install(TARGETS enyo DESTINATION $ENV{HOME}/local/bin)
