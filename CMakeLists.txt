cmake_minimum_required(VERSION 3.16)

project(enyo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENYO_USE_SYZYGY "Enable Syzygy tablebase probing via Fathom" OFF)

#---[ git versioning/output ]--------------------------------------------------
string(ASCII 27 Esc)
set(COLOR_RESET  "${Esc}[0m")
set(COLOR_CYAN   "${Esc}[36m")
set(COLOR_YELLOW "${Esc}[33m")

#---[ binary tagging ]---------------------------------------------------------
function(git_info GIT_HASH_OUTPUT GIT_DIRTY_OUTPUT)
    execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND git diff --quiet --ignore-submodules HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_DIFF_RESULT
    )

    if(GIT_DIFF_RESULT EQUAL 0)
        set(GIT_DIRTY 0)
    else()
        set(GIT_DIRTY 1)
    endif()

    set(${GIT_HASH_OUTPUT} ${GIT_HASH} PARENT_SCOPE)
    set(${GIT_DIRTY_OUTPUT} ${GIT_DIRTY} PARENT_SCOPE)
endfunction()

git_info(GIT_HASH GIT_DIRTY)
message(STATUS "Git hash: ${GIT_HASH} dirty: ${GIT_DIRTY}")

# build timestamp
string(TIMESTAMP BUILD_DATE "%d/%m-%y")

# force update version.cpp to reflect the current git info
add_custom_target(update_version
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_SOURCE_DIR}/version.cpp
    COMMENT "Updating version"
)

#---[ build version ]----------------------------------------------------------
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wfatal-errors")

# Set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release:
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -march=native -mtune=native -flto -ffast-math")
    if(APPLE)
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
add_compile_definitions(EVAL_FILE="${CMAKE_CURRENT_SOURCE_DIR}/nnue/default.net")

#---[ dependencies ]-----------------------------------------------------------
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

set(ENYO_SOURCES
    main.cpp
    fen.cpp
    board.cpp
    exepath.cpp
    magic/magic.cpp
    thread.cpp
    uci.cpp
    pgn.cpp
    nnue.cpp
    search.cpp
    zobrist.cpp
    probe.cpp
)

if(ENYO_USE_SYZYGY)
    if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/Fathom/src/tbprobe.c" AND EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/Fathom/src/tbprobe.h")
        add_library(fathom_lib STATIC 3rdparty/Fathom/src/tbprobe.c)
        target_include_directories(fathom_lib PUBLIC 3rdparty/Fathom/src)
    else()
        message(WARNING "ENYO_USE_SYZYGY is ON, but 3rdparty/Fathom sources were not found. Building without Syzygy support.")
        set(ENYO_USE_SYZYGY OFF)
    endif()
endif()

#---[ targets ]----------------------------------------------------------------
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/version_info.hpp
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DINPUT_FILE=${CMAKE_SOURCE_DIR}/version_info.hpp.in
        -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/version_info.hpp
        -DBUILD_DATE_VALUE=${BUILD_DATE}
        -DBUILD_TYPE_VALUE=${CMAKE_BUILD_TYPE}
        -P ${CMAKE_SOURCE_DIR}/cmake/update_version_info.cmake
    DEPENDS ${CMAKE_SOURCE_DIR}/version_info.hpp.in
    COMMENT "Generating version_info.hpp"
)

add_custom_target(update_version_info DEPENDS ${CMAKE_BINARY_DIR}/version_info.hpp)
add_library(version_lib OBJECT version.cpp)
add_dependencies(version_lib update_version_info)
target_include_directories(version_lib PRIVATE ${CMAKE_BINARY_DIR})

add_executable(enyo
    ${ENYO_SOURCES}
)

add_dependencies(enyo update_version)
target_compile_definitions(enyo PRIVATE ENYO_USE_SYZYGY=$<BOOL:${ENYO_USE_SYZYGY}>)
target_link_libraries(enyo PRIVATE version_lib fmt::fmt-header-only nlohmann_json::nlohmann_json)

if(ENYO_USE_SYZYGY)
    target_link_libraries(enyo PRIVATE fathom_lib)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_link_libraries(enyo PRIVATE ${PROFILER_LIBRARY})
endif()

target_include_directories(enyo PRIVATE ${CMAKE_BINARY_DIR} 3rdparty/Fathom/src)

option(BUILD_TESTS "Build the gtest target" ON)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)

    set(ENYO_TEST_SOURCES
        test/tests.cpp
        fen.cpp
        board.cpp
        exepath.cpp
        magic/magic.cpp
        thread.cpp
        uci.cpp
        pgn.cpp
        nnue.cpp
        search.cpp
        zobrist.cpp
        probe.cpp
    )

    add_executable(test
        ${ENYO_TEST_SOURCES}
    )
    target_compile_definitions(test PRIVATE ENYO_USE_SYZYGY=$<BOOL:${ENYO_USE_SYZYGY}>)
    target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(test PRIVATE version_lib fmt::fmt-header-only GTest::gtest nlohmann_json::nlohmann_json)

    if(ENYO_USE_SYZYGY)
        target_link_libraries(test PRIVATE fathom_lib)
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
