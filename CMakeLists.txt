cmake_minimum_required(VERSION 3.9)

project(enyo)

add_subdirectory(3rdparty/Fathom)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
find_package(fmt)
find_package(nlohmann_json REQUIRED)
include_directories(${NLOHMANN_JSON_INCLUDE_DIRS})

#---[ targets ]----------------------------------------------------------------
add_library(version OBJECT version.cpp)
target_compile_definitions(version PRIVATE
    BUILD_HASH="${GIT_HASH}"
    BUILD_DIRTY=${GIT_DIRTY}
    BUILD_DATE="${BUILD_DATE}"
    BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

add_executable(enyo
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

add_dependencies(enyo update_version)
target_link_libraries(enyo PRIVATE version fmt::fmt-header-only nlohmann_json::nlohmann_json fathom_lib)

if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_link_libraries(enyo PRIVATE ${PROFILER_LIBRARY})
endif()

target_include_directories(enyo PRIVATE ${CMAKE_BINARY_DIR} 3rdparty/Fathom/src)

#--- gtest
find_library(GTEST_LIBRARY gtest HINTS /opt/homebrew)
if(NOT GTEST_LIBRARY)
    message(FATAL_ERROR "Google Test library not found. Make sure it's installed or set GTEST_LIBRARY variable.")
endif()

# gtest:
add_executable(gtest
    test/gtest.cpp
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
target_include_directories(gtest PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(gtest PRIVATE fmt::fmt-header-only ${GTEST_LIBRARY} nlohmann_json::nlohmann_json)


#---[ target output ]----------------------------------------------------------
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} build: ${CMAKE_BUILD_TYPE}")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} flags: ${CMAKE_CXX_FLAGS}")
get_target_property(LINK_LIBRARIES ${PROJECT_NAME} LINK_LIBRARIES)
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} libs: ${LINK_LIBRARIES}")

#---[ install targets ]--------------------------------------------------------
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_SOURCE_DIR}
)

install(TARGETS enyo DESTINATION $ENV{HOME}/local/bin)
