cmake_minimum_required(VERSION 3.15)

project(magic)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${ENYO_LIBRARY_PATH}/include /opt/homebrew/include/)

add_executable(magic magic.cpp)

find_library(FMT_LIBRARY fmt)
if (FMT_LIBRARY)
    target_link_libraries(magic PRIVATE ${FMT_LIBRARY})
else()
    message(FATAL_ERROR "fmt library not found")
endif()

string(ASCII 27 Esc)
set(COLOR_RESET "${Esc}[0m")
set(COLOR_CYAN "${Esc}[36m")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} build: ${CMAKE_BUILD_TYPE}")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} flags: ${CMAKE_CXX_FLAGS}")
get_target_property(LINK_LIBRARIES ${PROJECT_NAME} LINK_LIBRARIES)
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} libs: ${LINK_LIBRARIES}")
