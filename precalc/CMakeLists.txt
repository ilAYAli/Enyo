cmake_minimum_required(VERSION 3.12)

project(precalc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(/opt/homebrew/include/)
link_directories(/opt/homebrew/lib/)

add_executable(precalc precalc.cpp)
add_executable(test_precalc test_precalc.cpp)

target_link_libraries(precalc PRIVATE enyo_lib fmt)
target_link_libraries(test_precalc PRIVATE enyo_lib fmt)

target_include_directories(precalc PRIVATE ${CMAKE_SOURCE_DIR}/..)
target_include_directories(test_precalc PRIVATE ${CMAKE_SOURCE_DIR}/..)

#add_custom_command(
#    TARGET precalc
#    POST_BUILD
#    COMMAND $<TARGET_FILE:precalc> -o > ${CMAKE_SOURCE_DIR}/precalc/rook_attacks.hpp
#    COMMAND $<TARGET_FILE:precalc> -d > ${CMAKE_SOURCE_DIR}/precalc/bishop_attacks.hpp
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    DEPENDS precalc
#)

string(ASCII 27 Esc)
set(COLOR_RESET "${Esc}[0m")
set(COLOR_CYAN "${Esc}[36m")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} build: ${CMAKE_BUILD_TYPE}")
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} flags: ${CMAKE_CXX_FLAGS}")
get_target_property(LINK_LIBRARIES ${PROJECT_NAME} LINK_LIBRARIES)
message(STATUS "${COLOR_CYAN}${PROJECT_NAME}${COLOR_RESET} libs: ${LINK_LIBRARIES}")
