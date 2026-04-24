execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND git diff --quiet --ignore-submodules HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE GIT_DIFF_RESULT
)

if(GIT_DIFF_RESULT EQUAL 0)
    set(GIT_DIRTY 0)
else()
    set(GIT_DIRTY 1)
endif()

set(BUILD_HASH "${GIT_HASH}")
set(BUILD_DATE "${BUILD_DATE_VALUE}")
set(BUILD_TYPE "${BUILD_TYPE_VALUE}")

configure_file("${INPUT_FILE}" "${OUTPUT_FILE}" @ONLY)
