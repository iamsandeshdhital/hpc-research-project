# Doxygen documentation for the HPC Research Project

find_package(Doxygen REQUIRED)

set(DOXYGEN_INPUT_DIR "${PROJECT_SOURCE_DIR}")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(DOXYGEN_PROJ_NAME    "${PROJECT_NAME}")
set(DOXYGEN_PROJ_VERSION "${PROJECT_VERSION}")
set(DOXYGEN_PROJ_BRIEF   "${PROJECT_DESCRIPTION}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY
)

add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)

# Doxygen exit codes: 0 = no warnings, 1 = warnings, 2 = warnings + errors.
# Treat warnings as errors so the docs cannot silently rot.
add_custom_target(docs-strict
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating documentation (warnings are errors)"
    VERBATIM
)
