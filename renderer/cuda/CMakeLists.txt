cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
project(mip-plicits LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
# CUDA architecture. "86" (Ampere) embeds both SASS and PTX, so newer GPUs
# (Ada, Blackwell, ...) run it via JIT; set your native arch for the fastest
# first launch (e.g. -DMIP_CUDA_ARCH=120 on an RTX 5090). The GEMM template
# parameters in layer_gemms.cu were tuned with the CUTLASS profiler (see
# ../scripts); re-tuning for other GPUs is optional but recommended.
set(MIP_CUDA_ARCH "86" CACHE STRING "CUDA architecture(s) for all targets")

# Render/window resolution, compiled into the kernels because the CUTLASS GEMM
# tile shapes depend on it. Supported: 512 (default) and 1024.
set(MIP_RESOLUTION "512" CACHE STRING "Framebuffer resolution (512 or 1024)")

option(MIP_BUILD_TESTS "Build the GoogleTest unit tests" OFF)

function(set_common_target_configs target)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${MIP_CUDA_ARCH}
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
    )

    target_compile_definitions(${target} PRIVATE RESOLUTION=${MIP_RESOLUTION})

    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --relocatable-device-code=true
        --ptxas-options=-v
    >)

    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/imgui
        ${PROJECT_SOURCE_DIR}/tools/util/include
    )

    target_link_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/lib
    )

    target_link_libraries(${target} PRIVATE
        cudart
        freeglut
        glew64
    )

    include(FetchContent)

    FetchContent_Declare(
        glm
        GIT_REPOSITORY	https://github.com/g-truc/glm.git
        GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
    )

    FetchContent_MakeAvailable(glm)

    target_link_libraries(${target} PRIVATE glm::glm)

    # The executable expects data/ (checkpoints, from the data archive) and the
    # runtime DLLs next to it.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/data $<TARGET_FILE_DIR:${target}>/data
    )

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/lib/dll $<TARGET_FILE_DIR:${target}>
    )
endfunction()

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Main executable
add_executable(MIP-plicitsRenderer
    renderer.cu
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_impl_glut.cpp
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
)

set_common_target_configs("MIP-plicitsRenderer")

if(MIP_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(MIP-plicitsTests
        tests/tests.cu
    )

    set_common_target_configs("MIP-plicitsTests")

    target_link_libraries(MIP-plicitsTests PRIVATE gtest gtest_main)

    add_test(NAME MIP-plicitsTests COMMAND MIP-plicitsTests)
endif()
