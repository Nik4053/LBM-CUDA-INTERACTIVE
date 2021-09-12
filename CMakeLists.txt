cmake_minimum_required(VERSION 3.10)

project(CUDA/OPENGL-SampleProject LANGUAGES CUDA CXX)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG)
    message("Debug build.")
else ()
    set(CMAKE_BUILD_TYPE Release)
    #add_definitions(-DNDEBUG)
    message("Release build.")
endif ()


SET(APP_EXE main)
if (HEADLESS)
    ADD_EXECUTABLE(${APP_EXE} ${APP_EXE}.cpp)
    message("Headless build.")
    target_sources(${APP_EXE} PRIVATE mainNoViz.cpp)
else ()
    ADD_EXECUTABLE(${APP_EXE} ${APP_EXE}.cpp)
    add_definitions(-DSIMVIZ)
    message("Visual build.")
    target_sources(${APP_EXE} PRIVATE mainViz.cpp)
endif()


IF(NOT HEADLESS)
    IF (UNIX)
        message("Unix build")
        find_package(OpenGL REQUIRED COMPONENTS OpenGL)
        target_link_libraries(${APP_EXE} OpenGL::OpenGL ${GLUT_LIBRARY})
        find_package(GLEW REQUIRED)
        target_link_libraries(${APP_EXE} GLEW::GLEW)
        target_link_libraries(${APP_EXE} glut)
        target_link_libraries(${APP_EXE} GLU)
    ELSEIF (WIN32)
        message("WIN32 build")
        include_directories(common/inc)
        target_link_libraries(${APP_EXE} "/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/lib/x64/freeglut.lib")
        target_link_libraries(${APP_EXE} "/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/lib/x64/glew64.lib")
    ENDIF ()
ENDIF()

ADD_LIBRARY(CUDAKERNEL
        kernel.cu
        )
set_target_properties(CUDAKERNEL PROPERTIES CUDA_ARCHITECTURES "53;60;61;62;70;72")
target_compile_options(CUDAKERNEL
        PRIVATE -Xptxas=-O3
        )


find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

target_link_libraries(${APP_EXE} CUDAKERNEL)

IF (WIN32)
    target_compile_options(${APP_EXE}
            PRIVATE /std:c++17 /Wall
            )
ELSEIF (UNIX)
    target_compile_options(${APP_EXE}
            PRIVATE -Wall -pedantic -std=c++17 -fmax-errors=5 -march=native -D_XOPEN_SOURCE=700
            )
endif ()
