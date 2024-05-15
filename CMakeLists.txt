cmake_minimum_required(VERSION 3.10)
project(FaceKit)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

set(lite.ai.toolkit_DIR /home/ubuntu/work/git/avatar/tool/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64)
find_package(lite.ai.toolkit REQUIRED PATHS ${lite.ai.toolkit_DIR})
if (lite.ai.toolkit_Found)
    message(STATUS "lite.ai.toolkit_INCLUDE_DIRS: ${lite.ai.toolkit_INCLUDE_DIRS}")
    message(STATUS "        lite.ai.toolkit_LIBS: ${lite.ai.toolkit_LIBS}")
    message(STATUS "   lite.ai.toolkit_LIBS_DIRS: ${lite.ai.toolkit_LIBS_DIRS}")
endif()

include_directories(3rd_party/util)
include_directories(src)
add_subdirectory(src)

add_executable(face_pipeline sample/face_pipeline.cpp)
target_link_libraries(face_pipeline src)