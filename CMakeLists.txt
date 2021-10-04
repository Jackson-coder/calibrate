cmake_minimum_required(VERSION 3.0.0)
project(calibrate VERSION 0.1.0)
set(OpenCV_DIR /home/opencv/build)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(calibrate main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries( calibrate ${OpenCV_LIBS} )