cmake_minimum_required(VERSION 3.0.0)
project(tt VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories("${PROJECT_SOURCE_DIR}/B")

#add_subdirectory(B)
add_executable(tt main.cpp)

target_link_libraries(tt ${OpenCV_LIBS} )
#target_link_libraries(tt B)
