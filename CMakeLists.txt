cmake_minimum_required(VERSION 3.0.0)
project(camera_pose_estimation VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "${PROJECT_SOURCE_DIR}/get_pose" )

add_subdirectory(get_pose)
add_executable(camera_pose_estimation main.cpp)

target_link_libraries(camera_pose_estimation ${OpenCV_LIBS} )
target_link_libraries(camera_pose_estimation get_pose)