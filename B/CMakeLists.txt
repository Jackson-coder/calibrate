include_directories("${PROJECT_SOURCE_DIR}/calibration")

find_library(tt ${PROJECT_SOURCE_DIR}/A)

add_library(B b.cpp)
target_link_libraries(B A)