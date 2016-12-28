cmake_minimum_required(VERSION 3.5)

project(image_descriptor)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include(../cmake_config/RoboAutoGlobal.cmake)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge
)

catkin_package(
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
        LIBRARIES image_descriptor
)

find_package(OpenCV REQUIRED)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${roscpp_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(image_descriptor
        src/Horizont.cpp
)

link_directories(
    ${roscpp_LIBRARY_DIRS}
    ${roslib_LIBRARY_DIRS}
    ${catkin_LIBRARY_DIRS}
)