cmake_minimum_required(VERSION 3.5)

project(motion)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include(../cmake_config/RoboAutoGlobal.cmake)

find_package(catkin REQUIRED COMPONENTS
    utils
    roslib
    NeuralNetwork
)

find_package(PkgConfig)
pkg_check_modules (libgps REQUIRED libgps)

catkin_package(
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${utils_INCLUDE_DIRS}
    LIBRARIES motion
    CATKIN_DEPENDS utils NeuralNetwork
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${utils_INCLUDE_DIRS}
    ${NeuralNetwork_INCLUDE_DIRS}
)

add_library(motion
        src/Motion.cpp
)

add_executable(motionExe
    src/main.cpp
    src/Motion.cpp
    src/MotionLearner.cpp
        src/MotionTest.cpp
)

target_link_libraries(
    motionExe
    motion
    ${utils_LIBRARIES}
        ${roslib_LIBRARIES}
    ${NeuralNetwork_LIBRARIES}
        ${libgps_LIBRARIES}
        ${motion_LIBRARIES}
)
