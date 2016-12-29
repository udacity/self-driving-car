cmake_minimum_required(VERSION 3.5)

project(utils)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include(../cmake_config/RoboAutoGlobal.cmake)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    cv_bridge
	image_descriptor
    tf2
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.55 REQUIRED system serialization) # system serialization

find_package(PkgConfig)
pkg_check_modules (libgps REQUIRED libgps)

catkin_package(
        INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS} ${roscpp_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS}
        LIBRARIES utils
        CATKIN_DEPENDS roscpp message_runtime
        DEPENDS system_lib eigen
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${roscpp_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${image_descriptor_INCLUDE_DIRS}
)

link_directories(
    ${roscpp_LIBRARY_DIRS}
    ${roslib_LIBRARY_DIRS}
    ${catkin_LIBRARY_DIRS}
)

add_library(utils
    src/Camera.cpp
        src/GpsCoords.cpp
		src/PoleDetector.cpp
    src/SignDetection.cpp
        src/Visualization.cpp
)

target_link_libraries(utils
    ${OpenCV_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    ${libgps_LIBRARIES}
    ${Boost_LIBRARIES}
)
