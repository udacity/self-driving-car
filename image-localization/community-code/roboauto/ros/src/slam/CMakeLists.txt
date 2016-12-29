cmake_minimum_required(VERSION 3.5)

project(slam)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include(../cmake_config/RoboAutoGlobal.cmake)


find_package(catkin REQUIRED COMPONENTS
    utils
    image_descriptor
    cv_bridge
    roslib
    motion
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
find_package(Boost 1.55 REQUIRED system serialization) # system serialization
pkg_check_modules (libgps REQUIRED libgps)

catkin_package(
    INCLUDE_DIRS ${utils_INCLUDE_DIRS} ${image_descriptor_INCLUDE_DIRS}
    CATKIN_DEPENDS utils image_descriptor
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    ${utils_INCLUDE_DIRS}
    ${image_descriptor_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${motion_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

link_directories(
    ${roscpp_LIBRARY_DIRS}
    ${local_LIBRARY_DIRS}
)

add_library(slamLibrary STATIC
        src/SlamImpl.cpp
        src/Slam.cpp
        src/WeightFunction.cpp
        )

add_executable(slam
    src/main.cpp
)

add_executable(paramSolver
    src/paramSolver.cpp
)

add_executable(dataSetCreator
    src/dataSetCreator.cpp
)

add_executable(horizonValidator
    src/horizonValidator.cpp
)

add_executable(slam_validator
    src/validator.cpp
)
add_executable(gpx_exp
    src/gpxExporter.cpp
)

target_link_libraries(
        gpx_exp
        ${catkin_LIBRARIES}
        ${roscpp_LIBRARIES}
        ${roslib_LIBRARIES}
)

target_link_libraries(
        slam
        slamLibrary
        ${catkin_LIBRARIES}
        ${roscpp_LIBRARIES}
        ${roslib_LIBRARIES}
        ${EIGEN3_LIBRARIES}
        ${utils_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${motion_LIBRARIES}
        ${libgps_LIBRARIES}
        ${Boost_LIBRARIES}
)

target_link_libraries(
    paramSolver
    slamLibrary
    ${catkin_LIBRARIES}
    ${roscpp_LIBRARIES}
    ${roslib_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${utils_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${motion_LIBRARIES}
    ${libgps_LIBRARIES}
    ${Boost_LIBRARIES}
)

target_link_libraries(
    dataSetCreator
    slamLibrary
    ${catkin_LIBRARIES}
    ${roscpp_LIBRARIES}
    ${roslib_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${utils_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${motion_LIBRARIES}
    ${libgps_LIBRARIES}
    ${Boost_LIBRARIES}
)

target_link_libraries(
        horizonValidator
        slamLibrary
        ${catkin_LIBRARIES}
        ${roscpp_LIBRARIES}
        ${roslib_LIBRARIES}
        ${EIGEN3_LIBRARIES}
        ${utils_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${motion_LIBRARIES}
        ${libgps_LIBRARIES}
        ${Boost_LIBRARIES}
)

target_link_libraries(
    slam_validator
    slamLibrary
    ${catkin_LIBRARIES}
    ${roscpp_LIBRARIES}
    ${roslib_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${utils_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${motion_LIBRARIES}
    ${libgps_LIBRARIES}
    ${Boost_LIBRARIES}
)