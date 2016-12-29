cmake_minimum_required(VERSION 3.5)

project(validator)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include(../cmake_config/RoboAutoGlobal.cmake)

find_package(catkin REQUIRED COMPONENTS
    utils
    motion
	roslib
)

find_package(PkgConfig)

catkin_package(
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${utils_INCLUDE_DIRS} ${motion_INCLUDE_DIRS}
    CATKIN_DEPENDS utils motion
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${utils_INCLUDE_DIRS}
	${motion_INCLUDE_DIRS}
)

add_executable(validator
    src/main_Validator.cpp
    src/Validator.cpp
)

add_executable(motion_validator
	src/main_MotionValidator.cpp
	src/MotionValidator.cpp
)

target_link_libraries(
    validator
    ${utils_LIBRARIES}
	${roscpp_LIBRARY_DIRS}
)

target_link_libraries(
	motion_validator
	${utils_LIBRARIES}
	${motion_LIBRARIES}
	${catkin_LIBRARIES}
	${roscpp_LIBRARIES}
	${roslib_LIBRARIES}
)
